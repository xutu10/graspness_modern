"""GraspNet-1Billion dataset and sparse-collate utilities.

Public API
----------
- GraspNetDataset        : PyTorch Dataset for GraspNet-1Billion
- SceneAwareSampler      : sampler that groups samples by scene (cache-friendly)
- LazyGraspLabels        : on-demand grasp label loader with LRU cache
- load_grasp_labels      : eager loader (all ~21 GB in RAM)
- load_grasp_labels_lazy : lazy loader factory
- sample_floor_aware     : point sampler that down-weights floor points
- spconv_collate_fn      : collate function that voxelizes a batch for sparse CNNs
"""

import os
import collections.abc as container_abcs
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from utils.data_utils import (
    CameraInfo,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
    transform_point_cloud,
)
from utils.stable_score_utils import (
    check_stable_labels_available,
    ensure_stable_labels_exist,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGE_W = 1280.0
IMAGE_H = 720.0
WORKSPACE_OUTLIER = 0.02
MIN_POINTS_PER_OBJECT = 50         
COLLISION_CACHE_MAXSIZE = 10        # ~1.8 GB for 10 scenes
STABLE_CACHE_MAXSIZE = 20           # ~several GB for 20 objects
LAZY_LABEL_CACHE_MAXSIZE = 20


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class SceneAwareSampler(Sampler):
    """Yield all frames of one scene before moving to the next.

    This maximizes collision-label cache hits.

    Args:
        dataset : GraspNetDataset instance
        shuffle : if True, shuffle scenes and frames within scenes
        seed    : RNG seed for reproducibility (set per epoch for DDP)
    """

    def __init__(self, dataset, shuffle=True, seed=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.scene_to_indices = {}
        for idx, scene in enumerate(dataset.scenename):
            self.scene_to_indices[scene].append(idx)

        self.scenes = list(self.scene_to_indices.keys())

    def __iter__(self):
        if not self.shuffle:
            for scene in self.scenes:
                yield from self.scene_to_indices[scene]
            return

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch if self.seed is not None else self.epoch)

        for scene_idx in torch.randperm(len(self.scenes), generator=g).tolist():
            scene = self.scenes[scene_idx]
            indices = self.scene_to_indices[scene]
            for i in torch.randperm(len(indices), generator=g).tolist():
                yield indices[i]

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class LazyGraspLabels:
    """Load grasp labels on demand with a true LRU cache.

    Stores (points, widths, scores) per object. Picklable for multiprocessing DataLoader workers.
    """

    def __init__(self, root: str, cache_maxsize: int = LAZY_LABEL_CACHE_MAXSIZE):
        self.root = root
        self.cache_maxsize = cache_maxsize
        self._cache = {}

    def __getitem__(self, obj_name: int):
        if obj_name in self._cache:
            return self._cache[obj_name]

        path = os.path.join(
            self.root, "grasp_label_simplified",
            f"{str(obj_name - 1).zfill(3)}_labels.npz",
        )
        npz = np.load(path)
        result = (
            npz["points"].astype(np.float32),
            npz["width"].astype(np.float32),
            npz["scores"].astype(np.float32),
        )

        if len(self._cache) >= self.cache_maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
                
        self._cache[obj_name] = result
        return result


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _random_sample(n_available: int, num_points: int) -> np.ndarray:
    """Sample `num_points` indices, with replacement if n_available < num_points."""
    if n_available >= num_points:
        return np.random.choice(n_available, num_points, replace=False)
    idxs = np.arange(n_available)
    extra = np.random.choice(n_available, num_points - n_available, replace=True)
    return np.concatenate([idxs, extra], axis=0)


def sample_floor_aware(
    cloud: np.ndarray,
    trans: np.ndarray,
    num_points: int,
    floor_height: float = 0.01,
    floor_keep_ratio: float = 0.01,
) -> np.ndarray:
    """Sample points, down-weighting floor/table points.

    Args:
        cloud            : (N, 3) points in camera coordinates
        trans            : (4, 4) camera→table transform (Z = height in table frame)
        num_points       : number of points to sample
        floor_height     : points with table-Z below this are "floor"
        floor_keep_ratio : relative sampling weight for floor points

    Returns:
        idxs : (num_points,) indices into `cloud`
    """
    n = len(cloud)
    cloud_homo = np.hstack([cloud, np.ones((n, 1))])
    heights = (trans @ cloud_homo.T).T[:, 2] # height of table frame

    weights = np.ones(n)
    weights[heights < floor_height] = floor_keep_ratio
    probs = weights / weights.sum()

    if n >= num_points:
        return np.random.choice(n, num_points, replace=False, p=probs)

    idxs1 = np.arange(n)
    idxs2 = np.random.choice(n, num_points - n, replace=True, p=probs)
    return np.concatenate([idxs1, idxs2], axis=0)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class GraspNetDataset(Dataset):
    """GraspNet-1Billion dataset.

    Each sample is one (scene, frame) pair. Depending on `load_label`:
        * load_label=True  → get_data_label()  (training: cloud + labels)
        * load_label=False → get_data()        (inference: cloud only)
    """

    # ----------------------------------------------------------------- splits
    _SPLITS = {
        "train":             list(range(0, 100)),
        "train_reduced":     list(range(0, 95)),
        "val_train":         list(range(95, 100)),
        "train_all":         list(range(0, 190)),
        "val":               list(range(109, 110)),
        "test":              list(range(100, 190)),
        "test_seen":         list(range(100, 130)),
        "test_seen_single":  [126],
        "test_seen_mini":    [101, 115, 128],
        "test_similar":      list(range(130, 160)),
        "test_similar_mini": [131, 145, 158],
        "test_novel":        list(range(160, 190)),
        "test_novel_single": [180],
        "test_novel_mini":   [161, 175, 188],
        "test_train_mini":   [0, 50, 80],
    }


    def __init__(
        self,
        root: str,
        grasp_labels=None,
        camera: str = "kinect",
        split: str = "train",
        num_points: int = 20000,
        voxel_size: float = 0.005,
        remove_outlier: bool = True,
        augment: bool = False,
        load_label: bool = True,
        use_rgb: bool = False,
        enable_stable_score: bool = False,
        floor_sampling: bool = False,
        view_start: int = 0,
        view_end: int = 256,
        include_floor: bool = False,
        augment_translation: bool = True,
    ):
        assert num_points <= 300_000, "num_points too large"

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.include_floor = include_floor
        self.augment_translation = augment_translation
        self.remove_outlier = False if include_floor else remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.use_rgb = use_rgb
        self.enable_stable_score = enable_stable_score
        self.floor_sampling = floor_sampling

        # Stable-score label cache
        self._stable_labels_path = os.path.join(root, "stable_labels")
        #self._stable_labels_cache: Dict[int, np.ndarray] = {}
        self._stable_labels_cache = {}
        if enable_stable_score:
            all_exist, missing = check_stable_labels_available(root)
            if not all_exist:
                print(f"Stable labels missing for {missing} objects. Computing...")
                if not ensure_stable_labels_exist(root, verbose=True):
                    raise RuntimeError("Failed to compute stable labels (install trimesh)")

        # Collision label cache
        self._collision_cache: Dict[str, Dict[int, np.ndarray]] = {}
        # Scene IDs (list of 'scene_XXXX')
        if split not in self._SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.sceneIds = [f"scene_{sid:04d}" for sid in self._SPLITS[split]]

        # Per-scene data root (train vs. test_images/<split>)
        self.scene_data_roots = {s: self._scene_data_root(s) for s in self.sceneIds}

        # Flat path lists — one entry per sample
        self.depthpath: List[str] = []
        self.rgbpath: List[str] = []
        self.labelpath: List[str] = []
        self.metapath: List[str] = []
        self.scenename: List[str] = []
        self.frameid: List[int] = []
        self.graspnesspath: List[str] = []
        self._build_path_lists(camera, view_start, view_end)

    def _build_path_lists(self, camera: str, view_start: int, view_end: int):
        graspness_subdir = "graspness_full" if self.include_floor else "graspness"
        for scene in tqdm(self.sceneIds, desc="Building path lists"):
            cam_root = os.path.join(self.scene_data_roots[scene], scene, camera)
            for frame in range(view_start, view_end):
                fname = f"{frame:04d}"
                self.depthpath.append(os.path.join(cam_root, "depth", f"{fname}.png"))
                self.rgbpath.append(os.path.join(cam_root, "rgb", f"{fname}.png"))
                self.labelpath.append(os.path.join(cam_root, "label", f"{fname}.png"))
                self.metapath.append(os.path.join(cam_root, "meta", f"{fname}.mat"))
                self.graspnesspath.append(
                    os.path.join(self.root, graspness_subdir, scene, camera, f"{fname}.npy")
                )
                self.scenename.append(scene)
                self.frameid.append(frame)

    # ------------------------------------------------------------ path utils
    def _scene_data_root(self, scene: str) -> str:
        scene_id = int(scene.rsplit("_", 1)[-1])
        if scene_id < 100:
            return os.path.join(self.root, "scenes")
        if scene_id < 130:
            sub = "test_seen"
        elif scene_id < 160:
            sub = "test_similar"
        else:
            sub = "test_novel"
        return os.path.join(self.root, "test_images", sub)

    def _scene_camera_path(self, scene: str) -> str:
        return os.path.join(self.scene_data_roots[scene], scene, self.camera)

    def scene_list(self) -> List[str]:
        return self.scenename

    # --------------------------------------------------------------- caches
    def _load_collision_labels(self, scene: str) -> Dict[int, np.ndarray]:
        """LRU-cached collision labels for one scene."""
        if scene in self._collision_cache:
            value = self._collision_cache.pop(scene)
            self._collision_cache[scene] = value
            return value

        npz = np.load(os.path.join(self.root, "collision_label", scene, "collision_labels.npz"))
        collision = {i: npz[f"arr_{i}"] for i in range(len(npz))}

        if len(self._collision_cache) >= COLLISION_CACHE_MAXSIZE:
            self._collision_cache.pop(next(iter(self._collision_cache)))
        self._collision_cache[scene] = collision
        return collision

    def _load_stable_labels(self, obj_idx: int) -> Optional[np.ndarray]:
        """LRU-cached stable-score labels for one object (or None if unavailable)."""
        if not self.enable_stable_score:
            return None
        if obj_idx in self._stable_labels_cache:
            value = self._stable_labels_cache.pop(obj_idx)
            self._stable_labels_cache[obj_idx] = value
            return value

        path = os.path.join(
            self._stable_labels_path, f"{str(obj_idx - 1).zfill(3)}_stable.npz"
        )
        if not os.path.exists(path):
            return None

        labels = np.load(path)["stable"].astype(np.float32)  # (Np, V, A)
        if len(self._stable_labels_cache) >= STABLE_CACHE_MAXSIZE:
            self._stable_labels_cache.pop(next(iter(self._stable_labels_cache)))
        self._stable_labels_cache[obj_idx] = labels
        return labels


    def augment_data(
        self,
        point_clouds: np.ndarray,
        object_poses_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Random flip along YZ plane (50%)
        if np.random.random() > 0.5:
            flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            point_clouds = transform_point_cloud(point_clouds, flip, "3x3")
            object_poses_list = [flip @ p for p in object_poses_list]

        # Random rotation about Z-axis in [-30°, +30°]
        angle = np.random.uniform(-np.pi / 6, np.pi / 6)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        point_clouds = transform_point_cloud(point_clouds, rot, "3x3")
        object_poses_list = [rot @ p for p in object_poses_list]

        # Optional translation (disabled in the original paper)
        if self.augment_translation:
            t = np.array([
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.1, 0.2),
            ], dtype=np.float32)
            point_clouds = point_clouds + t
            for p in object_poses_list:
                p[:, 3] += t

        return point_clouds, object_poses_list

   
    def _build_cloud(
        self,
        index: int,
        with_seg_and_rgb: bool = False,
    ):
        """Load depth/seg/meta, build cloud, crop to workspace.

        Returns:
            if with_seg_and_rgb:
                (cloud_masked, seg_masked, rgb_masked, meta, trans)
                rgb_masked is None when use_rgb=False
            else:
                (cloud_masked, trans)
            trans is None when remove_outlier=False
        """
        scene = self.scenename[index]
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])

        intrinsic = meta["intrinsic_matrix"]
        camera = CameraInfo(
            IMAGE_W, IMAGE_H,
            intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
            meta["factor_depth"],
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        depth_mask = depth > 0
        trans = None
        if self.remove_outlier:
            cam_root = self._scene_camera_path(scene)
            camera_poses = np.load(os.path.join(cam_root, "camera_poses.npy"))
            align_mat = np.load(os.path.join(cam_root, "cam0_wrt_table.npy"))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(
                cloud, seg, trans=trans, organized=True, outlier=WORKSPACE_OUTLIER
            )
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask

        cloud_masked = cloud[mask]
        if not with_seg_and_rgb:
            return cloud_masked, trans

        seg_masked = seg[mask]
        rgb_masked = None
        if self.use_rgb:
            rgb = np.array(Image.open(self.rgbpath[index]))
            rgb_masked = rgb[mask]
        return cloud_masked, seg_masked, rgb_masked, meta, trans
    
    # no need to have sample_floor_aware and replace random_sample with graspness_aware_sampling
    def _sample_indices(self, cloud_masked: np.ndarray, trans: Optional[np.ndarray]) -> np.ndarray:
        if self.floor_sampling and trans is not None:
            return sample_floor_aware(cloud_masked, trans, self.num_points)
        return _random_sample(len(cloud_masked), self.num_points)


    def _finalize_cloud_sample(
        self,
        cloud_sampled: np.ndarray,
        rgb_sampled: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shift coords to be ≥ 0 and build features.

        Returns:
            cloud_sampled : (N, 3) shifted point cloud
            feats         : (N, 3) ones or RGB in [0, 1]
            offset        : (3,)   offset that was added to the cloud
        """
        offset = -cloud_sampled.min(axis=0)
        cloud_sampled = cloud_sampled + offset
        if self.use_rgb:
            feats = rgb_sampled.astype(np.float32) / 255.0
        else:
            feats = np.ones_like(cloud_sampled, dtype=np.float32)
        return cloud_sampled, feats, offset

    # ----------------------------------------------------------- dunder API
    def __len__(self) -> int:
        return len(self.depthpath)

    def __getitem__(self, index: int):
        return self.get_data_label(index) if self.load_label else self.get_data(index)

    # ------------------------------------------------------------ public API
    def get_data(self, index: int, return_raw_cloud: bool = False):
        if return_raw_cloud:
            cloud_masked, _ = self._build_cloud(index, with_seg_and_rgb=False)
            return cloud_masked

        cloud_masked, trans = self._build_cloud(index, with_seg_and_rgb=False)
        idxs = self._sample_indices(cloud_masked, trans)
        cloud_sampled = cloud_masked[idxs]

        rgb_sampled = None
        if self.use_rgb:
            _, _, rgb_masked, _, _ = self._build_cloud(index, with_seg_and_rgb=True)
            rgb_sampled = rgb_masked[idxs]
        else:
            rgb_sampled = None

        cloud_sampled, feats, offset = self._finalize_cloud_sample(cloud_sampled, rgb_sampled)

        return {
            "point_clouds": cloud_sampled.astype(np.float32),
            "coors": (cloud_sampled / self.voxel_size).astype(np.float32),
            "feats": feats,
            "cloud_offset": offset.astype(np.float32),
        }

    def get_data_label(self, index: int):
        cloud_masked, seg_masked, rgb_masked, meta, trans = self._build_cloud(
            index, with_seg_and_rgb=True
        )
        graspness_full = np.load(self.graspnesspath[index])

        idxs = self._sample_indices(cloud_masked, trans)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness_full[idxs]
        rgb_sampled = rgb_masked[idxs] if self.use_rgb else None

        # Objectness: 0 = background, 1 = any object
        objectness = seg_sampled.copy()
        objectness[objectness > 1] = 1

        # --- Per-object grasp labels -------------------------------------
        scene = self.scenename[index]
        collision_labels = self._load_collision_labels(scene) if self.load_label else None

        obj_idxs = meta["cls_indexes"].flatten().astype(np.int32)
        poses = meta["poses"]

        object_poses, grasp_points, grasp_widths, grasp_scores, grasp_stable = [], [], [], [], []

        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < MIN_POINTS_PER_OBJECT:
                continue

            object_poses.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = collision_labels[i]  # (Np, V, A, D)

            # Subsample grasp points: keep ≥ 300, ≤ Np
            n_keep = min(max(int(len(points) / 4), 300), len(points))
            sub = np.random.choice(len(points), n_keep, replace=False)

            grasp_points.append(points[sub])
            grasp_widths.append(widths[sub])

            scores_sub = scores[sub].copy()
            scores_sub[collision[sub]] = 0
            grasp_scores.append(scores_sub)

            if self.enable_stable_score:
                stable = self._load_stable_labels(obj_idx)
                if stable is None:
                    stable = np.zeros((len(sub), scores.shape[1], scores.shape[2]), dtype=np.float32)
                else:
                    stable = stable[sub]
                grasp_stable.append(stable)

        # --- Augmentation ------------------------------------------------
        if self.augment:
            cloud_sampled, object_poses = self.augment_data(cloud_sampled, object_poses)

        # --- Finalize ----------------------------------------------------
        cloud_sampled, feats, offset = self._finalize_cloud_sample(cloud_sampled, rgb_sampled)

        out = {
            "point_clouds":     cloud_sampled.astype(np.float32),
            "coors":            (cloud_sampled / self.voxel_size).astype(np.float32),
            "feats":            feats,
            "cloud_offset":     offset.astype(np.float32),
            "graspness_label":  graspness_sampled.astype(np.float32),
            "objectness_label": objectness.astype(np.int64),
            "object_poses_list": object_poses,
            "grasp_points_list": grasp_points,
            "grasp_widths_list": grasp_widths,
            "grasp_scores_list": grasp_scores,
        }
        if self.enable_stable_score and grasp_stable:
            out["grasp_stable_list"] = grasp_stable
        return out


# --------------------------------------------------------------------------- #
# Label loading
# --------------------------------------------------------------------------- #
def load_grasp_labels(root: str) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Eagerly load all grasp labels (~21 GB of RAM)."""
    labels = {}
    for obj_name in tqdm(range(1, 89), desc="Loading grasp labels"):
        path = os.path.join(
            root, "grasp_label_simplified",
            f"{str(obj_name - 1).zfill(3)}_labels.npz",
        )
        npz = np.load(path)
        labels[obj_name] = (
            npz["points"].astype(np.float32),
            npz["width"].astype(np.float32),
            npz["scores"].astype(np.float32),
        )
    return labels


def load_grasp_labels_lazy(root: str) -> LazyGraspLabels:
    """Lazy loader factory — returns a picklable LazyGraspLabels."""
    return LazyGraspLabels(root)


# --------------------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------------------
def _voxelize(list_data):
    """Merge the points of all samples into unique (batch, x, y, z) voxels with mean features.

    Returns voxel coordinates (M, 4), voxel features (M, C) and, for every input point,
    the index of the voxel it fell into."""
    coord_chunks, feat_chunks = [], []
    for b, sample in enumerate(list_data):
        c = torch.as_tensor(sample['coors']).int()            # coords are >= 0, so truncation == floor
        coord_chunks.append(torch.cat([c.new_full((c.shape[0], 1), b), c], dim=1))
        feat_chunks.append(torch.as_tensor(sample['feats'], dtype=torch.float32))

    coords = torch.cat(coord_chunks, dim=0).contiguous()
    feats = torch.cat(feat_chunks, dim=0).contiguous()

    voxel_coords, point2voxel = torch.unique(coords, dim=0, return_inverse=True)
    num_voxels = voxel_coords.shape[0]
    feat_sum = feats.new_zeros((num_voxels, feats.shape[1])).index_add_(0, point2voxel, feats)
    counts = torch.bincount(point2voxel, minlength=num_voxels).clamp(min=1)
    return voxel_coords, feat_sum / counts.unsqueeze(1).to(feat_sum.dtype), point2voxel


def _stack_field(values):
    """Batch one dictionary entry across samples."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values, 0)
    if type(first).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(v) for v in values], 0)
    if isinstance(first, container_abcs.Sequence):
        # per-sample lists stay lists of tensors
        return [[torch.from_numpy(x) if type(x).__module__ == 'numpy' else x for x in v] for v in values]
    return values


def spconv_collate_fn(list_data):
    """Collate for spconv: voxelize the whole batch and batch every other field. 'quantize2original' maps each point to its voxel."""
    voxel_coords, voxel_feats, point2voxel = _voxelize(list_data)
    batch = {'coors': voxel_coords, 'feats': voxel_feats, 'quantize2original': point2voxel}
    for key in list_data[0]:
        if key not in ('coors', 'feats'):
            batch[key] = _stack_field([d[key] for d in list_data])
    return batch
