"""
Precompute point-wise graspness labels for GraspNet-1Billion.
Usage: python generate_graspness.py --dataset_root /data/GraspNet-1Billion --camera_type kinect
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'graspnetAPI'))

from utils.data_utils import (
    CameraInfo,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
    transform_point_cloud,
)
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUM_VIEWS = 300
NUM_ANGLES = 12
NUM_DEPTHS = 4
POINT_GRASP_NUM = NUM_VIEWS * NUM_ANGLES * NUM_DEPTHS  # 14,400
FRIC_COEF_THRESH = 0.8
CAMERA_WIDTH = 1280.0
CAMERA_HEIGHT = 720.0
FRAMES_PER_SCENE = 256


# --------------------------------------------------------------------------- #
# Per-scene cache (loaded once, reused across all frames)
# --------------------------------------------------------------------------- #
class SceneCache:
    """Loads and caches all per-scene data that doesn't change across frames."""

    def __init__(self, dataset_root, scene_id, camera_type):
        self.dataset_root = dataset_root
        self.scene_id = scene_id
        self.camera_type = camera_type
        self.scene_name = f'scene_{scene_id:04d}'
        self.scene_dir = os.path.join(dataset_root, 'scenes', self.scene_name, camera_type)

        # Camera poses and table alignment — (256, 4, 4) and (4, 4)
        self.camera_poses = np.load(os.path.join(self.scene_dir, 'camera_poses.npy'))
        self.align_mat = np.load(os.path.join(self.scene_dir, 'cam0_wrt_table.npy'))

        # Collision labels — one array per object in the scene
        collision_path = os.path.join(
            dataset_root, 'collision_label', self.scene_name, 'collision_labels.npz'
        )
        collision_npz = np.load(collision_path)
        self.collision_dump = [collision_npz[f'arr_{j}'] for j in range(len(collision_npz))]

        # Grasp labels — lazy loaded per object
        self._grasp_label_cache = {}

    def get_grasp_label(self, obj_idx):
        """Load grasp labels for one object (cached)."""
        if obj_idx not in self._grasp_label_cache:
            path = os.path.join(
                self.dataset_root, 'grasp_label', f'{str(obj_idx).zfill(3)}_labels.npz'
            )
            label = np.load(path)
            self._grasp_label_cache[obj_idx] = (
                label['points'].astype(np.float32),
                label['scores'].astype(np.float32),
            )
        return self._grasp_label_cache[obj_idx]


def load_frame_data(scene_cache, ann_id):
    """Load depth, seg, meta, and camera intrinsics for one frame."""
    scene_dir = scene_cache.scene_dir
    depth = np.array(Image.open(os.path.join(scene_dir, 'depth', f'{ann_id:04d}.png')))
    seg = np.array(Image.open(os.path.join(scene_dir, 'label', f'{ann_id:04d}.png')))
    meta = scio.loadmat(os.path.join(scene_dir, 'meta', f'{ann_id:04d}.mat'))

    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera = CameraInfo(
        CAMERA_WIDTH, CAMERA_HEIGHT,
        intrinsic[0][0], intrinsic[1][1],
        intrinsic[0][2], intrinsic[1][2],
        factor_depth,
    )
    return depth, seg, camera


def build_scene_cloud(depth, seg, camera, trans):
    """Build and crop the scene point cloud. Returns (cloud_masked, mask)."""
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    depth_mask = depth > 0
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = depth_mask & workspace_mask
    return cloud[mask], mask


def compute_object_graspness(
    scene_cache,
    obj_idx,
    obj_pose,
    collision,
    camera_pose,
):
    """
    Compute per-point graspness for one object and transform to camera coords.

    Returns:
        points_cam: (Np, 3) grasp points in camera coordinates
        graspness:  (Np, 1) per-point graspness in [0, 1]
    """
    points_obj, scores = scene_cache.get_grasp_label(obj_idx)
    num_points = points_obj.shape[0]

    # Valid grasp: low friction, positive score, no collision
    valid_mask = (scores <= FRIC_COEF_THRESH) & (scores > 0) & ~collision
    valid_mask = valid_mask.reshape(num_points, -1)
    graspness = valid_mask.sum(axis=1) / POINT_GRASP_NUM

    # object → world → camera
    points_world = transform_point_cloud(points_obj, obj_pose, '4x4')
    points_cam = transform_point_cloud(points_world, np.linalg.inv(camera_pose), '4x4')

    return points_cam, graspness.reshape(num_points, 1)


def match_cloud_to_grasp(cloud_masked, grasp_points, graspness):
    """
    For each scene cloud point, find its nearest grasp point and inherit its graspness.

    Uses cKDTree for fast KNN.
    """
    if grasp_points.shape[0] == 0:
        return np.zeros((cloud_masked.shape[0], 1), dtype=np.float32)

    tree = cKDTree(grasp_points)
    _, nn_inds = tree.query(cloud_masked, k=1, workers=-1)
    return graspness[nn_inds].astype(np.float32)


def process_frame(scene_cache, ann_id):
    """Process one frame: compute and save graspness."""
    depth, seg, camera = load_frame_data(scene_cache, ann_id)
    trans = np.dot(scene_cache.align_mat, scene_cache.camera_poses[ann_id])
    cloud_masked, _ = build_scene_cloud(depth, seg, camera, trans)

    # Load object poses for this frame
    scene_reader = xmlReader(
        os.path.join(scene_cache.scene_dir, 'annotations', f'{ann_id:04d}.xml')
    )
    pose_vectors = scene_reader.getposevectorlist()
    obj_list, pose_list = get_obj_pose_list(scene_cache.camera_poses[ann_id], pose_vectors)

    if not obj_list:
        return np.zeros((cloud_masked.shape[0], 1), dtype=np.float32)

    # Compute graspness for all objects in this frame
    all_points, all_graspness = [], []
    for i, (obj_idx, obj_pose) in enumerate(zip(obj_list, pose_list)):
        collision = scene_cache.collision_dump[i]
        points_cam, graspness = compute_object_graspness(
            scene_cache, obj_idx, obj_pose, collision, scene_cache.camera_poses[ann_id]
        )
        all_points.append(points_cam)
        all_graspness.append(graspness)

    grasp_points = np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float32)
    graspness = np.vstack(all_graspness) if all_graspness else np.zeros((0, 1), dtype=np.float32)

    # KNN: propagate graspness to scene cloud points
    cloud_graspness = match_cloud_to_grasp(cloud_masked, grasp_points, graspness)
    
    # Min-max normalize to [0, 1]
    lo, hi = cloud_graspness.min(), cloud_graspness.max()
    if hi > lo:
        return (cloud_graspness - lo) / (hi - lo)

    return cloud_graspness


def process_scene(args):
    """Process all frames in one scene."""
    scene_id, dataset_root, camera_type, ann_start, ann_end, save_root = args

    scene_cache = SceneCache(dataset_root, scene_id, camera_type)
    save_dir = os.path.join(save_root, f'scene_{scene_id:04d}', camera_type)
    os.makedirs(save_dir, exist_ok=True)

    for ann_id in range(ann_start, ann_end):
        graspness = process_frame(scene_cache, ann_id)
        # Save as float16 (graspness is in [0,1])
        np.save(
            os.path.join(save_dir, f'{ann_id:04d}.npy'),
            graspness.astype(np.float16),
        )

    return scene_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--camera_type', default='kinect', choices=['kinect', 'realsense'])
    parser.add_argument('--scene_start', type=int, default=0, help='inclusive')
    parser.add_argument('--scene_end', type=int, default=100, help='exclusive; 100 (original) or 190 (full)')
    parser.add_argument('--ann_start', type=int, default=0, help='inclusive')
    parser.add_argument('--ann_end', type=int, default=FRAMES_PER_SCENE, help='exclusive')
    return parser.parse_args()


def main():
    cfgs = parse_args()
    save_root = os.path.join(cfgs.dataset_root, 'graspness_new')

    jobs = [
        (scene_id, cfgs.dataset_root, cfgs.camera_type, cfgs.ann_start, cfgs.ann_end, save_root)
        for scene_id in range(cfgs.scene_start, cfgs.scene_end)
    ]

    for job in tqdm(jobs, desc='Scenes'):
        process_scene(job)


if __name__ == '__main__':
    main()
