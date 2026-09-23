""" Tools for data processing.
    Author: chenxi-wang
"""

import numpy as np
from scipy.spatial import cKDTree


class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """Project a depth image into camera coordinates.

    Organized output preserves the image layout, including invalid depth
    pixels. Unorganized output drops non-finite and non-positive depths.
    """
   
    if depth.shape != 2 or depth.shape[0] !=  camera.height or depth.shape[1] != camera.width:
        raise ValueError(f'depth must have shape ({camera.height}, {camera.width}), got {depth.shape}')
    if camera.scale == 0 or camera.fx == 0 or camera.fy == 0:
        raise ValueError('camera scale, fx, and fy must be non-zero')

    rows, cols = np.indices((int(camera.height), int(camera.width)))
    depth_m = depth / camera.scale
    cloud = np.empty((int(camera.height), int(camera.width), 3))
    cloud[..., 0] = (cols - camera.cx) * depth_m / camera.fx
    cloud[..., 1] = (rows - camera.cy) * depth_m / camera.fy
    cloud[..., 2] = depth_m

    valid = np.isfinite(depth_m) & (depth_m > 0)
    cloud = cloud[valid]

    if not organized:
        return cloud.reshape([-1,3])

    return cloud


def transform_point_cloud(cloud, transform, format='4x4'):
    """Apply a rigid transform to a set of 3D points.

    Supported transform shapes:
        - '3x3': rotation matrix only
        - '3x4': rotation + translation
        - '4x4': rotation + translation
    """
    cloud = np.asarray(cloud, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)

    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f'cloud must have shape (N, 3), got {cloud.shape}')
    if format not in ('3x3', '3x4', '4x4'):
        raise ValueError("Unknown transformation format, only support '3x3', '3x4', or '4x4'.")
    if format == '3x3' and transform.shape != (3, 3):
        raise ValueError(f'transform must have shape (3, 3) for format=\'3x3\', got {transform.shape}')
    if format in ('3x4', '4x4') and transform.shape not in ((3, 4), (4, 4)):
        raise ValueError(f'transform must have shape (3, 4) or (4, 4) for format=\'{format}\', got {transform.shape}')

    if format == '3x3':
        return (transform @ cloud.T).T

    ones = np.ones((cloud.shape[0], 1), dtype=np.float32)
    cloud_h = np.hstack((cloud, ones))
    transformed = (transform @ cloud_h.T).T
    return transformed[:, :3]


def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """Return grasp points that are within ``th`` of the scene cloud.

    Distance calculations are chunked to avoid allocating the full pairwise
    distance matrix for large scenes and grasp sets.
    """
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f'cloud must have shape (N, 3), got {cloud.shape}')
    if grasp_points.ndim != 2 or grasp_points.shape[1] != 3:
        raise ValueError(
            f'grasp_points must have shape (M, 3), got {grasp_points.shape}')
    if pose.shape not in ((3, 4), (4, 4)):
        raise ValueError(f'pose must have shape (3, 4) or (4, 4), got {pose.shape}')
   
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    tree = cKDTree(cloud)
    min_dists, _ = tree.query(grasp_points_trans, k=1, return_distance=True)
    visible_mask = min_dists < th

    return visible_mask


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """Keep points in the foreground workspace as a boolean mask.

    Args:
        cloud: scene points, either (H, W, 3) or (N, 3)
        seg: segmentation mask with same spatial shape as cloud for organized data,
            or a length-N label vector for flattened data
        trans: optional transform applied to points before computing the workspace
        organized: if True, preserve the original (H, W) mask layout
        outlier: padding added to the workspace bounds

    Returns:
        A boolean mask of shape (H, W) when organized=True, otherwise (N,)
    """
  
    if organized:
        if cloud.ndim != 3 or cloud.shape[2] != 3:
            raise ValueError(f'cloud must have shape (H, W, 3) when organized=True, got {cloud.shape}')
        if seg.shape != cloud.shape[:2]:
            raise ValueError(f'seg must have shape {cloud.shape[:2]}, got {seg.shape}')
        h,w,_ = cloud.shape
        cloud_flat = cloud.reshape(h*w, 3)
        seg_flat = seg.reshape(h*w)

    if trans is not None:
        cloud_flat = transform_point_cloud(cloud_flat, trans)

    foreground = cloud_flat[seg_flat > 0]
    if foreground.size == 0:
        if organized:
            return np.zeros((h, w), dtype=bool)
        return np.zeros(cloud_flat.shape[0], dtype=bool)

    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)

    mask_x = (cloud_flat[:, 0] >= xmin - outlier) & (cloud_flat[:, 0] <= xmax + outlier)
    mask_y = (cloud_flat[:, 1] >= ymin - outlier) & (cloud_flat[:, 1] <= ymax + outlier)
    mask_z = (cloud_flat[:, 2] >= zmin - outlier) & (cloud_flat[:, 2] <= zmax + outlier)
    workspace_mask = mask_x & mask_y & mask_z

    if organized:
        return workspace_mask.reshape(h, w)
    return workspace_mask
