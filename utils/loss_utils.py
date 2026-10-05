"""
    Helper functions for label generation and ApproachNet.
"""

import torch
import numpy as np

GRASP_MAX_WIDTH = 0.10  # TODO: Cange to 0.08 or .07 for training of PANDA deployment model
GRASPNESS_THRESHOLD = 0.1  # Threshold for graspness filtering
NUM_VIEW = 300  
NUM_ANGLE = 12  
NUM_DEPTH = 4
M_POINT = 1024


def transform_point_cloud(cloud, transform, format='4x4'):
    """Apply a rotation to an (N, 3) point cloud.

    For translation formats, either a (3, 4) or (4, 4) matrix is
    accepted.
    """
   
    if format not in ('3x3', '3x4', '4x4'):
        raise ValueError(
            "Unknown transformation format; expected '3x3', '3x4', or '4x4'."
        )
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f'cloud must have shape (N, 3), got {tuple(cloud.shape)}')

    transformed = cloud @ transform[:3, :3].T
    if format != '3x3':
        transformed = transformed + transform[:3, 3]

    return transformed


def generate_grasp_views(N=300, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3), r=1):
    """Sample N points on a sphere using a Fibonacci lattice.

    Returns a float32 tensor with shape (N, 3).
    """
    
    indices = np.arange(N, dtype=np.float64)
    z = (2 * indices + 1) / N - 1
    radial = np.sqrt(1 - z ** 2)
    phase = 2 * indices * np.pi * phi
    views = np.stack(
        (radial * np.cos(phase), radial * np.sin(phase), z),
        axis=-1,
    )
    views = r * views + center

    return torch.from_numpy(views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_template_view, batch_angle):
    """Convert template views and in-plane angles to rotation matrices."""
    
    if batch_template_view.ndim != 2 or batch_template_view.shape[1] != 3:
        raise ValueError(
            f'ndim of batch_template_view must be 2 and second dimension must be 3, got {tuple(batch_template_view.shape)}'
        )
    if batch_angle.ndim != 1 or batch_angle.shape[0] != batch_template_view.shape[0]:
        raise ValueError(
            f'batch_angle must have same shape as batch_template_view first dimension, got {tuple(batch_angle.shape)}'
        )
    
    axis_x_norm = torch.linalg.vector_norm(batch_template_view, dim=-1, keepdim=True)
    if torch.any(axis_x_norm == 0):
        raise ValueError('batch_template_view vectors must be non-zero')
    axis_x = batch_template_view / axis_x_norm

    zeros = torch.zeros_like(axis_x[:, 0])
    axis_y = torch.stack((-axis_x[:, 1], axis_x[:, 0], zeros), dim=-1)
    # handle the case where axis_x is aligned with the z-axis to avoid division by zero(No need due to Fibonacci lattice)
    mask_y = torch.linalg.vector_norm(axis_y, dim=-1) == 0
    axis_y[mask_y, 1] = 1
    axis_y = axis_y / torch.linalg.vector_norm(axis_y, dim=-1, keepdim=True)    
    axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)

    sin_angle = torch.sin(batch_angle)
    cos_angle = torch.cos(batch_angle)
    ones = torch.ones_like(cos_angle)
    in_plane_rotation = torch.stack(
        (ones, zeros, zeros, zeros, cos_angle, -sin_angle,
         zeros, sin_angle, cos_angle),
        dim=-1,
    ).reshape(-1, 3, 3)
    basis = torch.stack((axis_x, axis_y, axis_z), dim=-1)

    return basis @ in_plane_rotation

