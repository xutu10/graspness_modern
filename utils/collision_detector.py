""" ModelFreeCollisionDetector
"""

import os
import sys
from typing import Dict
import numpy as np
import open3d as o3d

class ModelFreeCollisionDetector():
    """ Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
    """
    # Constant
    EPSILON = 1e-6

    # TODO, width and length of gripper base?
    def __init__(self, scene_points, finger_width=0.01, finger_length=0.06, bottom_width=0.05, bottom_length=0.02, voxel_size=0.005):
        self.finger_width = finger_width
        self.finger_length = finger_length
        self.bottom_width = bottom_width
        self.bottom_length = bottom_length
        self.voxel_size = voxel_size

        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def _check_data_valid(self, grasp_group, scene_points):
        """ Check whether the input grasp group is valid.
            Raise error if not valid.
        """
        if not hasattr(grasp_group, 'translations') or not hasattr(grasp_group, 'rotation_matrices'):
            raise ValueError('Input grasp group must have translations and rotation_matrices attributes.')
        if not hasattr(grasp_group, 'heights') or not hasattr(grasp_group, 'depths') or not hasattr(grasp_group, 'widths'):
            raise ValueError('Input grasp group must have heights, depths and widths attributes.')
        if grasp_group.translations.shape[0] != grasp_group.rotation_matrices.shape[0]:
            raise ValueError('Number of translations and rotation matrices must match.')
        if grasp_group.translations.shape[0] != grasp_group.heights.shape[0] or grasp_group.translations.shape[0] != grasp_group.depths.shape[0] or grasp_group.translations.shape[0] != grasp_group.widths.shape[0]:
            raise ValueError('Number of translations, heights, depths and widths must match.')
        if grasp_group.translations.shape[0] == 0:
            raise ValueError('Input grasp group is empty. Please provide valid grasp poses for collision detection.')

        """ Check whether the input scene points are valid.
            Raise error if not valid.
        """
        if scene_points.size == 0:
            raise ValueError('Scene points are empty. Please provide valid scene points for collision detection.')

    def _compute_mask_and_iou(self, targets, heights, depths, widths, empty_thresh) -> Dict[str, np.ndarray]:
        ## collision detection
        mask_z_height = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        mask_x_depth = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask_y_left_finger_min = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask_y_left_finger_max = (targets[:,:,1] < -widths/2)
        mask_y_right_finger_min = (targets[:,:,1] > widths/2)
        mask_y_right_finger_max = (targets[:,:,1] < (widths/2 + self.finger_width))
        # just set a lowerlimit so that it will not go to negative, it also can be set to 0?
        mask_x_bottom = ((targets[:,:,0] <= depths - self.finger_length) & (targets[:,:,0] > 0))
        mask_y_bottom_min = (targets[:,:,1] > -self.bottom_width/2)
        mask_y_bottom_max = (targets[:,:,1] < self.bottom_width/2)

        # get collision mask of each point
        left_mask = (mask_z_height & mask_x_depth & mask_y_left_finger_min & mask_y_left_finger_max)
        right_mask = (mask_z_height & mask_x_depth & mask_y_right_finger_min & mask_y_right_finger_max)
        bottom_mask = (mask_z_height & mask_y_bottom_min & mask_y_bottom_max & mask_x_bottom)
        global_mask = (left_mask | right_mask | bottom_mask)

        # calculate dynamic volume of each part
        vol_unit = 1.0 / (max(self.voxel_size, self.EPSILON) ** 3)
        left_right_volume = (heights * self.finger_length * self.finger_width * vol_unit).reshape(-1)
        bottom_volume = (heights * self.bottom_width * self.bottom_length * vol_unit).reshape(-1) 
        volume = left_right_volume*2 + bottom_volume

        # mask and volume for inner space detection
        inner_mask = mask_z_height & mask_x_depth & (~mask_y_left_finger_max) & (~mask_y_right_finger_min)
        inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)

        empty_mask = (inner_mask.sum(axis=-1) / np.maximum(inner_volume, self.EPSILON) < empty_thresh)
        global_iou = global_mask.sum(axis=1) / np.maximum(volume, self.EPSILON)
        left_iou = left_mask.sum(axis=1) / np.maximum(left_right_volume, self.EPSILON)
        right_iou = right_mask.sum(axis=1) / np.maximum(left_right_volume, self.EPSILON)
        bottom_iou = bottom_mask.sum(axis=1) / np.maximum(bottom_volume, self.EPSILON)

        return {
            "global_mask": global_mask, 
            "global_iou": global_iou,
            "empty_mask": empty_mask,
            "left_iou": left_iou, 
            "right_iou": right_iou, 
            "bottom_iou": bottom_iou,
            }

    def detect(self, grasp_group, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M]
                collision_thresh: [float]
                return_empty_grasp: [bool]
                empty_thresh: [float]
                return_ious: [bool]
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
    
        """
        self._check_data_valid(grasp_group, self.scene_points)

        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]

        # transform scene points to grasp frame, (M, N, 3)
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)
        
        mask_iou_dict = self._compute_mask_and_iou(targets, heights, depths, widths, empty_thresh)
        collision_mask = (mask_iou_dict["global_iou"] > collision_thresh)

        if not (return_empty_grasp or return_ious): 
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:        
            ret_value.append(mask_iou_dict["empty_mask"])
        if return_ious:
            ret_value.append([mask_iou_dict["global_iou"], mask_iou_dict["left_iou"], mask_iou_dict["right_iou"], mask_iou_dict["bottom_iou"]])
        return ret_value
