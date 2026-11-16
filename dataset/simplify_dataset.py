"""Simplify GraspNet-1Billion grasp labels.

The original `grasp_label/{id}_labels.npz` files store `offsets` as
(Np, V, A, D, 3). The network only uses the gripper width, which is the
Z-component of the offset (index 2). Dropping the X/Y offset channels
reduces each file to ~1/3 its original size with no loss of training signal.

Usage:
    python simplify_dataset.py --dataset_root /data/GraspNet-1Billion

Output:
    {dataset_root}/grasp_label_simplified/{id:03d}_labels.npz
        - points : (Np, 3)           float32
        - scores : (Np, V, A, D)     float32
        - width  : (Np, V, A, D)     float32
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

NUM_OBJECTS = 88
OFFSET_WIDTH_INDEX = 2  # offsets[..., 2] is the gripper width (Z)


def simplify_one_object(root: str, save_path: str, obj_id: int) -> bool:
    """Simplify one object's grasp label file.

    Returns True on success, False if the source file is missing.
    """
    src = os.path.join(root, "grasp_label", f"{obj_id:03d}_labels.npz")
    dst = os.path.join(save_path, f"{obj_id:03d}_labels.npz")

    if not os.path.exists(src):
        print(f"  [skip] {src} not found", file=sys.stderr)
        return False

    label = np.load(src)
    points = label["points"]                                   # (Np, 3)
    scores = label["scores"]                                   # (Np, V, A, D)
    width = label["offsets"][:, :, :, :, OFFSET_WIDTH_INDEX]   # (Np, V, A, D)

    np.savez(dst, points=points, scores=scores, width=width)
    return True


def simplify_grasp_labels(root: str, save_path: str) -> None:
    """Simplify all 88 objects' grasp label files."""
    os.makedirs(save_path, exist_ok=True)

    n_ok = 0
    for obj_id in tqdm(range(NUM_OBJECTS), desc="Simplifying grasp labels"):
        if simplify_one_object(root, save_path, obj_id):
            n_ok += 1

    print(f"\nDone: {n_ok}/{NUM_OBJECTS} objects simplified → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default=None, required=True)
    cfgs = parser.parse_args()
    root = cfgs.dataset_root  # set root and save path
    save_path = os.path.join(root, 'grasp_label_simplified')
    simplify_grasp_labels(root, save_path)
