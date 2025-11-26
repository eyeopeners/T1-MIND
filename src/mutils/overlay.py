# src/mutils/overlay.py
# -*- coding: utf-8 -*-
"""
Overlay utilities:
- load_nifti: load NIfTI image (keeps affine)
- resample_label_to_target: resample an atlas/label image to a target T1 grid using nearest-neighbor, honoring affine
- slice_roi_cover: compute per-slice ROI coverage proportions
"""

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def load_nifti(path: str) -> nib.spatialimages.SpatialImage:
    """
    Load a NIfTI as a nibabel image object, preserving header/affine.
    """
    return nib.load(path)


def resample_label_to_target(label_src, target_src) -> np.ndarray:
    """
    Resample a label/atlas image to the grid (shape + affine) of the target image.
    - label_src: path or nibabel image for the atlas/label (integer labels)
    - target_src: path or nibabel image for the subject T1 (any scalar image)
    Returns:
        numpy int32 array with the same shape as target image data, aligned via affine.
    """
    label_img = label_src if isinstance(label_src, nib.spatialimages.SpatialImage) else nib.load(label_src)
    target_img = target_src if isinstance(target_src, nib.spatialimages.SpatialImage) else nib.load(target_src)

    # Nearest-neighbor to preserve integer labels; cval=0 for background
    res = resample_from_to(label_img, target_img, order=0, cval=0)
    arr = res.get_fdata()
    # Safety: ensure integer dtype
    return arr.astype(np.int32, copy=False)


def slice_roi_cover(overlay_3d: np.ndarray, n_rois: int = 360, axis: int = 2) -> np.ndarray:
    """
    Compute per-slice ROI coverage proportions for a 3D integer label map.
    Args:
        overlay_3d: int array of shape (X, Y, Z) already in the T1 grid.
        n_rois: number of cortical ROIs (e.g., 360 for HCP-MMP).
        axis: slicing axis (2=axial, 0=sagittal, 1=coronal)
    Returns:
        cov: (S, n_rois) array; cov[s, r] is the proportion of non-zero pixels
             in slice s that belong to ROI r+1 (labels are assumed 1..n_rois).
    """
    if overlay_3d.ndim != 3:
        raise ValueError(f"overlay_3d must be 3D, got shape {overlay_3d.shape}")

    S = overlay_3d.shape[axis]
    cov = np.zeros((S, n_rois), dtype=np.float32)

    for s in range(S):
        sl = np.take(overlay_3d, s, axis=axis)  # 2D integer label slice
        mask = sl > 0
        total = int(mask.sum())
        if total == 0:
            continue
        # bincount index 0 is background; labels 1..n_rois map to indices 1..n_rois
        binc = np.bincount(sl[mask].ravel(), minlength=n_rois + 1)
        # normalize to get proportions
        cov[s] = (binc[1:n_rois + 1] / float(total)).astype(np.float32, copy=False)

    return cov
