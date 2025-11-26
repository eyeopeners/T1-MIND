
import numpy as np, nibabel as nib
from scipy.ndimage import zoom
def load_overlay(path: str):
    img = nib.load(path); return img.get_fdata().astype(np.int32), img.affine
def resample_nn(arr, target_shape):
    if tuple(arr.shape)==tuple(target_shape): return arr
    scale = [t/o for t,o in zip(target_shape, arr.shape)]
    return zoom(arr, zoom=scale, order=0)
def slice_roi_cover(overlay_3d, n_rois=360, axis=2):
    S = overlay_3d.shape[axis]
    cov = np.zeros((S, n_rois), dtype=np.float32)
    for s in range(S):
        sl = np.take(overlay_3d, s, axis=axis)
        mask = sl>0; total = mask.sum()
        if total==0: continue
        binc = np.bincount(sl[mask].ravel(), minlength=n_rois+1)
        cov[s] = (binc[1:n_rois+1]/float(total)).astype(np.float32)
    return cov
def project_node_to_slice(w_node, cover):
    import numpy as np
    if cover.ndim==2:
        ps = cover @ w_node.T; ps = (ps/(ps.sum(axis=0, keepdims=True)+1e-6)).T; return ps
    else:
        ps = (cover * w_node[:,None,:]).sum(axis=2); return ps/(ps.sum(axis=1, keepdims=True)+1e-6)
