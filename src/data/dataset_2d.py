# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.mutils.mind_io import read_mind_csv
from src.mutils.overlay import resample_label_to_target, slice_roi_cover


class SimpleCache:
    """ 简单的内存缓存（按 subject_id 缓存切片与覆盖），默认关闭 """
    def __init__(self, max_items=0):
        self.max_items = int(max_items)
        self.buff = {}
        self.order = []

    def get(self, key):
        if self.max_items <= 0:
            return None
        return self.buff.get(key, None)

    def put(self, key, value):
        if self.max_items <= 0:
            return
        if key in self.buff:
            self.buff[key] = value
            return
        if len(self.order) >= self.max_items:
            old = self.order.pop(0)
            self.buff.pop(old, None)
        self.order.append(key)
        self.buff[key] = value


class Neuro2DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, slices=64, axis='axial', img_size=224,
                 n_rois=426, fit_scaler=False, scaler: StandardScaler=None, race_vocab=None,
                 cache_max_items: int = 0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.slices = int(slices)
        assert self.slices > 0
        assert axis in ('axial', 'sagittal', 'coronal')
        self.axis = axis
        self.img_size = int(img_size)
        self.n_rois = int(n_rois)

        if race_vocab is None:
            race_vocab = [1,2,3,4,5,6,7]
        self.race_vocab = list(race_vocab)
        self.race_to_idx = {int(v): i for i, v in enumerate(self.race_vocab)}
        self.R = len(self.race_vocab)

        self.scaler = scaler if scaler is not None else StandardScaler()
        if fit_scaler:
            cov = self._cov_matrix(self.df)
            self.scaler.fit(cov)

        # overlay 配置
        self.use_overlay = False
        self.overlay_shared_path = None
        self._has_overlay_col = ("overlay_path" in self.df.columns)
        if self._has_overlay_col:
            uniq = self.df["overlay_path"].fillna("").astype(str).unique()
            uniq = [u for u in uniq if len(u) > 0]
            if len(uniq) == 1 and os.path.exists(uniq[0]):
                self.use_overlay = True
                self.overlay_shared_path = uniq[0]
            elif len(uniq) > 1:
                self.use_overlay = True
                self.overlay_shared_path = None

        # 可控缓存（默认 0 = 关闭）
        self.cache = SimpleCache(max_items=cache_max_items)

    def __len__(self): return len(self.df)

    def _load_nifti(self, path: str):
        img = nib.load(path)
        return img.get_fdata().astype(np.float32, copy=False), img

    def _resize2d(self, arr2d: np.ndarray, size: int) -> np.ndarray:
        x = torch.from_numpy(arr2d[None, None, ...])  # (1,1,H,W)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return x[0, 0].numpy()

    def _sample_slices(self, vol: np.ndarray):
        if self.axis == 'axial':
            S = vol.shape[2]; get = lambda i: vol[:, :, i]; axis = 2
        elif self.axis == 'sagittal':
            S = vol.shape[0]; get = lambda i: vol[i, :, :]; axis = 0
        else:
            S = vol.shape[1]; get = lambda i: vol[:, i, :]; axis = 1

        idx = np.linspace(0, S - 1, self.slices).round().astype(int)
        stack = []
        for i in idx:
            sl = get(int(i))
            nz = sl != 0
            m = float(sl[nz].mean() if nz.any() else sl.mean())
            s = float(sl[nz].std()  if nz.any() else sl.std()) + 1e-6
            sl = (sl - m) / s
            sl = self._resize2d(sl, self.img_size)
            stack.append(sl)
        arr = np.stack(stack, axis=0)  # (S,H,W)
        return arr, idx, axis

    def _cov_vector(self, row_like) -> np.ndarray:
        age = float(row_like["age"])
        if "edu_years" in row_like and pd.notnull(row_like["edu_years"]):
            edu = float(row_like["edu_years"])
        else:
            edu = float(row_like["edu"]) if "edu" in row_like and pd.notnull(row_like["edu"]) else 0.0
        try:
            sex = 1.0 if int(row_like["sex"]) == 1 else 0.0
        except Exception:
            sex = 0.0
        rv = np.zeros(self.R, dtype=np.float32)
        try:
            r = int(row_like["race"])
        except Exception:
            r = 7
        if r in self.race_to_idx:
            rv[self.race_to_idx[r]] = 1.0
        elif 7 in self.race_to_idx:
            rv[self.race_to_idx[7]] = 1.0
        return np.concatenate([[age, edu, sex], rv]).astype(np.float32, copy=False)

    def _cov_matrix(self, df_like: pd.DataFrame) -> np.ndarray:
        return np.stack([self._cov_vector(row) for _, row in df_like.iterrows()], axis=0)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        sid = str(row.subject_id)
        cached = self.cache.get(sid)
        if cached is None:
            t1_path = str(row.t1w_path)
            vol, t1_img = self._load_nifti(t1_path)
            img_slices, idx, axis = self._sample_slices(vol)
            cover = np.zeros((self.slices, self.n_rois), dtype=np.float32)
            if self.use_overlay:
                if self.overlay_shared_path is not None:
                    ov_src = self.overlay_shared_path
                else:
                    ov_src = ""
                    if self._has_overlay_col and pd.notnull(row.overlay_path):
                        ov_src = str(row.overlay_path)
                if ov_src and os.path.exists(ov_src):
                    ov_arr = resample_label_to_target(ov_src, t1_img)
                    cover_full = slice_roi_cover(ov_arr, self.n_rois, axis=axis)
                    cover = cover_full[idx]
            cached = (img_slices, cover)
            self.cache.put(sid, cached)
        else:
            img_slices, cover = cached

        mind_mat, roi_labels = read_mind_csv(str(row.mind_path))
        cov_raw = self._cov_vector(row)
        cov = self.scaler.transform(cov_raw.reshape(1, -1))[0]

        x_img = torch.from_numpy(img_slices[:, None, :, :].astype(np.float32, copy=False))
        x_cov = torch.from_numpy(cov.astype(np.float32, copy=False))
        x_cover = torch.from_numpy(cover.astype(np.float32, copy=False))
        x_mind = torch.from_numpy(mind_mat.astype(np.float32, copy=False))
        y = torch.tensor(int(row.label), dtype=torch.long)

        return {
            "img": x_img,
            "mind": x_mind,
            "slice_cover": x_cover,
            "cov": x_cov,
            "y": y,
            "subject_id": str(row.subject_id),
            "roi_labels": roi_labels
        }
