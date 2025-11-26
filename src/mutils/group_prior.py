# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from .mind_io import read_mind_csv

def _sim_from_mind(D: np.ndarray, mode: str="exp", tau: float=1.0, eps: float=1e-6):
    if mode == "exp":
        mu = D.mean(); sd = D.std() + eps
        Z = (D - mu) / sd
        S = np.exp(-Z / max(tau, eps))
    else:
        S = 1.0 / (1.0 + D)
    S = 0.5*(S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

def compute_group_roi_weights(df_train_fold: pd.DataFrame, n_rois=360,
                              sim_mode="exp", tau=1.0, mode="degree"):
    """
    从训练折按 (NC=0, MCI=1, AD=2) 统计每类的组级 ROI 权重：
      - mode='degree' : 用相似度矩阵 S 的行和（或行均值）作为中心性
      - mode='meanrow': 用 D 或 S 的行均值（与 degree 类似，稳健简单）
    返回 w ∈ (3, n_rois)，各行 L1 归一化
    """
    ws = []
    for c in [0,1,2]:
        sub = df_train_fold[df_train_fold["label"].astype(int)==c]
        stats = []
        for _, r in sub.iterrows():
            mat, _ = read_mind_csv(str(r["mind_path"]))
            S = _sim_from_mind(mat, mode=sim_mode, tau=tau)
            if mode == "degree":
                v = S.sum(axis=1) - 1.0   # 去掉自环也行，不去也行，差别很小
            else:
                v = S.mean(axis=1)
            stats.append(v.astype(np.float32))
        if len(stats)==0:
            w = np.ones((n_rois,), dtype=np.float32) / n_rois
        else:
            w = np.stack(stats,0).mean(0)
            w = np.maximum(w, 0)
            s = w.sum()
            w = w/s if s>0 else np.ones_like(w)/len(w)
        ws.append(w)
    return np.stack(ws,0)  # (3, n_rois)
