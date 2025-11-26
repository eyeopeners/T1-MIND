# scripts/metrics.py
# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, roc_curve, auc, brier_score_loss)
from sklearn.calibration import calibration_curve

EPS = 1e-9

@dataclass
class MultiClassMetrics:
    acc: float
    macro_f1: float
    macro_prec: float
    macro_rec: float
    kappa: float
    auc_ovo: float
    auc_ovr: float
    per_class: Dict[int, Dict[str, float]]

def cohen_kappa(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if abs(1 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1 - pe)

def multiclass_metrics(y_true: np.ndarray, prob: np.ndarray, labels: List[int]=(0,1,2)) -> MultiClassMetrics:
    pred = prob.argmax(1)
    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average='macro')
    macro_prec = precision_score(y_true, pred, average='macro', zero_division=0)
    macro_rec  = recall_score(y_true, pred, average='macro', zero_division=0)
    kappa = cohen_kappa(y_true, pred)

    # AUC
    auc_ovo = roc_auc_score(y_true, prob, multi_class='ovo')
    auc_ovr = roc_auc_score(y_true, prob, multi_class='ovr')

    # per-class one-vs-rest
    per_cls = {}
    for c in labels:
        y_bin = (y_true == c).astype(int)
        per_cls[c] = {
            "sens_recall": recall_score(y_bin, (prob[:, c] >= 0.5).astype(int), zero_division=0),  # at 0.5
            "spec":       recall_score(1 - y_bin, (prob[:, c] < 0.5).astype(int), zero_division=0),
            "auc_ovr":    roc_auc_score(y_bin, prob[:, c]),
            "f1":         f1_score(y_bin, (prob[:, c] >= 0.5).astype(int), zero_division=0),
            "prec":       precision_score(y_bin, (prob[:, c] >= 0.5).astype(int), zero_division=0),
        }
    return MultiClassMetrics(acc, macro_f1, macro_prec, macro_rec, kappa, auc_ovo, auc_ovr, per_cls)

def binary_metrics(y_true_bin: np.ndarray, score_pos: np.ndarray, thr: float=0.5) -> Dict[str, float]:
    y_pred = (score_pos >= thr).astype(int)
    acc = accuracy_score(y_true_bin, y_pred)
    f1  = f1_score(y_true_bin, y_pred, zero_division=0)
    sens = recall_score(y_true_bin, y_pred, zero_division=0)
    spec = recall_score(1 - y_true_bin, 1 - y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true_bin, score_pos)
    except Exception:
        roc = float('nan')
    return {"acc": acc, "f1": f1, "sens": sens, "spec": spec, "auc": roc}

def mean_std(d: List[float]) -> Tuple[float, float]:
    a = np.array(d, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int=15) -> float:
    # top-1 confidence calibration error
    conf = prob.max(axis=1)
    pred = prob.argmax(1)
    correct = (pred == y_true).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0: continue
        ece += np.abs(correct[m].mean() - conf[m].mean()) * (m.sum()/len(conf))
    return float(ece)
