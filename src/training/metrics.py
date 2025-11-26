# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

def multiclass_metrics(y_true, prob_3):
    """
    y_true: (N,) in {0,1,2}
    prob_3: (N,3)
    """
    pred = prob_3.argmax(1)
    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro")
    try:
        auc_ovo = roc_auc_score(y_true, prob_3, multi_class="ovo")
    except Exception:
        auc_ovo = np.nan
    try:
        auc_ovr = roc_auc_score(y_true, prob_3, multi_class="ovr")
    except Exception:
        auc_ovr = np.nan
    macro_prec = precision_score(y_true, pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, pred, average="macro", zero_division=0)
    return dict(acc=acc, macro_f1=macro_f1, auc_ovo=auc_ovo, auc_ovr=auc_ovr,
                macro_prec=macro_prec, macro_rec=macro_rec)

def binary_slice(y_true, prob_3, pos_class, neg_class):
    """
    从三分类概率中抽取两个类，返回二分类标签与阳性概率
    """
    mask = (y_true==pos_class) | (y_true==neg_class)
    yy = y_true[mask]
    pp = prob_3[mask][:, [neg_class, pos_class]]  # 列 0=neg, 1=pos
    yb = (yy==pos_class).astype(int)
    prob_pos = pp[:,1]
    return yb, prob_pos

def binary_metrics_from_probs(y_true_bin, prob_pos, threshold=0.5):
    pred = (prob_pos >= threshold).astype(int)
    acc = accuracy_score(y_true_bin, pred)
    f1  = f1_score(y_true_bin, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true_bin, prob_pos)
    except Exception:
        auc = np.nan
    # 敏感度（召回阳性）、特异度（召回阴性）
    tn, fp, fn, tp = confusion_matrix(y_true_bin, pred, labels=[0,1]).ravel()
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    return dict(acc=acc, f1=f1, sens=sens, spec=spec, auc=auc)

def threshold_search(y_true_bin, prob_pos, target="f1"):
    """
    在 [0.1, 0.9] 搜索阈值，最大化 target
    """
    best_t, best_v = 0.5, -1
    for t in np.linspace(0.1, 0.9, 33):
        m = binary_metrics_from_probs(y_true_bin, prob_pos, threshold=t)
        v = m.get(target, 0)
        if np.isnan(v): v = 0
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v
