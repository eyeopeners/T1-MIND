# -*- coding: utf-8 -*-
"""
从各 fold 的 npz（engine_mind 保存）与 cv_summary.json 汇总报告：
- 打印 3-way 与三个二分类的 mean±std
- 画混淆矩阵
- 简易 t-SNE（用每折最后一轮的融合特征也可，但这里先用概率）
- 切片注意力热图、prior vs attn（示例，读取 best fold 单个 batch 的存档可扩展）
- GradCAM++ 可选：因当前 slice encoder 是浅 CNN，可给出一个例子（留作扩展）
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE


def _mean_std(xs):
    return np.nanmean(xs), np.nanstd(xs)

def _bin_collect(prefix, folds):
    m = dict(acc=[], f1=[], sens=[], spec=[], auc=[])
    for f in folds:
        for k in m: m[k].append(f[prefix][k])
    return {k: (np.nanmean(v), np.nanstd(v)) for k,v in m.items()}

def _calc_bin(y, prob, thr=0.5):
    pred = (prob>=thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, prob) if (len(np.unique(y))==2) else np.nan
    return dict(acc=acc, f1=f1, sens=sens, spec=spec, auc=auc)

def generate_report(cfg):
    outdir = cfg["outdir"]
    folds = []
    for k in range(1, cfg["folds"]+1):
        fn = os.path.join(outdir, f"fold{ k }_val_last.npz")
        if not os.path.exists(fn):
            continue
        data = np.load(fn, allow_pickle=True)
        rec = {}
        # 三分类（用 probs 求预测做 cm 或直接统计 acc/f1——这里仅演示收集）
        # 二分类指标：
        rec["NC_AD"]  = _calc_bin(data["nc_ad_y"],  data["nc_ad_prob"])
        rec["NC_MCI"] = _calc_bin(data["nc_mci_y"], data["nc_mci_prob"])
        rec["MCI_AD"] = _calc_bin(data["mci_ad_y"], data["mci_ad_prob"])
        folds.append(rec)

    if len(folds)==0:
        print("[report] no fold npz found, skip.")
        return

    # 打印均值±标准差
    for name in ["NC_AD","NC_MCI","MCI_AD"]:
        stats = _bin_collect(name, folds)
        print(f"[Binary {name}]"
              f" acc={stats['acc'][0]:.4f}±{stats['acc'][1]:.4f},"
              f" f1={stats['f1'][0]:.4f}±{stats['f1'][1]:.4f},"
              f" sens={stats['sens'][0]:.4f}±{stats['sens'][1]:.4f},"
              f" spec={stats['spec'][0]:.4f}±{stats['spec'][1]:.4f},"
              f" auc={stats['auc'][0]:.4f}±{stats['auc'][1]:.4f}")

    # 画一个示例 t-SNE（基于 (p0,p1,p2) ）
    # 将所有折合并
    all_prob = []
    all_y = []
    for k in range(1, cfg["folds"]+1):
        fn = os.path.join(outdir, f"fold{ k }_val_last.npz")
        if os.path.exists(fn):
            data = np.load(fn, allow_pickle=True)
            all_prob.append(data["prob3"])
            all_y.append(data["y"])
    P = np.concatenate(all_prob, axis=0)
    Y = np.concatenate(all_y, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random", random_state=42)
    Z = tsne.fit_transform(P)
    plt.figure(figsize=(6,5))
    for c, name, col in [(0,"NC","tab:blue"), (1,"MCI","tab:orange"), (2,"AD","tab:red")]:
        m = (Y==c)
        plt.scatter(Z[m,0], Z[m,1], s=10, alpha=0.7, label=name, c=col)
    plt.legend(); plt.title("t-SNE on class probabilities"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_probs.png"), dpi=200)
    plt.close()

    print("[report] saved:", os.path.join(outdir, "tsne_probs.png"))
