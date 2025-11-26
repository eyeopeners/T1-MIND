# scripts/plotting.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def _ensure(outdir):
    os.makedirs(outdir, exist_ok=True)

def plot_confusion(y_true, y_pred, labels, out_png, normalize=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}" if normalize else int(cm[i,j]),
                    ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180); plt.close(fig)

def plot_roc_ovr(y_true, prob, class_names, out_png):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1], [0,1], "--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("OVR ROC")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def plot_pr_ovr(y_true, prob, class_names, out_png):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        p, r, _ = precision_recall_curve(y_bin, prob[:, i])
        ax.plot(r, p, label=f"{name}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("OVR PR")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def plot_reliability(y_true, prob, out_png, n_bins=15):
    conf = prob.max(axis=1)
    pred = prob.argmax(1)
    correct = (pred == y_true).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    accs, confs = [], []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0: continue
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot([0,1],[0,1],"--")
    ax.plot(confs, accs, marker="o")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
