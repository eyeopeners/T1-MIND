# scripts/metrics_report.py
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from itertools import product

def plot_conf_mat(cm, classes, save):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save, dpi=180); plt.close(fig)

def binary_curves(y_true, score, pos_name, save_prefix):
    fpr, tpr, thr = roc_curve(y_true, score)
    auc = roc_auc_score(y_true, score)
    prec, rec, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)

    # ROC
    fig = plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC: {pos_name}')
    plt.legend(); fig.savefig(f"{save_prefix}_roc.png", dpi=180); plt.close(fig)

    # PR
    fig = plt.figure(figsize=(4,4))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR: {pos_name}')
    plt.legend(); fig.savefig(f"{save_prefix}_pr.png", dpi=180); plt.close(fig)

    return auc, ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='runs/val_preds.csv')
    ap.add_argument('--outdir', default='runs/report')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.preds)
    classes = ['NC','MCI','AD']
    y = df['y'].values
    p3 = df[['p_nc','p_mci','p_ad']].values
    yhat = p3.argmax(1)

    # --- 三分类报告 ---
    cm = confusion_matrix(y, yhat, labels=[0,1,2])
    plot_conf_mat(cm, classes, os.path.join(args.outdir,'confusion_matrix.png'))
    rep = classification_report(y, yhat, labels=[0,1,2], target_names=classes, output_dict=True)
    pd.DataFrame(rep).to_csv(os.path.join(args.outdir,'clf_report_3c.csv'))

    # 宏AUC（OvR）
    macro_ovr_auc = roc_auc_score(y, p3, multi_class='ovr', labels=[0,1,2])
    macro_ovo_auc = roc_auc_score(y, p3, multi_class='ovo', labels=[0,1,2])
    with open(os.path.join(args.outdir,'auc_summary.txt'),'w') as f:
        f.write(f"macro AUC (OvR): {macro_ovr_auc:.4f}\nmacro AUC (OvO): {macro_ovo_auc:.4f}\n")

    # --- 任意二分类（重标） ---
    pairs = [
        ('NC vs AD', 0, 2, 'p_nc_ad'),
        ('NC vs MCI', 0, 1, 'p_nc_mci'),
        ('MCI vs AD', 1, 2, 'p_mci_ad'),
    ]
    rows=[]
    for name,c1,c2,col in pairs:
        mask = np.isin(y,[c1,c2])
        y_bin = (y[mask]==c2).astype(int) # 以后一类(c2)为阳性
        score = df.loc[mask, col].values
        auc, ap = binary_curves(y_bin, score, name, os.path.join(args.outdir, f"pair_{c1}_{c2}"))
        rows.append(dict(pair=name, auc=auc, ap=ap, n=int(mask.sum())))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir,'pairwise_auc_ap.csv'), index=False)

    # --- 错判清单 ---
    err = df[df['y']!=yhat].copy()
    err.rename(columns={'y':'y_true'}, inplace=True)
    err['y_pred'] = yhat[err.index]
    err.to_csv(os.path.join(args.outdir,'misclassified.csv'), index=False)

    print("Report saved to", args.outdir)

if __name__ == "__main__":
    main()
