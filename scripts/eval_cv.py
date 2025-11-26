# scripts/eval_cv.py
# -*- coding: utf-8 -*-

# === 确保能 import src 包 ===
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from scripts.metrics import multiclass_metrics, binary_metrics, mean_std, ece_score
from scripts.plotting import plot_confusion, plot_roc_ovr, plot_pr_ovr, plot_reliability

from src.models.ppal_mind import PPAL_MIND
from src.data.dataset_2d import Neuro2DDataset
from torch.utils.data import DataLoader


def load_cfg(cfg_path):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _get_manifest_path(cfg: dict) -> str:
    # 支持两种写法：cfg["manifest"] 或 cfg["data"]["manifest"]
    if "manifest" in cfg:
        return cfg["manifest"]
    if "data" in cfg and isinstance(cfg["data"], dict) and "manifest" in cfg["data"]:
        return cfg["data"]["manifest"]
    raise KeyError("manifest path not found in config. "
                   "Use either 'manifest: <path>' or 'data: { manifest: <path> }'.")

def _get_outdir(cfg: dict) -> str:
    # 优先 outdir；没有就回退到 logdir/checkpoints
    if "outdir" in cfg:
        return cfg["outdir"]
    if "logdir" in cfg:
        return os.path.join(cfg["logdir"], "checkpoints")
    # 最后兜底到当前目录下的 runs/checkpoints
    return os.path.join("runs", "checkpoints")

def compute_class_weights(y):
    cls, cnt = np.unique(y, return_counts=True)
    w = cnt.sum() / np.maximum(cnt, 1)
    w = w / w.mean()
    vec = np.ones(3, dtype=np.float32)
    vec[cls] = w
    return vec

def forward_model(model, dl, device):
    model.eval()
    ys, prob3, logits_adnon, logits_ncmci, sids = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            img   = batch["img"].to(device)
            mind  = batch["mind"].to(device)
            cover = batch["slice_cover"].to(device)
            cov   = batch["cov"].to(device)
            y     = batch["y"].to(device)
            out = model({"img":img, "mind":mind, "slice_cover":cover, "cov":cov})
            p3 = torch.softmax(out["logits_3"], dim=1).cpu().numpy()
            ys.append(y.cpu().numpy()); prob3.append(p3)
            logits_adnon.append(out["logits_ad_non"].cpu().numpy())
            logits_ncmci.append(out["logits_nc_mci"].cpu().numpy())
            sids += list(batch["subject_id"])
    return (np.concatenate(ys), 
            np.concatenate(prob3),
            np.concatenate(logits_adnon),
            np.concatenate(logits_ncmci),
            sids)

def to_bin_subset(y, prob3, cls_pos, cls_neg):
    m = (y==cls_pos) | (y==cls_neg)
    yb = (y[m]==cls_pos).astype(int)
    sb = prob3[m, cls_pos]
    return yb, sb

def main(args):
    cfg = load_cfg(args.config)

    # --- 读取关键路径 ---
    manifest_path = _get_manifest_path(cfg)
    outdir_ckpt   = _get_outdir(cfg)

    # Windows YAML 建议把路径用引号括起来避免转义问题
    # 但 pandas 读取时一般没问题，这里不过度干预
    manifest = pd.read_csv(manifest_path)

    # --- 设备、折数、dataloader 超参 ---
    want_cuda = (cfg.get("device", "cuda") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if want_cuda else "cpu")
    n_splits = int(cfg.get("folds", 5))
    seed     = int(cfg.get("seed", 3407))
    num_workers = int(cfg.get("num_workers", 0))
    batch_size  = int(cfg.get("batch_size", 4))

    y_all = manifest["label"].values.astype(int)

    # 评估输出目录
    report_root = os.path.join(_get_outdir(cfg), "..", "eval_report")  # 放到 checkpoints 同级的 eval_report
    report_root = os.path.abspath(report_root)
    os.makedirs(report_root, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds_metrics = []
    folds_bin = {"NC_AD":[], "NC_MCI_3way":[], "NC_MCI_head":[], "MCI_AD":[]}
    preds_rows = []
    class_names = ["NC","MCI","AD"]

    for k,(tr,va) in enumerate(skf.split(np.zeros_like(y_all), y_all), start=1):
        df_va = manifest.iloc[va].copy()

        # === 与训练一致：用训练折拟合 cov scaler & race_vocab 覆盖 1..7 ===
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_tr = manifest.iloc[tr].copy()
        ds_fit = Neuro2DDataset(df_tr,
                                slices=int(cfg.get("slices",64)),
                                axis=str(cfg.get("axis","axial")),
                                img_size=int(cfg.get("img_size",224)),
                                fit_scaler=True, scaler=scaler,
                                race_vocab=[1,2,3,4,5,6,7])
        ds_va  = Neuro2DDataset(df_va,
                                slices=int(cfg.get("slices",64)),
                                axis=str(cfg.get("axis","axial")),
                                img_size=int(cfg.get("img_size",224)),
                                fit_scaler=False, scaler=scaler,
                                race_vocab=[1,2,3,4,5,6,7])
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=want_cuda)

        # === 构建模型 & 加载本折最优权重 ===
        mcfg = type("MCFG",(object,),{
            "lambda_prior": float(cfg.get("lambda_prior", 0.2)),
            "mind_sim": cfg.get("mind",{}).get("sim", "exp"),
            "mind_tau": float(cfg.get("mind",{}).get("tau", 1.0)),
            "cov_dim": ds_fit.scaler.mean_.shape[0]
        })()
        model = PPAL_MIND(mcfg).to(device)

        # 优先 outdir；找不到就回退 logdir/checkpoints
        ckpt = os.path.join(outdir_ckpt, f"fold{k}_best.pt")
        if not os.path.exists(ckpt):
            alt = os.path.join(cfg.get("logdir", "runs/exp"), "checkpoints", f"fold{k}_best.pt")
            if os.path.exists(alt):
                ckpt = alt
        if not os.path.exists(ckpt):
            print(f"[WARN] checkpoint not found for fold{k}: {ckpt} (skip this fold)")
            continue

        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        # === 推理 ===
        y, p3, l_adnon, l_ncmci, sids = forward_model(model, dl_va, device)
        y_pred = p3.argmax(1)

        for sid, yt, pr in zip(sids, y, p3):
            preds_rows.append({
                "fold": k,
                "subject_id": sid,
                "y_true": int(yt),
                "p_nc": float(pr[0]),
                "p_mci": float(pr[1]),
                "p_ad": float(pr[2]),
                "pred": int(np.argmax(pr))
            })

        # === 多分类指标 + 图 ===
        mc = multiclass_metrics(y, p3, labels=[0,1,2])
        folds_metrics.append(mc.__dict__)

        plot_confusion(y, y_pred, labels=[0,1,2],
                       out_png=os.path.join(report_root, f"fold{k}_cm_raw.png"),
                       normalize=None, title=f"Confusion (fold{k})")
        plot_confusion(y, y_pred, labels=[0,1,2],
                       out_png=os.path.join(report_root, f"fold{k}_cm_rownorm.png"),
                       normalize='true', title=f"Confusion Row-Norm (fold{k})")
        plot_roc_ovr(y, p3, class_names, os.path.join(report_root, f"fold{k}_roc_ovr.png"))
        plot_pr_ovr(y, p3, class_names, os.path.join(report_root, f"fold{k}_pr_ovr.png"))
        plot_reliability(y, p3, os.path.join(report_root, f"fold{k}_reliability.png"))

        # === 任意两类二分类 ===
        # NC vs AD（正类 AD=2）
        yb, sb = to_bin_subset(y, p3, cls_pos=2, cls_neg=0)
        folds_bin["NC_AD"].append(binary_metrics(yb, sb))

        # NC vs MCI：a) 来自三分类概率（正类 MCI=1）
        yb2, sb2 = to_bin_subset(y, p3, cls_pos=1, cls_neg=0)
        folds_bin["NC_MCI_3way"].append(binary_metrics(yb2, sb2))

        # NC vs MCI：b) 层级头 logits_nc_mci
        prob_mci = torch.softmax(torch.from_numpy(l_ncmci), dim=1).numpy()[:,1]
        m = (y==0) | (y==1)
        folds_bin["NC_MCI_head"].append(binary_metrics((y[m]==1).astype(int), prob_mci[m]))

        # MCI vs AD（正类 AD=2）
        yb3, sb3 = to_bin_subset(y, p3, cls_pos=2, cls_neg=1)
        folds_bin["MCI_AD"].append(binary_metrics(yb3, sb3))

    # === 汇总保存 ===
    os.makedirs(report_root, exist_ok=True)
    preds_df = pd.DataFrame(preds_rows)
    preds_df.to_csv(os.path.join(report_root, "preds_cv.csv"), index=False)
    err_df = preds_df[preds_df["y_true"] != preds_df["pred"]].copy()
    err_df.to_csv(os.path.join(report_root, "errors_cv.csv"), index=False)

    def agg_mc(key):
        vals = [f[key] for f in folds_metrics]
        return mean_std(vals)

    mc_summary = {
        "acc_mean_std":  agg_mc("acc"),
        "macro_f1_mstd": agg_mc("macro_f1"),
        "macro_prec_mstd": agg_mc("macro_prec"),
        "macro_rec_mstd":  agg_mc("macro_rec"),
        "kappa_mstd":    agg_mc("kappa"),
        "auc_ovo_mstd":  agg_mc("auc_ovo"),
        "auc_ovr_mstd":  agg_mc("auc_ovr"),
        "ece_mstd":      mean_std([
            ece_score(
              preds_df[preds_df["fold"]==i]["y_true"].values,
              preds_df[preds_df["fold"]==i][["p_nc","p_mci","p_ad"]].values,
            )
            for i in range(1, 1+int(cfg.get("folds",5))) if (preds_df["fold"]==i).any()
        ])
    }

    bin_summary = {}
    for pair, lst in folds_bin.items():
        for metric in ["acc","f1","sens","spec","auc"]:
            m, s = mean_std([d[metric] for d in lst]) if len(lst)>0 else (float("nan"), float("nan"))
            bin_summary.setdefault(pair, {})[metric] = {"mean": m, "std": s}

    with open(os.path.join(report_root, "metrics_multiclass.json"), "w") as f:
        json.dump({"folds": folds_metrics, "summary": mc_summary}, f, indent=2)
    with open(os.path.join(report_root, "metrics_binary.json"), "w") as f:
        json.dump(bin_summary, f, indent=2)

    # 控制台打印
    print("\n== Multiclass (mean±std over folds) ==")
    for name, (m,s) in mc_summary.items():
        if isinstance(m, float):
            print(f"{name}: {m:.4f} ± {s:.4f}")
    print("\n== Binary pairs (mean±std over folds) ==")
    for pair, d in bin_summary.items():
        line = ", ".join([f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k,v in d.items()])
        print(f"{pair}: {line}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="same yaml used for training")
    args = ap.parse_args()
    main(args)
