import os
import time
import json
import random

import numpy as np# -*- coding: utf-8 -*-
import os, json, time, yaml, numpy as np, pandas as pd, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset_2d import Neuro2DDataset
from src.models.ppal_mind import PPAL_MIND
from src.models.text_prompt_encoder import TextPromptEncoder
from src.training.criterions import ce_loss, focal_loss, kl_symmetric, distill_kl
from src.training.metrics import multiclass_metrics, binary_slice, binary_metrics_from_probs, threshold_search


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_class_weights(y, n_classes=3):
    vals, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    w = np.ones(n_classes, dtype=np.float32)
    for v, c in zip(vals, counts):
        w[int(v)] = total / (len(vals) * c)
    return w

def make_sampler(y):
    vals, counts = np.unique(y, return_counts=True)
    freq = {int(v): float(c) for v,c in zip(vals, counts)}
    weights = np.array([1.0 / freq[int(t)] for t in y], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def _get(cfg, key, default):
    return cfg.get(key, default)

def run_fold(df_tr, df_va, cfg, fold_id):
    device = torch.device("cuda" if (cfg.get("device","cuda")=="cuda" and torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg.get("amp", True)) and (device.type=="cuda")

    cudnn.benchmark = True

    # ====== Data ======
    race_vocab = [1,2,3,4,5,6,7]
    scaler = StandardScaler()
    ds_tr = Neuro2DDataset(
        df_tr, slices=cfg["slices"], axis=cfg["axis"], img_size=cfg["img_size"],
        fit_scaler=True, scaler=scaler, race_vocab=race_vocab,
        cache_max_items=int(cfg.get("data", {}).get("cache_max_items_train", 0))
    )
    ds_va = Neuro2DDataset(
        df_va, slices=cfg["slices"], axis=cfg["axis"], img_size=cfg["img_size"],
        fit_scaler=False, scaler=scaler, race_vocab=race_vocab,
        cache_max_items=int(cfg.get("data", {}).get("cache_max_items_val", 0))
    )

    # Sampler
    if cfg.get("imbalance", {}).get("use_sampler", True):
        sampler = make_sampler(df_tr["label"].astype(int).values)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # ==== 分开 train/val 的 DataLoader 参数 ====
    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(_get(cfg, "batch_size", 12)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(_get(cfg, "num_workers_train", _get(cfg, "num_workers", 8))),
        pin_memory=bool(_get(cfg, "pin_memory_train", True)),
        persistent_workers=bool(_get(cfg, "persistent_workers_train", True)),
        prefetch_factor=int(_get(cfg, "prefetch_factor_train", _get(cfg, "prefetch_factor", 2)))
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(_get(cfg, "batch_size_val", _get(cfg, "batch_size", 12))),
        shuffle=False,
        num_workers=int(_get(cfg, "num_workers_val", _get(cfg, "num_workers", 2))),
        pin_memory=bool(_get(cfg, "pin_memory_val", False)),
        persistent_workers=bool(_get(cfg, "persistent_workers_val", False)),
        prefetch_factor=int(_get(cfg, "prefetch_factor_val", _get(cfg, "prefetch_factor", 2)))
    )

    # class weights
    class_w = compute_class_weights(df_tr["label"].values, n_classes=3)
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)

    # ====== Model ======
    mcfg = type("MCFG",(object,),{
        "lambda_prior": cfg["lambda_prior"],
        "mind_sim": cfg["mind"]["sim"],
        "mind_tau": cfg["mind"]["tau"],
        "cov_dim": ds_tr.scaler.mean_.shape[0]
    })()
    model = PPAL_MIND(mcfg).to(device)

    # 文本原型
    text_cfg = cfg.get("text", {})
    use_text = bool(text_cfg.get("use", False))
    if use_text:
        text_enc = TextPromptEncoder(
            clip_arch=text_cfg.get("clip_arch","ViT-B/32"),
            device=device,
            temperature=text_cfg.get("temperature", 0.07),
            normalize=bool(text_cfg.get("normalize", True))
        ).to(device)
        text_enc.build_class_prototypes(
            textprompt_csv=text_cfg.get("textprompt_csv",""),
            per_fold_dir=text_cfg.get("per_fold_dir",""),
            fold_id=fold_id
        )
        alpha_text = float(text_cfg.get("alpha_text", 0.3))
        lambda_text = float(text_cfg.get("lambda_text", 0.2))
        T_text = float(text_cfg.get("temperature", 0.07))
    else:
        text_enc, alpha_text, lambda_text, T_text = None, 0.0, 0.0, 1.0

    # Optim
    #opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl_tr)*cfg["epochs"]))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 不均衡
    imb_cfg = cfg.get("imbalance", {})
    use_focal = bool(imb_cfg.get("focal_loss", True))
    focal_gamma = float(imb_cfg.get("focal_gamma", 2.0))
    label_smoothing = float(imb_cfg.get("label_smoothing", 0.0))

    # 稀疏选片
    sp_cfg = cfg.get("sparsify", {})
    use_sparse = bool(sp_cfg.get("use", True))
    topk_ratio = float(sp_cfg.get("topk_ratio", 0.25))
    lambda_sparse = float(sp_cfg.get("lambda_sparse", 0.01))

    best_score, best_state = -1.0, None
    os.makedirs(cfg["outdir"], exist_ok=True)

    save_mid = cfg.get("report", {}).get("save_per_fold_npz", True)
    fold_cache = []

    # 新增：记录每个 epoch 的 train / val loss
    train_loss_hist = []
    val_loss_hist = []

    for ep in range(1, cfg["epochs"]+1):
        # ===== Train =====
        model.train()
        t0 = time.time()
        loss_sum, total = 0.0, 0

        for batch in tqdm(dl_tr, desc=f"[fold{fold_id}] train ep{ep}"):
            img   = batch["img"].to(device, non_blocking=True)
            mind  = batch["mind"].to(device, non_blocking=True)
            cover = batch["slice_cover"].to(device, non_blocking=True)
            cov   = batch["cov"].to(device, non_blocking=True)
            y     = batch["y"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model({"img":img, "mind":mind, "slice_cover":cover, "cov":cov})
                logits_3 = out["logits_3"]
                if use_focal:
                    loss_main = focal_loss(logits_3, y, weight=class_w_t, gamma=focal_gamma)
                else:
                    loss_main = ce_loss(logits_3, y, weight=class_w_t, label_smoothing=label_smoothing)

                y_ad = (y==2).long()
                loss_ad = ce_loss(out["logits_ad_non"], y_ad)

                mask_non = (y!=2)
                if mask_non.any():
                    loss_nm = ce_loss(out["logits_nc_mci"][mask_non], (y[mask_non]==1).long())
                else:
                    loss_nm = logits_3.new_tensor(0.0)

                p_slice = out["slice_prior"]
                a_slice = out["slice_attn"]
                loss_prior = kl_symmetric(a_slice, p_slice)

                if use_sparse:
                    B, S = a_slice.shape
                    k = max(1, int(S * topk_ratio))
                    topk, _ = torch.topk(a_slice, k, dim=1)
                    loss_sparse = (a_slice.sum(dim=1) - topk.sum(dim=1)).mean()
                else:
                    loss_sparse = logits_3.new_tensor(0.0)

                if use_text:
                    logits_text, _ = text_enc(out["feat"].float())
                    loss_text = distill_kl(logits_3, logits_text, T=T_text) * lambda_text
                    logits_fused = (1.0 - alpha_text) * logits_3 + alpha_text * logits_text
                    if use_focal:
                        loss_fused = focal_loss(logits_fused, y, weight=class_w_t, gamma=focal_gamma)
                    else:
                        loss_fused = ce_loss(logits_fused, y, weight=class_w_t, label_smoothing=label_smoothing)
                else:
                    loss_text = logits_3.new_tensor(0.0)
                    loss_fused = loss_main

                # 你调整后的 loss_ad / loss_nm 权重保留
                #loss = loss_fused + 0.5*loss_ad + 0.5*loss_nm + cfg["lambda_prior"]*loss_prior + lambda_sparse*loss_sparse + loss_text
                loss = loss_fused + loss_ad + loss_nm + cfg["lambda_prior"]*loss_prior + lambda_sparse*loss_sparse + loss_text

            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()

            bs = y.size(0); total += bs
            loss_sum += float(loss.item()) * bs

        tr_loss = loss_sum / max(1, total)
        train_loss_hist.append(tr_loss)
        et = time.time() - t0

        # ===== Valid =====
        model.eval()
        ys_list, ps_list, ad_prob_list = [], [], []

        # 新增：计算 val loss
        val_loss_sum, val_total = 0.0, 0

        with torch.no_grad():
            for batch in tqdm(dl_va, desc=f"[fold{fold_id}] valid ep{ep}"):
                img   = batch["img"].to(device, non_blocking=True)
                mind  = batch["mind"].to(device, non_blocking=True)
                cover = batch["slice_cover"].to(device, non_blocking=True)
                cov   = batch["cov"].to(device, non_blocking=True)
                y     = batch["y"].to(device, non_blocking=True)

                out  = model({"img":img, "mind":mind, "slice_cover":cover, "cov":cov})

                logits_3 = out["logits_3"]
                if use_text:
                    logits_text, _ = text_enc(out["feat"].float())
                    logits_3 = (1.0 - alpha_text) * logits_3 + alpha_text * logits_text

                # val loss（用融合后的 logits_3）
                if use_focal:
                    loss_val_batch = focal_loss(logits_3, y, weight=class_w_t, gamma=focal_gamma)
                else:
                    loss_val_batch = ce_loss(logits_3, y, weight=class_w_t, label_smoothing=label_smoothing)

                bs = y.size(0)
                val_total += bs
                val_loss_sum += float(loss_val_batch.item()) * bs

                prob_3 = torch.softmax(logits_3, dim=1).cpu().numpy()
                ys_list.append(y.cpu().numpy())
                ps_list.append(prob_3)

                p_ad = torch.softmax(out["logits_ad_non"], dim=1)[:,1].cpu().numpy()
                ad_prob_list.append(p_ad)

        if val_total > 0:
            va_loss = val_loss_sum / val_total
        else:
            va_loss = float("nan")
        val_loss_hist.append(va_loss)

        if len(ys_list) == 0:
            print(f"[fold{fold_id}] ep{ep} WARNING: empty validation set.")
            score = 0.0
            print(f"[fold{fold_id}] ep{ep} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} time={et/60:.1f}m")
            continue

        ys = np.concatenate(ys_list, axis=0)
        ps = np.concatenate(ps_list, axis=0)
        ad_prob = np.concatenate(ad_prob_list, axis=0)

        m3 = multiclass_metrics(ys, ps)

        nc_ad_y,  nc_ad_prob  = binary_slice(ys, ps, pos_class=2, neg_class=0)
        nc_mci_y, nc_mci_prob = binary_slice(ys, ps, pos_class=1, neg_class=0)
        mci_ad_y, mci_ad_prob = binary_slice(ys, ps, pos_class=2, neg_class=1)

        b_nc_ad  = binary_metrics_from_probs(nc_ad_y,  nc_ad_prob,  threshold=0.5)
        b_nc_mci = binary_metrics_from_probs(nc_mci_y, nc_mci_prob, threshold=0.5)
        b_mci_ad = binary_metrics_from_probs(mci_ad_y, mci_ad_prob, threshold=0.5)

        if cfg.get("report", {}).get("do_ad_threshold_search", True):
            ad_bin = (ys==2).astype(int)
            best_t, _ = threshold_search(ad_bin, ad_prob, target="f1")
            b_ad_tuned = binary_metrics_from_probs(ad_bin, ad_prob, threshold=best_t)
            tuned_str = f" | ADvsNON@t={best_t:.2f} f1={b_ad_tuned['f1']:.3f} sens={b_ad_tuned['sens']:.3f} spec={b_ad_tuned['spec']:.3f} auc={b_ad_tuned['auc']:.3f}"
        else:
            tuned_str = ""

        score = float(np.nan_to_num(m3["auc_ovo"], nan=m3["acc"]))

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(cfg["outdir"], f"fold{fold_id}_best.pt"))

        print(f"[fold{fold_id}] ep{ep} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} | "
              f"3way acc={m3['acc']:.3f} f1={m3['macro_f1']:.3f} auc_ovo={m3['auc_ovo']:.3f} auc_ovr={m3['auc_ovr']:.3f} | "
              f"NC-AD acc={b_nc_ad['acc']:.3f} f1={b_nc_ad['f1']:.3f} sens={b_nc_ad['sens']:.3f} spec={b_nc_ad['spec']:.3f} auc={b_nc_ad['auc']:.3f} | "
              f"NC-MCI acc={b_nc_mci['acc']:.3f} f1={b_nc_mci['f1']:.3f} sens={b_nc_mci['sens']:.3f} spec={b_nc_mci['spec']:.3f} auc={b_nc_mci['auc']:.3f} | "
              f"MCI-AD acc={b_mci_ad['acc']:.3f} f1={b_mci_ad['f1']:.3f} sens={b_mci_ad['sens']:.3f} spec={b_mci_ad['spec']:.3f} auc={b_mci_ad['auc']:.3f} | "
              f"time={et/60:.1f}m{tuned_str}")

        if save_mid and ep == cfg["epochs"]:
            fold_cache.append(dict(
                y=ys, prob3=ps, nc_ad_y=nc_ad_y, nc_ad_prob=nc_ad_prob,
                nc_mci_y=nc_mci_y, nc_mci_prob=nc_mci_prob,
                mci_ad_y=mci_ad_y, mci_ad_prob=mci_ad_prob, ad_prob=ad_prob
            ))

    # ============ 训练结束：保证 best_state 存在 ============
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(cfg["outdir"], f"fold{fold_id}_best.pt"))

    # ============ 保存最后一轮的 npz（你原来的逻辑） ============
    if save_mid and len(fold_cache)>0:
        out_npz = os.path.join(cfg["outdir"], f"fold{fold_id}_val_last.npz")
        np.savez_compressed(out_npz, **fold_cache[-1])

    # ============ 绘制 train/val loss 曲线 ============
    epochs = len(train_loss_hist)
    if epochs > 0:
        plt.figure()
        plt.plot(range(1, epochs+1), train_loss_hist, label="train_loss")
        plt.plot(range(1, epochs+1), val_loss_hist, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold_id} Train/Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg["outdir"], f"fold{fold_id}_loss_curve.png"))
        plt.close()

    # ============ 使用 best_state 在验证集上画混淆矩阵 ============
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    all_y, all_pred = [], []
    with torch.no_grad():
        for batch in dl_va:
            img   = batch["img"].to(device, non_blocking=True)
            mind  = batch["mind"].to(device, non_blocking=True)
            cover = batch["slice_cover"].to(device, non_blocking=True)
            cov   = batch["cov"].to(device, non_blocking=True)
            y     = batch["y"].to(device, non_blocking=True)

            out  = model({"img":img, "mind":mind, "slice_cover":cover, "cov":cov})
            logits_3 = out["logits_3"]
            if use_text:
                logits_text, _ = text_enc(out["feat"].float())
                logits_3 = (1.0 - alpha_text) * logits_3 + alpha_text * logits_text

            pred = torch.argmax(logits_3, dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    if len(all_y) > 0:
        y_true = np.concatenate(all_y, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Fold {fold_id} Confusion Matrix (NC=0, MCI=1, AD=2)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, ["NC", "MCI", "AD"])
        plt.yticks(tick_marks, ["NC", "MCI", "AD"])

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg["outdir"], f"fold{fold_id}_confmat.png"))
        plt.close()

    return float(best_score)

def run(cfg):
    set_seed(cfg.get("seed",42))
    df = pd.read_csv(cfg["manifest"])
    y = df["label"].astype(int).values

    skf = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=42)
    scores=[]
    for k,(tr,va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        df_tr, df_va = df.iloc[tr].copy(), df.iloc[va].copy()
        sc = run_fold(df_tr, df_va, cfg, k)
        scores.append(sc)

    os.makedirs(cfg["outdir"], exist_ok=True)
    with open(os.path.join(cfg["outdir"], "cv_summary.json"), "w") as f:
        json.dump({"scores": scores, "mean": float(np.mean(scores))}, f, indent=2)

    print("[run] done.")

    if cfg.get("report",{}).get("viz_after_train", True):
        try:
            from scripts.post_train_report import generate_report
            generate_report(cfg)
        except Exception as e:
            print("[report] failed:", e)