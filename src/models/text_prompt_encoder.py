# -*- coding: utf-8 -*-
import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import clip


class TextPromptEncoder(nn.Module):
    """
    将诊断类别文本先验编码为“类别原型”，并与图像融合：
      - 支持单一 CSV（全局）或每折独立 CSV（per_fold_dir/textprompt_fold{k}.csv）
      - CSV 至少包含列：class,label,text
         * class/label 建议都放，class 取 {0,1,2}；label 可与现有 manifest 对齐
         * text 可为中文/英文，clip.tokenize 会处理基本 Unicode
      - 输出：
         1) self.class_proto: (C, D_clip) 的归一化文本原型
         2) forward(feat): 计算 logits_text（feat 与文本原型的余弦相似度/温度缩放）
    """
    def __init__(self,
                 clip_arch: str = "ViT-B/32",
                 device: torch.device = torch.device("cpu"),
                 temperature: float = 0.07,
                 normalize: bool = True):
        super().__init__()
        self.device = device
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        # 加载 CLIP
        self.clip_model, _ = clip.load(clip_arch, device=self.device, jit=False)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # 缓存原型
        self.register_buffer("class_proto", torch.zeros(3, self.clip_model.text_projection.shape[1]))
        self.num_classes = 3

        # 将图像侧特征投到 CLIP 文本空间（或反之）。这里选：把图像/融合特征投到 CLIP 文本维度
        d_clip = self.clip_model.text_projection.shape[1]
        self.proj_img2txt = nn.Linear(256, d_clip, bias=False)  # 假设融合前特征维度 256（与现版一致）
        nn.init.normal_(self.proj_img2txt.weight, std=0.02)

    def _load_prompts_df(self, textprompt_csv: str, per_fold_dir: str, fold_id: int) -> pd.DataFrame:
        # 优先 per_fold_dir/textprompt_fold{fold_id}.csv；否则用 textprompt_csv
        if per_fold_dir and os.path.isdir(per_fold_dir):
            fp = os.path.join(per_fold_dir, f"textprompt_fold{fold_id}.csv")
            if os.path.exists(fp):
                return pd.read_csv(fp)
        # 回退：单一 CSV
        if textprompt_csv and os.path.exists(textprompt_csv):
            df = pd.read_csv(textprompt_csv)
            # 如果有 fold 列，就只取当前折
            if "fold" in df.columns:
                df = df[df["fold"] == fold_id].copy()
                if len(df) == 0:
                    # 如果没有当前折，退回取全部（全局先验）
                    df = pd.read_csv(textprompt_csv)
            return df
        raise FileNotFoundError("No valid text prompt CSV found. Please check config.text.textprompt_csv / per_fold_dir.")

    @torch.no_grad()
    def build_class_prototypes(self, textprompt_csv: str, per_fold_dir: str, fold_id: int):
        df = self._load_prompts_df(textprompt_csv, per_fold_dir, fold_id)
        # 允许 class 或 label 命名；目标：三类 {0,1,2}
        if "class" in df.columns:
            cls_col = "class"
        elif "label" in df.columns:
            cls_col = "label"
        else:
            raise ValueError("text prompt CSV must contain a 'class' or 'label' column.")

        # 清洗
        df = df[[cls_col, "text"]].dropna()
        df[cls_col] = df[cls_col].astype(int)
        # 每类多句 → tokenize → text enc → 平均/最大
        protos = []
        for c in range(self.num_classes):
            sub = df[df[cls_col] == c]["text"].astype(str).tolist()
            if len(sub) == 0:
                # 若缺失，放一个简单句子防止崩
                sub = ["A clinical description of this class."]
            tokens = clip.tokenize(sub, truncate=True).to(self.device)
            feats = self.clip_model.encode_text(tokens)  # (N_c, d_clip)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            proto = feats.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            protos.append(proto)
        class_proto = torch.stack(protos, dim=0)  # (C, d_clip)
        self.class_proto.copy_(class_proto)

    def forward(self, feat_256: torch.Tensor):
        """
        feat_256: (B, 256) 融合前特征（来自 FusionHead.trunk 输出）
        返回：
          logits_text: (B, C)
          cos_sim:     (B, C)
        """
        x = self.proj_img2txt(feat_256)  # (B, d_clip)
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            p = self.class_proto / self.class_proto.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            p = self.class_proto
        cos = x @ p.t()  # (B, C)
        logits_text = cos / max(self.temperature, 1e-6)
        return logits_text, cos
