# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.mind_encoder import MindEncoder
from src.models.fusion_head import FusionHead
from src.mynn.film import FiLM
from src.mynn.transformer_block import TransformerBlock


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SliceEncoder(nn.Module):
    """
    将 (B,S,1,H,W) 的切片堆栈编码为 (B,S,D) 的切片级 embedding。
    使用轻量 2D CNN + GAP + 线性投影。
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, emb_dim: int = 256):
        super().__init__()
        self.emb_dim = emb_dim
        ch = base_ch
        self.backbone = nn.Sequential(
            ConvBlock(in_ch,  ch,   k=3, s=1, p=1),
            ConvBlock(ch,     ch,   k=3, s=1, p=1),
            ConvBlock(ch,     ch*2, k=3, s=2, p=1),  # 下采样 1/2
            ConvBlock(ch*2,   ch*2, k=3, s=1, p=1),
            ConvBlock(ch*2,   ch*4, k=3, s=2, p=1),  # 下采样 1/4
            ConvBlock(ch*4,   ch*4, k=3, s=1, p=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch * 4, emb_dim)

    def forward(self, x):
        """
        x: (B,S,1,H,W)
        return: (B,S,D)
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        h = self.backbone(x)           # (B*S, C', H', W')
        h = self.pool(h).view(B * S, -1)
        h = self.proj(h)               # (B*S, D)
        h = h.view(B, S, self.emb_dim) # (B,S,D)
        return h


class SliceAggregator(nn.Module):
    """
    对 (B,S,D) 的切片序列做注意力池化：
      - 输出：z_img: (B,D) 全脑图像 embedding
             a:     (B,S) 每张切片的重要性权重
    """
    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, xs):
        """
        xs: (B,S,D)
        """
        B, S, D = xs.shape
        s = self.score(xs).squeeze(-1)      # (B,S)
        a = torch.softmax(s, dim=1)         # (B,S)
        z = torch.bmm(a.unsqueeze(1), xs).squeeze(1)  # (B,D)
        return z, a


class SubcortexPool(nn.Module):
    """
    对皮下 ROI 特征做注意力池化：
      - 输入: roi_sub_feat: (B,N_sub,D)
      - 输出: z_sub:   (B,out_dim)
             w_sub:   (B,N_sub)
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, roi_sub_feat):
        """
        roi_sub_feat: (B,N_sub,D)
        """
        B, N, D = roi_sub_feat.shape
        h = self.proj(roi_sub_feat)          # (B,N,out_dim)
        score = self.gate(h).squeeze(-1)     # (B,N)
        w = torch.softmax(score, dim=1)      # (B,N)
        z = torch.bmm(w.unsqueeze(1), h).squeeze(1)  # (B,out_dim)
        return z, w


class PPAL_MIND(nn.Module):
    """
    整体模型（升级版）：
      - Slice 分支：2D 切片编码 + Transformer + 注意力选片
      - Cortex+MIND 分支：基于皮层 ROI 特征 + MIND 图的图编码
      - Subcortex 分支：基于 HCPex 皮下 ROI 特征的注意力池化
      - Covariate 分支：年龄/性别/教育/种族等 tabular 特征
      - 融合头：FusionHead (接口与原版保持一致)
    """

    def __init__(self, cfg):
        super().__init__()
        # cfg 来自 engine_mind 中的 mcfg
        self.lambda_prior = float(getattr(cfg, "lambda_prior", 1.0))
        mind_sim = getattr(cfg, "mind_sim", "exp")
        mind_tau = float(getattr(cfg, "mind_tau", 1.0))
        cov_dim = int(getattr(cfg, "cov_dim", 4))
        # 皮层 ROI 数（与 MIND 的维度一致）
        self.n_cortex = int(getattr(cfg, "n_cortex", 360))

        # --- 切片分支 ---
        self.slice_dim = 256
        self.slice_enc = SliceEncoder(in_ch=1, base_ch=32, emb_dim=self.slice_dim)
        self.slice_tr = TransformerBlock(
            embedding_dim=self.slice_dim,
            num_heads=4,
            dropout=0.1,
            forward_expansion=4,      # ★ 必须加上
            activation="gelu"         # 可选，但推荐（你的 block 支持）
        )

        self.slice_agg = SliceAggregator(dim=self.slice_dim, dropout=0.1)

        # --- MIND 图分支（仅皮层 360 区） ---
        self.graph_dim = 256
        self.mind = MindEncoder(
            n_nodes=self.n_cortex,
            in_dim=self.slice_dim,    # 拼接 ROI 图像特征
            hid=256,
            out=self.graph_dim,
            sim_mode=mind_sim,
            tau=mind_tau,
            dropout=0.1,
            num_heads=4
        )

        # --- 皮下分支 ---
        self.sub_dim = 64
        self.sub_pool = SubcortexPool(
            in_dim=self.slice_dim,
            out_dim=self.sub_dim,
            dropout=0.1
        )

        # --- tabular / covariate 分支 ---
        self.demo_dim = 16
        self.tab_mlp = nn.Sequential(
            nn.LayerNorm(cov_dim),
            nn.Linear(cov_dim, self.demo_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # FiLM 条件调制（让 covariates 调节影像 & 图分支）
        self.film_img = FiLM(self.slice_dim, cov_dim)
        self.film_graph = FiLM(self.graph_dim, cov_dim)

        # --- 融合头 ---
        dim_tab = self.demo_dim + self.sub_dim
        self.head = FusionHead(
            dim_img=self.slice_dim,
            dim_graph=self.graph_dim,
            dim_tab=dim_tab,
            hidden=256,
            n_classes=3,
            hierarchical=True
        )

    def _roi_from_slices(self, xs, cover, n_cortex):
        """
        利用切片 embedding + slice_cover 近似构造 ROI 级特征：

          xs:    (B,S,D)   切片特征（经过 slice_tr 之后）
          cover: (B,S,R)   每张切片上各 ROI 的覆盖比例（HCPex label 统计而来，R>=n_cortex）
          n_cortex: 皮层 ROI 数（与 MIND 的 360 对齐）

        返回:
          roi_cortex_feat: (B,n_cortex,D)
          roi_sub_feat:    (B,n_sub,D) 或 None
        """
        B, S, D = xs.shape
        R = cover.size(-1)
        assert n_cortex <= R, f"n_cortex={n_cortex} > n_rois={R}"

        # --- Cortex: 使用前 n_cortex 个 ROI（与 MIND 对齐） ---
        cover_ctx = cover[:, :, :n_cortex]                    # (B,S,n_cortex)
        denom_ctx = cover_ctx.sum(dim=1, keepdim=True) + 1e-6 # (B,1,n_cortex)
        cover_ctx = cover_ctx / denom_ctx                     # 归一化为每个 ROI 的“跨切片权重”
        # (B,S,D) x (B,S,n_cortex) -> (B,n_cortex,D)
        roi_cortex_feat = torch.einsum("bsd,bsr->brd", xs, cover_ctx)

        # --- Subcortex: 剩余 ROI（HCPex 的 66 个皮下区） ---
        if R > n_cortex:
            cover_sub = cover[:, :, n_cortex:]                # (B,S,n_sub)
            denom_sub = cover_sub.sum(dim=1, keepdim=True) + 1e-6
            cover_sub = cover_sub / denom_sub
            roi_sub_feat = torch.einsum("bsd,bsu->bud", xs, cover_sub)  # (B,n_sub,D)
        else:
            roi_sub_feat = None

        return roi_cortex_feat, roi_sub_feat

    def forward(self, batch):
        """
        batch:
          img:         (B,S,1,H,W)
          mind:        (B,n_cortex,n_cortex)
          slice_cover: (B,S,R)   R = 360(+66) 对应 HCPex ROI
          cov:         (B,C)
        """
        x = batch["img"]            # (B,S,1,H,W)
        D = batch["mind"]           # (B,n_cortex,n_cortex)
        cov = batch["cov"]          # (B,C)
        cover = batch["slice_cover"]# (B,S,R)

        # --- 1) Slice 分支 ---
        xs = self.slice_enc(x)                 # (B,S,D)
        xs = self.slice_tr(xs, xs, xs)         # (B,S,D)
        z_img, a_slice = self.slice_agg(xs)    # (B,D), (B,S)

        # --- 2) 从切片构造 ROI 特征（皮层 + 皮下） ---
        roi_cortex_feat, roi_sub_feat = self._roi_from_slices(xs, cover, self.n_cortex)

        # --- 3) MIND 图分支（仅皮层） ---
        z_graph, w_node, _ = self.mind(D, roi_cortex_feat)    # (B,Dg), (B,n_cortex)

        # --- 4) 构造 slice 级 prior: 由 cortex 节点重要性 + 覆盖度倒推 ---
        cover_ctx = cover[:, :, :self.n_cortex]               # (B,S,n_cortex)
        p_slice = (cover_ctx * w_node[:, None, :]).sum(dim=2) # (B,S)
        p_slice = p_slice / (p_slice.sum(dim=1, keepdim=True) + 1e-6)

        # --- 5) Subcortex 分支 ---
        if roi_sub_feat is not None:
            z_sub, w_sub = self.sub_pool(roi_sub_feat)        # (B,sub_dim), (B,n_sub)
        else:
            B = x.size(0)
            z_sub = z_img.new_zeros(B, self.sub_dim)
            w_sub = None

        # --- 6) Covariate 分支 + FiLM 调制 ---
        z_demo = self.tab_mlp(cov)               # (B,d_demo)
        z_img_mod = self.film_img(z_img, cov)    # (B,D)
        z_graph_mod = self.film_graph(z_graph, cov)  # (B,Dg)

        # tabular 最终向量 = 人口学 + 皮下
        z_tab = torch.cat([z_demo, z_sub], dim=1)  # (B, d_demo+sub_dim)

        # --- 7) 融合预测 ---
        heads = self.head(z_img_mod, z_graph_mod, z_tab)

        out = {
            **heads,
            "slice_attn": a_slice,
            "slice_prior": p_slice,
            "node_w": w_node,
        }
        if w_sub is not None:
            out["subcort_w"] = w_sub

        return out
