# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def d2s(D, mode: str = "exp", tau: float = 1.0, eps: float = 1e-6):
    """
    将 MIND "距离"/散度矩阵 D (B,N,N) 转为相似度矩阵 S (B,N,N)，并做对称化与 [0,1] 裁剪。
    默认使用 z-score + exp(-z/tau) 形式。
    """
    if mode == "exp":
        mu = D.mean(dim=(1, 2), keepdim=True)
        sd = D.std(dim=(1, 2), keepdim=True) + eps
        Z = (D - mu) / sd
        S = torch.exp(-Z / max(tau, eps))
    else:
        S = 1.0 / (1.0 + D)

    S = (S + S.transpose(1, 2)) * 0.5
    return S.clamp(0.0, 1.0)


class MindEncoder(nn.Module):
    """
    升级版 MIND 图编码器：

    输入:
      - D: (B, N, N)   MIND 矩阵 (N = 皮层 ROI 数，例如 360)
      - roi_feat: (B, N, F) 皮层 ROI 级图像特征 (可选，可为 None)

    输出:
      - z:      (B, out)   图级 embedding
      - w_node: (B, N)     节点注意力（可解释：每个皮层 ROI 的重要性）
      - S:      (B, N, N)  相似度矩阵 (d2s(D) 结果)

    核心思路：
      * 每个节点的基础特征 = 该节点在 MIND 网络中的一整行（与所有 ROI 的相似性）
      * 可选地拼接来自切片分支的 ROI 图像特征
      * 经 MLP + 多头 Self-Attention 进行“节点间交互”
      * gate 得到节点重要性分布 w_node，加权汇聚得到图级表示 z
    """

    def __init__(
        self,
        n_nodes: int = 360,
        in_dim: int = 256,
        hid: int = 256,
        out: int = 256,
        sim_mode: str = "exp",
        tau: float = 1.0,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n = int(n_nodes)
        self.in_dim = int(in_dim)
        self.sim_mode = sim_mode
        self.tau = float(tau)

        # 输入维度 = MIND 行向量维度 (N) + ROI 图像特征维度 (in_dim)
        self.linear_in = nn.Linear(self.n + self.in_dim, hid)
        self.linear_h = nn.Linear(hid, out)
        self.norm_in = nn.LayerNorm(hid)
        self.norm_attn = nn.LayerNorm(out)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=out, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.node_gate = nn.Sequential(
            nn.LayerNorm(out),
            nn.Linear(out, out // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out // 2, 1),
        )

    def forward(self, D: torch.Tensor, roi_feat: torch.Tensor = None):
        """
        D: (B, N, N)  MIND 矩阵
        roi_feat: (B, N, F) ROI 图像特征；若为 None，则仅使用 MIND 行向量
        """
        B, N, _ = D.shape
        assert N == self.n, f"MindEncoder: expect n_nodes={self.n}, got {N}"

        device = D.device
        # 距离 -> 相似度
        S = d2s(D, self.sim_mode, self.tau)  # (B, N, N)

        # 基础网络特征: 每个节点的 MIND 行向量 (B, N, N)
        base_feat = S

        if roi_feat is not None:
            assert roi_feat.shape[:2] == (B, N), \
                f"roi_feat shape must be (B,{N},F), got {tuple(roi_feat.shape)}"
            # 若 ROI 特征维度与 in_dim 不一致，应在外部对齐；此处直接拼接
            X_in = torch.cat([base_feat, roi_feat], dim=-1)  # (B, N, N+F)
        else:
            # 若没有 ROI 特征，拼接零向量占位以保持输入维度一致
            zeros = torch.zeros(B, N, self.in_dim, device=device, dtype=D.dtype)
            X_in = torch.cat([base_feat, zeros], dim=-1)     # (B, N, N+in_dim)

        # MLP -> out 维度
        X = self.linear_in(X_in).relu()
        X = self.norm_in(X)
        X = self.linear_h(X).relu()                         # (B, N, out)

        # 一层 Multi-Head Self-Attention（节点间交互）
        X_attn, _ = self.self_attn(X, X, X)                 # (B, N, out)
        X = self.norm_attn(X + X_attn)                      # 残差 + LN

        # 节点 gate -> 注意力权重
        gate = self.node_gate(X).squeeze(-1)                # (B, N)
        w = torch.softmax(gate, dim=1)                      # (B, N)

        # 图级表示：注意力加权求和
        z = torch.bmm(w.unsqueeze(1), X).squeeze(1)         # (B, out)

        return z, w, S
