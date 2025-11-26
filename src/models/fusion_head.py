# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, dim_img=256, dim_graph=256, dim_tab=16, hidden=256, n_classes=3, hierarchical=True):
        super().__init__()
        self.hierarchical = hierarchical
        in_dim = dim_img + dim_graph + dim_tab
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU()
        )
        self.head_3way = nn.Linear(hidden, n_classes)
        if hierarchical:
            self.head_ad_vs_non = nn.Linear(hidden, 2)
            self.head_nc_mci    = nn.Linear(hidden, 2)

    def forward(self, F_img, Z_graph, Z_tab):
        x = torch.cat([F_img, Z_graph, Z_tab], dim=1)
        h = self.trunk(x)         # (B, hidden) 作为“融合前特征”
        out = {"logits_3": self.head_3way(h), "feat": h}
        if self.hierarchical:
            out["logits_ad_non"] = self.head_ad_vs_non(h)
            out["logits_nc_mci"] = self.head_nc_mci(h)
        return out
