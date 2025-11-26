# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def ce_loss(logits, targets, weight=None, label_smoothing=0.0):
    return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)

def focal_loss(logits, targets, weight=None, gamma=2.0, reduction="mean"):
    """
    经典 focal: CE * (1-p)^gamma
    """
    logpt = -F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(logpt)  # = softmax 的概率（正确类）
    loss = -(1 - pt) ** gamma * logpt
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def kl_symmetric(p, q, eps=1e-6):
    p = torch.clamp(p, min=eps); p = p / (p.sum(dim=1, keepdim=True) + eps)
    q = torch.clamp(q, min=eps); q = q / (q.sum(dim=1, keepdim=True) + eps)
    kl1 = (p * (p/q).log()).sum(dim=1)
    kl2 = (q * (q/p).log()).sum(dim=1)
    return (kl1 + kl2).mean()

def distill_kl(logits_s, logits_t, T=0.07):
    """
    蒸馏 KL: softmax(logits_s/T) || softmax(logits_t/T)
    """
    p = F.log_softmax(logits_s / T, dim=1)
    q = F.softmax(logits_t / T, dim=1)
    return F.kl_div(p, q, reduction="batchmean") * (T * T)
