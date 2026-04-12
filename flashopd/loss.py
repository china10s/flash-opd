"""KL 散度损失函数：forward / reverse / JSD / top-k sparse，支持温度缩放."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_type: str = "reverse",
    temperature: float = 1.0,
    top_k: int = 0,
    chunk_size: int = 128,
) -> torch.Tensor:
    """计算 token 级 KL 散度，自动分块避免 OOM.

    Args:
        student_logits: (B, L, V)
        teacher_logits: (B, L, V)
        kl_type: "reverse" = KL(s||t), "forward" = KL(t||s)
        temperature: softmax 温度
        top_k: >0 时仅在 teacher top-k token 上计算
        chunk_size: 沿 seq 维分块大小

    Returns:
        标量 KL loss（已做 T² 缩放）
    """
    min_v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = student_logits[:, :, :min_v]
    t = teacher_logits[:, :, :min_v]

    B, L, _ = s.shape
    total = s.new_tensor(0.0)

    for i in range(0, L, chunk_size):
        j = min(i + chunk_size, L)
        sc, tc = s[:, i:j], t[:, i:j]

        if 0 < top_k < sc.shape[-1]:
            idx = tc.topk(top_k, dim=-1).indices
            sc = sc.gather(-1, idx)
            tc = tc.gather(-1, idx)

        s_lp = F.log_softmax(sc / temperature, dim=-1)
        t_lp = F.log_softmax(tc / temperature, dim=-1)

        if kl_type == "forward":
            t_p = F.softmax(tc / temperature, dim=-1).detach()
            total = total + (t_p * (t_lp.detach() - s_lp)).sum()
        else:  # reverse
            s_p = F.softmax(sc / temperature, dim=-1)
            total = total + (s_p * (s_lp - t_lp.detach())).sum()

    return total / (B * L) * (temperature ** 2)


def jsd_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 0.5,
    top_k: int = 0,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Jensen-Shannon 散度 = alpha * KL(s||m) + (1-alpha) * KL(t||m)."""
    min_v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = student_logits[:, :, :min_v]
    t = teacher_logits[:, :, :min_v]

    B, L, _ = s.shape
    total = s.new_tensor(0.0)

    for i in range(0, L, chunk_size):
        j = min(i + chunk_size, L)
        sc, tc = s[:, i:j], t[:, i:j]

        if 0 < top_k < sc.shape[-1]:
            idx = tc.topk(top_k, dim=-1).indices
            sc = sc.gather(-1, idx)
            tc = tc.gather(-1, idx)

        s_p = F.softmax(sc / temperature, dim=-1)
        t_p = F.softmax(tc / temperature, dim=-1).detach()
        m_p = alpha * s_p + (1 - alpha) * t_p
        m_lp = m_p.log()

        kl_s = (s_p * (s_p.log() - m_lp)).sum()
        kl_t = (t_p * (t_p.log() - m_lp)).sum()
        total = total + alpha * kl_s + (1 - alpha) * kl_t

    return total / (B * L) * (temperature ** 2)


def kl_from_sparse_logprobs(
    student_logits: torch.Tensor,
    teacher_top_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    kl_type: str = "reverse",
    temperature: float = 1.0,
) -> torch.Tensor:
    """使用 API 返回的 sparse logprobs 计算 KL（无需 teacher 完整 logits）.

    Args:
        student_logits: (B, L, V)
        teacher_top_ids: (B, L, K)
        teacher_top_logprobs: (B, L, K) — 已是 log prob
    """
    T = temperature
    B, L, K = teacher_top_ids.shape

    s_lp = F.log_softmax(student_logits / T, dim=-1)
    s_lp_at_t = s_lp.gather(-1, teacher_top_ids)

    if T != 1.0:
        t_lp = teacher_top_logprobs / T
        t_lp = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)
    else:
        t_lp = teacher_top_logprobs

    if kl_type == "forward":
        t_p = t_lp.exp().detach()
        kl = (t_p * (t_lp.detach() - s_lp_at_t)).sum()
    else:
        s_p = s_lp_at_t.exp()
        kl = (s_p * (s_lp_at_t - t_lp.detach())).sum()

    return kl / (B * L) * (T ** 2)


def clip_kl(kl: torch.Tensor, kl_min: float = 0.0, kl_max: float = 0.0) -> torch.Tensor:
    """对 KL loss 做上下界截断."""
    if kl_min > 0:
        kl = torch.clamp(kl, min=kl_min)
    if kl_max > 0:
        kl = torch.clamp(kl, max=kl_max)
    return kl
