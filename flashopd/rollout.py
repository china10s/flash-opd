"""Student 自回归 rollout：KV-cache 加速 + 灵活采样策略."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_token(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """从 logits 中采样一个 token：支持 greedy / top-k / top-p."""
    if top_k == 0 and top_p >= 1.0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals = torch.topk(logits, k, dim=-1).values
        logits[logits < topk_vals[..., -1:]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = (cum - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = logits.scatter(-1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)


@torch.no_grad()
def student_rollout(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 512,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Student 模型 on-policy rollout，返回生成的 token IDs.

    使用 KV cache 逐 token 生成，比 naive forward 快 5-10x。

    Args:
        model: HuggingFace CausalLM
        input_ids: (B, prompt_len) prompt token ids
        attention_mask: (B, prompt_len)
        max_new_tokens: 最大生成长度
        top_k / top_p / temperature: 采样参数
        eos_token_id: 遇到则停止

    Returns:
        generated_ids: (B, gen_len) 不含 prompt
    """
    B = input_ids.shape[0]
    device = input_ids.device

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    next_tok = _sample_token(out.logits[:, -1, :], top_k, top_p, temperature)

    gen = [next_tok]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None:
            finished = finished | (next_tok == eos_token_id)
        if finished.all():
            break

        out = model(
            input_ids=next_tok.unsqueeze(1),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_tok = _sample_token(out.logits[:, -1, :], top_k, top_p, temperature)

        if pad_token_id is not None:
            next_tok = next_tok.masked_fill(finished, pad_token_id)
        gen.append(next_tok)

    del past
    return torch.stack(gen, dim=1)
