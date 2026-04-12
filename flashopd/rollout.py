"""Student 自回归 rollout：基于 model.generate() 的可靠实现.

使用 HuggingFace 原生 generate()，自动处理 GQA KV-cache、
attention_mask 扩展、position_ids 等所有模型架构差异。
"""
from __future__ import annotations

import torch


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

    内部使用 model.generate()，天然兼容所有 HuggingFace 模型架构
    （MHA / GQA / MQA），无需手动管理 KV cache。

    Args:
        model: HuggingFace CausalLM（支持 PEFT 包装）
        input_ids: (B, prompt_len) prompt token ids
        attention_mask: (B, prompt_len)
        max_new_tokens: 最大生成长度
        top_k / top_p / temperature: 采样参数（全 0/1.0 = greedy）
        eos_token_id: 遇到则停止
        pad_token_id: padding token id

    Returns:
        generated_ids: (B, gen_len) 仅包含生成部分，不含 prompt
    """
    is_training = model.training
    model.eval()

    do_sample = not (top_k == 0 and top_p >= 1.0)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    outputs = model.generate(**gen_kwargs)

    if is_training:
        model.train()

    prompt_len = input_ids.shape[1]
    return outputs[:, prompt_len:]
