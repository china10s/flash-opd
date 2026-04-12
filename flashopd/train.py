"""FlashOPD 训练入口：单文件完成从模型加载到训练完成的全流程.

CleanRL 哲学：所有逻辑可见、可 copy-paste、可 hack。
"""
from __future__ import annotations

import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from flashopd.config import OPDConfig
from flashopd.teacher import create_teacher
from flashopd.trainer import OPDTrainer


def _apply_lora(model, cfg: OPDConfig):
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in cfg.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_prompt(instruction: str, input_text: str = "") -> str:
    """将 instruction + input 组装成 prompt 文本."""
    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def prepare_dataset(cfg: OPDConfig, tokenizer):
    """加载并 tokenize 数据.

    支持两种数据格式：
      1. SFT 格式: {"instruction": "...", "input": "...", "output": "..."}
      2. 纯文本: {"text": "..."}（不区分 prompt/response，仅用于纯 CE 训练）

    SFT 格式会生成 labels，prompt 部分标记为 -100（不计算 CE loss），
    同时记录 prompt_length 供 OPD rollout 使用。
    """
    if cfg.data_path.endswith((".jsonl", ".json")):
        ds = load_dataset("json", data_files=cfg.data_path, split="train")
    else:
        ds = load_dataset(cfg.data_path, split="train")

    is_sft = "instruction" in ds.column_names and "output" in ds.column_names
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    IGNORE_INDEX = -100

    def tokenize_sft(example):
        prompt = build_prompt(example["instruction"], example.get("input", ""))
        response = example["output"]

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids
        prompt_len = len(prompt_ids)

        if len(full_ids) > cfg.max_seq_length:
            full_ids = full_ids[: cfg.max_seq_length]
            prompt_len = min(prompt_len, cfg.max_seq_length)

        labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
        attn_mask = [1] * len(full_ids)

        pad_len = cfg.max_seq_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
            attn_mask = attn_mask + [0] * pad_len

        return {
            "input_ids": full_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "prompt_length": prompt_len,
        }

    def tokenize_text(example):
        text_col = "text" if "text" in example else list(example.keys())[0]
        enc = tokenizer(
            example[text_col],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["prompt_length"] = 0
        return enc

    tokenize_fn = tokenize_sft if is_sft else tokenize_text
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


def run_training(cfg: OPDConfig):
    """从 OPDConfig 启动完整训练流程."""
    rank = int(os.getenv("RANK", "0"))

    if rank == 0:
        print(f"\n  FlashOPD v0.1.0")
        print(f"  Student: {cfg.student_model}")
        print(f"  Teacher: {cfg.teacher_model or cfg.teacher_api_url or 'None'}")
        print(f"  KL type: {cfg.kl_type} | T={cfg.temperature}")
        print(f"  CE={cfg.ce_coef} KL={cfg.kl_coef} balance={cfg.loss_balance}\n")

    # ---- 1. Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Student Model ----
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    if cfg.use_lora:
        student = _apply_lora(student, cfg)

    # ---- 3. Teacher ----
    teacher = None
    if cfg.teacher_model or cfg.teacher_api_url:
        teacher = create_teacher(cfg, student_tokenizer=tokenizer)

    # ---- 4. Data ----
    dataset = prepare_dataset(cfg, tokenizer)

    # ---- 5. Training Args ----
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        deepspeed=cfg.deepspeed,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # ---- 6. Trainer ----
    trainer = OPDTrainer(
        opd_config=cfg,
        teacher=teacher,
        model=student,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ---- 7. Train ----
    trainer.train()
    trainer.save_model()

    if rank == 0:
        print("\n  FlashOPD training complete!\n")
