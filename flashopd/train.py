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


def _load_data(cfg: OPDConfig, tokenizer):
    if cfg.data_path.endswith(".jsonl") or cfg.data_path.endswith(".json"):
        ds = load_dataset("json", data_files=cfg.data_path, split="train")
    else:
        ds = load_dataset(cfg.data_path, split="train")

    def tokenize(examples):
        text_col = "text" if "text" in examples else list(examples.keys())[0]
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
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
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if cfg.use_lora:
        student = _apply_lora(student, cfg)

    # ---- 3. Teacher ----
    teacher = None
    if cfg.teacher_model or cfg.teacher_api_url:
        teacher = create_teacher(cfg, student_tokenizer=tokenizer)

    # ---- 4. Data ----
    dataset = _load_data(cfg, tokenizer)

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
        tokenizer=tokenizer,
    )

    # ---- 7. Train ----
    trainer.train()
    trainer.save_model()

    if rank == 0:
        print("\n  FlashOPD training complete!\n")
