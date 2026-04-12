"""OPDTrainer: 核心训练循环，将 rollout / teacher / loss / balancer 粘合在一起."""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
from transformers import Trainer, TrainerCallback

from flashopd.balancer import LossBalancer
from flashopd.config import OPDConfig
from flashopd.loss import clip_kl, jsd_divergence, kl_divergence, kl_from_sparse_logprobs
from flashopd.rollout import student_rollout
from flashopd.teacher import TeacherBackend


class OPDTrainer(Trainer):
    """On-Policy Distillation Trainer.

    在 HuggingFace Trainer 基础上注入 OPD 逻辑：
    1. 每个 training step 先做 student rollout（on-policy 序列）
    2. 用 teacher 对 rollout 序列产生 logits / logprobs
    3. 计算 KL loss 并与 CE loss 加权组合
    4. 支持训练进度调度（disable_after_ratio）

    使用方式与标准 HuggingFace Trainer 完全一致，
    额外传入 OPDConfig 和 TeacherBackend 即可。
    """

    def __init__(
        self,
        opd_config: OPDConfig,
        teacher: TeacherBackend | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opd_cfg = opd_config
        self.teacher = teacher
        self.balancer = LossBalancer(
            mode=opd_config.loss_balance,
            ce_coef=opd_config.ce_coef,
            kl_coef=opd_config.kl_coef,
            ema_decay=opd_config.ema_decay,
        )

        self._opd_stats: Dict[str, float] = {}
        self._step_times: Dict[str, float] = {}

        self.add_callback(_OPDProgressCallback(self))

    @property
    def opd_active(self) -> bool:
        if self.teacher is None:
            return False
        if self.state.max_steps <= 0:
            return True
        progress = self.state.global_step / self.state.max_steps
        return progress < self.opd_cfg.disable_after_ratio

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override: 在 CE 基础上注入 OPD KL loss."""
        t0 = time.perf_counter()

        outputs = model(**inputs)
        ce_loss = outputs.loss

        if not self.opd_active:
            self._opd_stats = {"opd/active": 0.0, "opd/ce_loss": ce_loss.item()}
            return (ce_loss, outputs) if return_outputs else ce_loss

        cfg = self.opd_cfg
        tokenizer = self.tokenizer

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None:
            self._opd_stats = {"opd/active": 1.0, "opd/ce_loss": ce_loss.item(), "opd/kl_loss": 0.0}
            return (ce_loss, outputs) if return_outputs else ce_loss

        # ---- Step 1: Student Rollout ----
        t_rollout = time.perf_counter()
        gen_ids = student_rollout(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            top_k=cfg.rollout_top_k,
            top_p=cfg.rollout_top_p,
            temperature=cfg.rollout_temperature,
            eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            pad_token_id=tokenizer.pad_token_id if tokenizer else None,
        )
        rollout_len = gen_ids.shape[1]
        self._step_times["rollout_ms"] = (time.perf_counter() - t_rollout) * 1000

        if rollout_len == 0:
            self._opd_stats = {"opd/active": 1.0, "opd/ce_loss": ce_loss.item(), "opd/kl_loss": 0.0}
            return (ce_loss, outputs) if return_outputs else ce_loss

        full_ids = torch.cat([input_ids, gen_ids], dim=1)

        # ---- Step 2: Teacher Forward ----
        t_teacher = time.perf_counter()
        if self.teacher.is_api:
            teacher_top_ids, teacher_top_lps = self.teacher.get_sparse_logprobs(
                full_ids, rollout_len
            )
        else:
            full_mask = torch.ones_like(full_ids)
            teacher_logits = self.teacher.get_logits(full_ids, full_mask)
            teacher_logits = teacher_logits[:, -rollout_len - 1 : -1, :]
        self._step_times["teacher_ms"] = (time.perf_counter() - t_teacher) * 1000

        # ---- Step 3: Student Forward on rollout ----
        t_student = time.perf_counter()
        model.train()
        student_out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
        student_logits = student_out.logits[:, -rollout_len - 1 : -1, :]
        self._step_times["student_fwd_ms"] = (time.perf_counter() - t_student) * 1000

        # ---- Step 4: KL Loss ----
        t_kl = time.perf_counter()
        if self.teacher.is_api:
            kl_loss = kl_from_sparse_logprobs(
                student_logits, teacher_top_ids, teacher_top_lps,
                kl_type=cfg.kl_type, temperature=cfg.temperature,
            )
        elif cfg.kl_type == "jsd":
            kl_loss = jsd_divergence(
                student_logits, teacher_logits,
                temperature=cfg.temperature, top_k=cfg.kl_top_k,
            )
        else:
            kl_loss = kl_divergence(
                student_logits, teacher_logits,
                kl_type=cfg.kl_type, temperature=cfg.temperature, top_k=cfg.kl_top_k,
            )

        if cfg.kl_has_clip:
            kl_loss = clip_kl(kl_loss, cfg.kl_clip_min, cfg.kl_clip_max)
        self._step_times["kl_ms"] = (time.perf_counter() - t_kl) * 1000

        # ---- Step 5: Combine ----
        total_loss = self.balancer.combine(ce_loss, kl_loss)

        progress = self.state.global_step / max(self.state.max_steps, 1)
        self._opd_stats = {
            "opd/active": 1.0,
            "opd/ce_loss": ce_loss.item(),
            "opd/kl_loss": kl_loss.item(),
            "opd/total_loss": total_loss.item(),
            "opd/rollout_len": float(rollout_len),
            "opd/progress": progress,
            **{f"opd/{k}": v for k, v in self._step_times.items()},
            **{f"opd/{k}": v for k, v in self.balancer.stats.items()},
        }

        self._step_times["total_opd_ms"] = (time.perf_counter() - t0) * 1000

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float]) -> None:
        if self._opd_stats:
            logs.update(self._opd_stats)
        super().log(logs)


class _OPDProgressCallback(TrainerCallback):
    """在训练日志中注入 OPD 状态信息."""

    def __init__(self, trainer: OPDTrainer):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        cfg = self._trainer.opd_cfg
        rank = int(os.getenv("RANK", "0"))
        if rank == 0:
            print(
                f"\n{'=' * 60}\n"
                f"  FlashOPD v0.1 — On-Policy Distillation\n"
                f"  KL type: {cfg.kl_type} | T: {cfg.temperature}\n"
                f"  CE coef: {cfg.ce_coef} | KL coef: {cfg.kl_coef}\n"
                f"  Balance: {cfg.loss_balance}\n"
                f"  Rollout: max_new_tokens={cfg.max_new_tokens}"
                f"  top_k={cfg.rollout_top_k} top_p={cfg.rollout_top_p}\n"
                f"  Teacher: {'API' if cfg.teacher_backend == 'api' else 'Local'}\n"
                f"  Disable after: {cfg.disable_after_ratio:.0%}\n"
                f"{'=' * 60}\n"
            )
