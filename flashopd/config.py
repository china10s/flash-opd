"""OPD 全局配置：一个 dataclass 统管所有蒸馏超参."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class OPDConfig:
    """On-Policy Distillation 完整配置.

    设计原则（学习 CleanRL 的"所有参数一目了然"）：
    将 loss / rollout / teacher / scheduler 全部收进同一个 dataclass，
    用户只需填一份 YAML 即可启动训练，无需在多个类之间跳转。
    """

    # ---- 模型路径 ----
    student_model: str = ""
    teacher_model: str = ""

    # ---- Loss 系数 ----
    ce_coef: float = 1.0
    kl_coef: float = 0.1

    # ---- Loss 平衡策略 ----
    loss_balance: Literal["fixed", "dynamic"] = "fixed"
    ema_decay: float = 0.99  # dynamic 模式的 EMA 衰减系数

    # ---- KL 散度 ----
    kl_type: Literal["reverse", "forward", "jsd"] = "reverse"
    temperature: float = 1.0
    kl_clip_min: float = 0.0  # 0 = 不截断
    kl_clip_max: float = 0.0  # 0 = 不截断
    kl_top_k: int = 0         # 0 = 全词表

    # ---- Student Rollout ----
    max_new_tokens: int = 512
    rollout_top_k: int = 0       # 0 = greedy argmax
    rollout_top_p: float = 1.0   # 1.0 = 不过滤
    rollout_temperature: float = 1.0

    # ---- OPD 调度 ----
    disable_after_ratio: float = 1.0  # 1.0 = 不关闭

    # ---- Teacher Backend ----
    teacher_backend: Literal["local", "api"] = "local"
    teacher_api_url: str = ""
    teacher_api_model: str = "default"
    teacher_api_logprobs: int = 20  # vLLM 默认上限为 20，需启动时 --max-logprobs N 提高

    # ---- 训练超参 ----
    learning_rate: float = 5e-6
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.05
    bf16: bool = True
    deepspeed: Optional[str] = None

    # ---- 保存 & 日志 ----
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3

    # ---- 数据 ----
    data_path: str = ""
    max_seq_length: int = 4096

    # ---- LoRA（可选）----
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

    @classmethod
    def from_yaml(cls, path: str) -> "OPDConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        import yaml
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    @property
    def is_greedy_rollout(self) -> bool:
        return self.rollout_top_k == 0 and self.rollout_top_p >= 1.0

    @property
    def kl_has_clip(self) -> bool:
        return self.kl_clip_min > 0 or self.kl_clip_max > 0
