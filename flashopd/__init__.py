"""
FlashOPD — 300行核心代码实现 On-Policy Distillation

一个极简、高效、可扩展的 LLM 在线蒸馏框架。
"""
__version__ = "0.1.0"

from flashopd.config import OPDConfig
from flashopd.loss import kl_divergence, jsd_divergence
from flashopd.rollout import student_rollout
from flashopd.teacher import LocalTeacher, APITeacher, create_teacher
from flashopd.balancer import LossBalancer
from flashopd.trainer import OPDTrainer

__all__ = [
    "OPDConfig",
    "kl_divergence",
    "jsd_divergence",
    "student_rollout",
    "LocalTeacher",
    "APITeacher",
    "create_teacher",
    "LossBalancer",
    "OPDTrainer",
]
