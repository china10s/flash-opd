"""Loss 动态平衡器：fixed 固定权重 / dynamic EMA 自适应."""
from __future__ import annotations

import torch


class LossBalancer:
    """CE + KL 损失平衡.

    - fixed:  loss = ce_coef * CE + kl_coef * KL
    - dynamic: 用 EMA 跟踪 CE/KL 的量级，自动缩放使两者贡献对齐
    """

    def __init__(
        self,
        mode: str = "fixed",
        ce_coef: float = 1.0,
        kl_coef: float = 0.1,
        ema_decay: float = 0.99,
    ):
        self.mode = mode
        self.ce_coef = ce_coef
        self.kl_coef = kl_coef
        self.ema_decay = ema_decay

        self._ema_ce: float | None = None
        self._ema_kl: float | None = None

    def combine(self, ce_loss: torch.Tensor, kl_loss: torch.Tensor) -> torch.Tensor:
        """返回加权总 loss."""
        if self.mode == "fixed":
            return self.ce_coef * ce_loss + self.kl_coef * kl_loss

        ce_val = ce_loss.detach().item()
        kl_val = kl_loss.detach().item()

        if self._ema_ce is None:
            self._ema_ce = ce_val
            self._ema_kl = kl_val + 1e-8
        else:
            d = self.ema_decay
            self._ema_ce = d * self._ema_ce + (1 - d) * ce_val
            self._ema_kl = d * self._ema_kl + (1 - d) * kl_val

        ratio = self._ema_ce / (self._ema_kl + 1e-8)
        return self.ce_coef * ce_loss + self.kl_coef * ratio * kl_loss

    @property
    def stats(self) -> dict:
        return {
            "balancer/ema_ce": self._ema_ce or 0,
            "balancer/ema_kl": self._ema_kl or 0,
            "balancer/dynamic_ratio": (
                (self._ema_ce / (self._ema_kl + 1e-8)) if self._ema_ce else 0
            ),
        }
