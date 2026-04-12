"""Teacher 后端：本地模型 / vLLM OpenAI API，统一接口."""
from __future__ import annotations

import abc
from typing import Tuple

import torch


class TeacherBackend(abc.ABC):
    """Teacher 抽象基类——无论本地还是 API，对 Trainer 暴露统一接口."""

    @abc.abstractmethod
    def get_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """返回 teacher 对 input_ids 的完整 logits (B, L, V)."""

    @abc.abstractmethod
    def get_sparse_logprobs(
        self, input_ids: torch.Tensor, rollout_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 rollout 部分的 top-K logprobs.

        Returns:
            top_ids: (B, rollout_len, K)
            top_logprobs: (B, rollout_len, K)
        """

    @property
    @abc.abstractmethod
    def is_api(self) -> bool:
        ...


class LocalTeacher(TeacherBackend):
    """在同一进程内加载 teacher 模型（适合 teacher 较小或同机多卡）."""

    def __init__(self, model, device: torch.device | None = None):
        self.model = model
        self._device = device or next(model.parameters()).device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def is_api(self) -> bool:
        return False

    @torch.no_grad()
    def get_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def get_sparse_logprobs(
        self, input_ids: torch.Tensor, rollout_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("LocalTeacher 提供完整 logits，不需要 sparse 路径")


def _parse_logprob_entry(pos_data, K: int):
    """Parse one position's logprob data from vLLM API response.

    vLLM returns two possible dict formats:
      - flat:   {"token_id": logprob_float, ...}
      - nested: {"token_id": {"logprob": float, "rank": int, ...}, ...}
    Also handles list-of-dicts format from some API versions.
    """
    if isinstance(pos_data, dict):
        parsed = []
        for tok_id, val in pos_data.items():
            if isinstance(val, dict):
                lp = float(val.get("logprob", -100))
            else:
                lp = float(val)
            parsed.append((int(tok_id), lp))
        parsed.sort(key=lambda x: x[1], reverse=True)
        parsed = parsed[:K]
        t_ids = [p[0] for p in parsed]
        t_lps = [p[1] for p in parsed]
    elif isinstance(pos_data, list):
        items = sorted(pos_data, key=lambda x: x.get("logprob", -100), reverse=True)[:K]
        t_ids = [int(it.get("token_id", 0)) for it in items]
        t_lps = [float(it.get("logprob", -100)) for it in items]
    else:
        return [0] * K, [-100.0] * K

    t_ids = (t_ids + [0] * K)[:K]
    t_lps = (t_lps + [-100.0] * K)[:K]
    return t_ids, t_lps


class APITeacher(TeacherBackend):
    """通过 vLLM OpenAI-compatible API 获取 teacher logprobs（适合大模型分离部署）."""

    def __init__(
        self,
        api_url: str,
        model_name: str = "default",
        top_k: int = 20,
        timeout: int = 120,
        pad_token_id: int = 0,
    ):
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.top_k = top_k
        self.timeout = timeout
        self.pad_token_id = pad_token_id

        self._verify_connection()

    def _verify_connection(self):
        """Fail fast if the API teacher is unreachable."""
        import requests

        models_url = f"{self.api_url}/v1/models"
        try:
            resp = requests.get(models_url, timeout=5)
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
            if self.model_name and self.model_name not in available:
                print(
                    f"[FlashOPD] WARNING: model '{self.model_name}' not in "
                    f"API models {available}. Will use as-is."
                )
            print(
                f"[FlashOPD] API teacher connected: {self.api_url} "
                f"(models: {available})"
            )
        except requests.ConnectionError:
            raise ConnectionError(
                f"[FlashOPD] Cannot reach teacher API at {self.api_url}. "
                f"Please check that the vLLM server is running."
            )
        except requests.Timeout:
            raise ConnectionError(
                f"[FlashOPD] Teacher API at {self.api_url} timed out (5s). "
                f"Please check network connectivity."
            )

    @property
    def is_api(self) -> bool:
        return True

    def get_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("API 模式不返回完整 logits，请使用 get_sparse_logprobs")

    def get_sparse_logprobs(
        self, input_ids: torch.Tensor, rollout_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import requests

        B, S = input_ids.shape
        K = self.top_k
        device = input_ids.device

        all_ids, all_lps = [], []

        for b in range(B):
            ids = [t for t in input_ids[b].tolist() if t != self.pad_token_id]
            payload = {
                "model": self.model_name,
                "prompt": ids,
                "max_tokens": 1,
                "prompt_logprobs": K,
            }
            resp = requests.post(
                f"{self.api_url}/v1/completions", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            result = resp.json()

            plp = (
                result.get("choices", [{}])[0].get("prompt_logprobs")
                or result.get("prompt_logprobs")
                or []
            )

            start = max(0, len(ids) - rollout_len)
            b_ids, b_lps = [], []

            for pos in range(start, len(ids)):
                if pos < len(plp) and plp[pos] is not None:
                    pos_data = plp[pos]
                    t_ids, t_lps = _parse_logprob_entry(pos_data, K)
                else:
                    t_ids, t_lps = [0] * K, [-100.0] * K

                b_ids.append(t_ids)
                b_lps.append(t_lps)

            all_ids.append(b_ids)
            all_lps.append(b_lps)

        return (
            torch.tensor(all_ids, dtype=torch.long, device=device),
            torch.tensor(all_lps, dtype=torch.float32, device=device),
        )


def create_teacher(
    cfg,
    student_tokenizer=None,
) -> TeacherBackend:
    """根据 OPDConfig 自动选择 teacher 后端."""
    if cfg.teacher_backend == "api" and cfg.teacher_api_url:
        pad_id = student_tokenizer.pad_token_id if student_tokenizer else 0
        return APITeacher(
            api_url=cfg.teacher_api_url,
            model_name=cfg.teacher_api_model,
            top_k=cfg.teacher_api_logprobs,
            pad_token_id=pad_id or 0,
        )

    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return LocalTeacher(teacher)
