"""Teacher 后端：本地模型 / vLLM OpenAI API，统一接口."""
from __future__ import annotations

import abc
import os
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

    def get_think_then_score_logprobs(
        self,
        prompt_ids: torch.Tensor,
        rollout_ids: torch.Tensor,
        tokenizer,
        think_cfg: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Think-then-Score: teacher 先思考再评分.

        子类可覆盖，默认 fallback 到 get_sparse_logprobs.
        """
        full_ids = torch.cat([prompt_ids, rollout_ids], dim=1)
        return self.get_sparse_logprobs(full_ids, rollout_ids.shape[1])

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

    def _score_with_prompt_logprobs(
        self, token_ids: list[int], rollout_len: int, K: int
    ) -> Tuple[list, list]:
        """Send token_ids to API with prompt_logprobs, extract rollout portion."""
        import requests

        payload = {
            "model": self.model_name,
            "prompt": token_ids,
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

        start = max(0, len(token_ids) - rollout_len)
        b_ids, b_lps = [], []

        for pos in range(start, len(token_ids)):
            if pos < len(plp) and plp[pos] is not None:
                t_ids, t_lps = _parse_logprob_entry(plp[pos], K)
            else:
                t_ids, t_lps = [0] * K, [-100.0] * K
            b_ids.append(t_ids)
            b_lps.append(t_lps)

        pad_ids, pad_lps = [0] * K, [-100.0] * K
        while len(b_ids) < rollout_len:
            b_ids.append(pad_ids)
            b_lps.append(pad_lps)
        return b_ids[:rollout_len], b_lps[:rollout_len]

    def get_sparse_logprobs(
        self, input_ids: torch.Tensor, rollout_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = input_ids.shape[0]
        K = self.top_k
        device = input_ids.device

        all_ids, all_lps = [], []
        for b in range(B):
            ids = [t for t in input_ids[b].tolist() if t != self.pad_token_id]
            b_ids, b_lps = self._score_with_prompt_logprobs(ids, rollout_len, K)
            all_ids.append(b_ids)
            all_lps.append(b_lps)

        return (
            torch.tensor(all_ids, dtype=torch.long, device=device),
            torch.tensor(all_lps, dtype=torch.float32, device=device),
        )

    def get_think_then_score_logprobs(
        self,
        prompt_ids: torch.Tensor,
        rollout_ids: torch.Tensor,
        tokenizer,
        think_cfg: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Think-then-Score: 两阶段 API 调用.

        Pass 1: teacher 对 prompt 生成 <think>...</think> 思考过程
        Pass 2: [prompt + think + student_rollout] → prompt_logprobs 对 rollout 打分
        """
        import requests

        B = prompt_ids.shape[0]
        rollout_len = rollout_ids.shape[1]
        K = self.top_k
        device = prompt_ids.device

        think_max = think_cfg.get("max_tokens", 1024)
        think_temp = think_cfg.get("temperature", 0.7)
        think_top_p = think_cfg.get("top_p", 0.6)
        think_top_k = think_cfg.get("top_k", 20)
        think_timeout = min(think_max // 5 + 60, 480)

        all_ids, all_lps = [], []
        rank = int(os.getenv("RANK", "0"))

        for b in range(B):
            p_ids = [t for t in prompt_ids[b].tolist() if t != self.pad_token_id]
            r_ids = rollout_ids[b].tolist()

            # Pass 1: teacher 生成思考过程
            gen_payload = {
                "model": self.model_name,
                "prompt": p_ids,
                "max_tokens": think_max,
                "temperature": think_temp,
                "top_p": think_top_p,
                "top_k": think_top_k,
                "repetition_penalty": 1.05,
                "stop": ["</think>"],
            }
            try:
                gen_resp = requests.post(
                    f"{self.api_url}/v1/completions",
                    json=gen_payload,
                    timeout=think_timeout,
                )
                gen_resp.raise_for_status()
                think_text = gen_resp.json().get("choices", [{}])[0].get("text", "")
            except Exception as e:
                if rank == 0:
                    print(f"[FlashOPD] Think API failed ({type(e).__name__}), fallback")
                all_ids.append([[0] * K] * rollout_len)
                all_lps.append([[-100.0] * K] * rollout_len)
                continue

            think_token_ids = tokenizer.encode(think_text, add_special_tokens=False)
            end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
            think_token_ids = think_token_ids + end_think_ids

            # Pass 2: [prompt + think + student_rollout] → prompt_logprobs
            full_ids = p_ids + think_token_ids + r_ids

            if not hasattr(self, "_debug_think_printed"):
                self._debug_think_printed = False
            if not self._debug_think_printed and rank == 0:
                self._debug_think_printed = True
                print(
                    f"\n{'=' * 60}\n"
                    f"[FlashOPD] Think-then-Score (first sample)\n"
                    f"  prompt: {len(p_ids)} tokens\n"
                    f"  think:  {len(think_token_ids)} tokens\n"
                    f"  rollout: {len(r_ids)} tokens\n"
                    f"  total:  {len(full_ids)} tokens\n"
                    f"  think preview: {think_text[:300]}\n"
                    f"{'=' * 60}"
                )

            try:
                b_ids, b_lps = self._score_with_prompt_logprobs(
                    full_ids, rollout_len, K
                )
            except Exception as e:
                if rank == 0:
                    print(f"[FlashOPD] Score API failed ({type(e).__name__}), fallback")
                b_ids = [[0] * K] * rollout_len
                b_lps = [[-100.0] * K] * rollout_len

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

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = "auto"

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    return LocalTeacher(teacher)
