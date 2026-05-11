"""Minimal local HuggingFace (Transformers) chat completion helper.

This is intentionally lightweight and lazy-imported so the rest of the project
can run without Transformers/Torch unless the user opts into local HF models.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass


def is_hf_local_model(model: str | None) -> bool:
    if not model:
        return False
    m = model.strip()
    return m.startswith("hf:") or m.startswith("hf/")


def parse_hf_model_id(model: str) -> str:
    m = model.strip()
    if m.startswith("hf:"):
        return m[len("hf:") :].strip()
    if m.startswith("hf/"):
        return m[len("hf/") :].strip()
    return m


@dataclass(frozen=True)
class HFGenerateKwargs:
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float | None = None
    device_map: str = "auto"
    torch_dtype: str | None = None  # e.g. "bfloat16", "float16"


_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple[object, object]] = {}


def _torch_dtype_from_str(dtype: str | None):
    if not dtype:
        return None
    try:
        import torch
    except Exception:
        return None
    s = dtype.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    return None


def _load_tokenizer_and_model(model_id_or_path: str, *, device_map: str, torch_dtype: str | None):
    # Lazy import to avoid hard dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch

    # Respect offline mode if user has downloaded weights.
    # (If they pass a repo id without local cache, this may still fail.)
    local_files_only = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")

    tok = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    dtype_obj = _torch_dtype_from_str(torch_dtype)

    # Primary path: use device_map (multi-device/sharded) when available.
    # Fallback path: if Accelerate is not installed, load without device_map.
    load_kwargs: dict = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": dtype_obj,
    }
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map=device_map,
            **load_kwargs,
        )
    except Exception as e:
        msg = str(e).lower()
        needs_accelerate = "requires `accelerate`" in msg or "requires accelerate" in msg
        if not needs_accelerate:
            raise
        allow_single_gpu_fallback = os.getenv("HF_LOCAL_ALLOW_SINGLE_GPU_FALLBACK", "").lower() in ("1", "true", "yes")
        if not allow_single_gpu_fallback:
            raise RuntimeError(
                "Local HF multi-GPU loading requested (device_map='auto') but `accelerate` is missing. "
                "Install accelerate to shard across GPUs: `pip install accelerate` (or `uv sync --extra rollout`). "
                "If you intentionally want single-GPU fallback, set HF_LOCAL_ALLOW_SINGLE_GPU_FALLBACK=1."
            ) from e

        print(
            "[local_hf] Warning: `accelerate` not available; falling back to single-device model load. "
            "This may cause GPU OOM on large models."
        )
        mdl = AutoModelForCausalLM.from_pretrained(model_id_or_path, **load_kwargs)
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(target_device)
    return tok, mdl


def get_hf_client(model_id_or_path: str, *, device_map: str = "auto", torch_dtype: str | None = None):
    key = json.dumps(
        {"id": model_id_or_path, "device_map": device_map, "torch_dtype": torch_dtype},
        sort_keys=True,
    )
    with _CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
    tok, mdl = _load_tokenizer_and_model(model_id_or_path, device_map=device_map, torch_dtype=torch_dtype)
    hf_device_map = getattr(mdl, "hf_device_map", None)
    if hf_device_map:
        devices = sorted({str(v) for v in hf_device_map.values()})
        print(
            f"[local_hf] loaded model='{model_id_or_path}' with device_map='{device_map}', "
            f"torch_dtype='{torch_dtype}', shards_on={devices}"
        )
    else:
        try:
            single_device = str(next(mdl.parameters()).device)
        except Exception:
            single_device = "unknown"
        print(
            f"[local_hf] loaded model='{model_id_or_path}' with device_map='{device_map}', "
            f"torch_dtype='{torch_dtype}', single_device='{single_device}'"
        )
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = (tok, mdl)
    return tok, mdl


def _fallback_chat_prompt(messages: list[dict]) -> str:
    # Simple, robust formatting if tokenizer doesn't have chat_template.
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}".strip())
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def hf_chat_completion_text(
    messages: list[dict],
    *,
    model_id_or_path: str,
    gen: HFGenerateKwargs | None = None,
) -> str:
    gen = gen or HFGenerateKwargs()

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Local HF models require `torch` and `transformers` installed. "
            "Try: `uv sync --extra rollout` (or install torch+transformers manually)."
        ) from e

    tok, mdl = get_hf_client(model_id_or_path, device_map=gen.device_map, torch_dtype=gen.torch_dtype)

    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = _fallback_chat_prompt(messages)

    inputs = tok(prompt, return_tensors="pt")

    # Some setups use device_map="auto" and place modules on multiple devices.
    # Token tensors need to be on a device accepted by the model; we pick the
    # embedding device when possible, otherwise fallback to first param device.
    try:
        device = next(mdl.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    generate_kwargs: dict = {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.temperature > 0,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
    }
    if gen.repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = gen.repetition_penalty

    with torch.inference_mode():
        out = mdl.generate(**inputs, **generate_kwargs)

    # Decode only newly generated tokens if possible
    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = out[0][input_len:]
    text = tok.decode(gen_tokens, skip_special_tokens=True)
    return (text or "").strip()

