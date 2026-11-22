"""Runtime precision policies for W1 classifier inference."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

from twc.repro.public_artifacts import validate_public_no_raw_payload
from twc.repro.hashing import stable_json_hash


W1_PRECISION_FP32 = "fp32"
W1_PRECISION_TF32 = "tf32"
W1_PRECISION_AMP_BF16 = "amp_bf16"
W1_PRECISION_MODES = (W1_PRECISION_FP32, W1_PRECISION_TF32, W1_PRECISION_AMP_BF16)


def normalize_w1_precision_mode(value: str | None) -> str:
    mode = W1_PRECISION_FP32 if value is None else str(value)
    if mode not in W1_PRECISION_MODES:
        raise ValueError(f"unsupported W1 precision mode: {mode}")
    return mode


def build_w1_precision_policy(mode: str | None) -> dict[str, Any]:
    normalized = normalize_w1_precision_mode(mode)
    payload = _w1_precision_policy_payload(normalized)
    payload["precision_policy_hash"] = stable_json_hash(payload)
    validate_w1_precision_policy(payload)
    return payload


def _w1_precision_policy_payload(normalized: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_stage": "w1_window_mil_inference",
        "precision_mode": normalized,
        "torch_autocast_enabled": normalized == W1_PRECISION_AMP_BF16,
        "torch_autocast_dtype": "bfloat16" if normalized == W1_PRECISION_AMP_BF16 else None,
        "torch_allow_tf32": normalized in {W1_PRECISION_TF32, W1_PRECISION_AMP_BF16},
        "float32_matmul_precision": "high" if normalized in {W1_PRECISION_TF32, W1_PRECISION_AMP_BF16} else "highest",
        "weights_dtype": "float32",
        "classifier_behavior_scope": "precision_policy_only",
    }


def validate_w1_precision_policy(payload: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(payload)
    if payload.get("schema_version") != 1:
        raise ValueError("W1 precision policy schema_version mismatch")
    if payload.get("model_stage") != "w1_window_mil_inference":
        raise ValueError("W1 precision policy stage mismatch")
    mode = normalize_w1_precision_mode(str(payload.get("precision_mode") or ""))
    expected = _w1_precision_policy_payload(mode)
    expected["precision_policy_hash"] = stable_json_hash(expected)
    if dict(payload) != expected:
        raise ValueError("W1 precision policy payload mismatch")


def apply_w1_precision_policy(torch: Any, mode: str | None) -> dict[str, Any]:
    policy = build_w1_precision_policy(mode)
    allow_tf32 = bool(policy["torch_allow_tf32"])
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        setattr(matmul_backend, "allow_tf32", allow_tf32)
    cudnn_backend = getattr(backends, "cudnn", None) if backends is not None else None
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        setattr(cudnn_backend, "allow_tf32", allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(policy["float32_matmul_precision"]))
    return policy


def w1_precision_context(torch: Any, *, mode: str | None, device_type: str) -> Any:
    normalized = normalize_w1_precision_mode(mode)
    if normalized != W1_PRECISION_AMP_BF16:
        return nullcontext()
    if device_type != "cuda":
        raise RuntimeError("amp_bf16 W1 precision requires a CUDA device")
    dtype = getattr(torch, "bfloat16", None)
    if dtype is None:
        raise RuntimeError("amp_bf16 W1 precision requires torch.bfloat16")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def runtime_w1_precision_observation(
    torch: Any,
    *,
    mode: str | None,
    device: Any | None = None,
) -> dict[str, Any]:
    policy = build_w1_precision_policy(mode)
    cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    device_type = str(getattr(device, "type", "") or ("cuda" if cuda_available else "cpu"))
    payload: dict[str, Any] = {
        "precision_policy": policy,
        "precision_policy_hash": policy["precision_policy_hash"],
        "torch_version": str(getattr(torch, "__version__", "") or ""),
        "torch_cuda_runtime": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
        "cuda_available": cuda_available,
        "device_type": device_type,
    }
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        payload["torch_matmul_allow_tf32_observed"] = bool(getattr(matmul_backend, "allow_tf32"))
    cudnn_backend = getattr(backends, "cudnn", None) if backends is not None else None
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        payload["torch_cudnn_allow_tf32_observed"] = bool(getattr(cudnn_backend, "allow_tf32"))
    if cuda_available and device_type == "cuda":
        current_device = int(getattr(torch.cuda, "current_device", lambda: 0)())
        payload["cuda_device_index"] = current_device
        payload["cuda_device_name"] = str(getattr(torch.cuda, "get_device_name", lambda _index: "")(current_device))
    validate_public_no_raw_payload(payload)
    return payload
