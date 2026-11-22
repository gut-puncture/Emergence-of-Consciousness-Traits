"""Shared Window-MIL model, tokenization, pooling, and batch primitives."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any


DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"

W1_ATTN_IMPLEMENTATION_SDPA = "sdpa"

W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2 = "flash_attention_2"

SUPPORTED_W1_ATTN_IMPLEMENTATIONS = frozenset(
    {W1_ATTN_IMPLEMENTATION_SDPA, W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2}
)

WINDOW_POOLING_MAX = "max"

WINDOW_POOLING_TOPK_MEAN = "topk_mean_logits"

SUPPORTED_WINDOW_POOLING_MODES = frozenset({WINDOW_POOLING_MAX, WINDOW_POOLING_TOPK_MEAN})


def normalize_w1_attn_implementation(value: str | None) -> str:
    normalized = str(value or W1_ATTN_IMPLEMENTATION_SDPA).strip().lower()
    if normalized not in SUPPORTED_W1_ATTN_IMPLEMENTATIONS:
        allowed = ", ".join(sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS))
        raise ValueError(f"w1_attn_implementation must be one of: {allowed}")
    return normalized


def _observed_attn_implementation(encoder: Any, *, requested: str) -> str:
    config = getattr(encoder, "config", None)
    for attr in ("_attn_implementation", "attn_implementation"):
        observed = getattr(config, attr, None) if config is not None else None
        if observed:
            return str(observed).strip().lower()
    return requested


@dataclass(frozen=True)
class PreparedWindowMilBatch:
    """CPU-tokenized window batch for exact inference-only W1 scoring."""

    encoded: Mapping[str, Any]
    mapping: Any
    document_count: int
    window_count: int
    max_windows_per_document: int
    padded_token_count: int
    pooling_mode: str
    pooling_top_k: int
    pin_memory_requested: bool = False
    pin_memory_effective: bool = False
    telemetry: Mapping[str, Any] | None = None
    metadata: tuple[Mapping[str, Any], ...] = ()
    candidate_ids: tuple[str, ...] = ()

    def slice(self, start: int, end: int) -> "PreparedWindowMilBatch":
        if start < 0 or end < start or end > int(self.document_count):
            raise ValueError("prepared W1 batch slice bounds are invalid")
        if start == 0 and end == int(self.document_count):
            return self
        local_indexes = [
            index
            for index, value in enumerate(self.mapping.detach().cpu().tolist())
            if int(start) <= int(value) < int(end)
        ]
        if not local_indexes:
            raise ValueError("prepared W1 batch slice selected no windows")
        index_tensor = self.mapping.new_tensor(local_indexes, dtype=self.mapping.dtype)
        sliced_mapping = self.mapping.index_select(0, index_tensor) - int(start)
        sliced_encoded = {
            key: (
                value.index_select(0, index_tensor)
                if hasattr(value, "index_select") and int(getattr(value, "shape", (0,))[0]) == int(self.window_count)
                else value
            )
            for key, value in self.encoded.items()
        }
        telemetry = dict(self.telemetry or {})
        window_count = int(sliced_mapping.numel())
        first_tensor = next((value for value in sliced_encoded.values() if hasattr(value, "numel")), None)
        padded_token_count = int(first_tensor.numel()) if first_tensor is not None else window_count
        return replace(
            self,
            encoded=sliced_encoded,
            mapping=sliced_mapping,
            document_count=int(end) - int(start),
            window_count=window_count,
            padded_token_count=padded_token_count,
            telemetry=telemetry,
            metadata=tuple(self.metadata[int(start):int(end)]),
            candidate_ids=tuple(self.candidate_ids[int(start):int(end)]),
        )


class _TransformerWindowMilModel:
    """Light wrapper around an HF encoder with a binary risk head."""

    def __new__(cls, torch: Any, encoder: Any) -> Any:
        class TorchModule(torch.nn.Module):
            def __init__(self, wrapped_encoder: Any) -> None:
                super().__init__()
                self.encoder = wrapped_encoder
                hidden_size = int(getattr(wrapped_encoder.config, "hidden_size", 768))
                self.classifier = torch.nn.Linear(hidden_size, 1)

            def forward(self, **encoded: Any) -> Any:
                try:
                    outputs = self.encoder(**encoded)
                except TypeError:
                    encoded = {key: value for key, value in encoded.items() if key != "token_type_ids"}
                    outputs = self.encoder(**encoded)
                hidden = outputs.last_hidden_state
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    pooled = hidden[:, 0, :]
                else:
                    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                return self.classifier(pooled).squeeze(-1)

        return TorchModule(encoder)


def default_model_bundle_factory(
    *,
    model_name: str,
    torch_module: Any,
    attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
) -> tuple[Any, Any, str]:
    requested_attn_implementation = normalize_w1_attn_implementation(attn_implementation)
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("transformer window-MIL training requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        encoder = AutoModel.from_pretrained(model_name, attn_implementation=requested_attn_implementation)
    except Exception as exc:
        if requested_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
            raise RuntimeError(
                "W1 flash_attention_2 attention backend requested but unavailable; "
                "install compatible flash-attn/transformers dependencies or use sdpa"
            ) from exc
        raise
    observed_attn_implementation = _observed_attn_implementation(
        encoder,
        requested=requested_attn_implementation,
    )
    if (
        requested_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
        and observed_attn_implementation != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
    ):
        raise RuntimeError(
            "W1 flash_attention_2 attention backend requested but unavailable; "
            f"observed attention backend {observed_attn_implementation!r}"
        )
    config = getattr(encoder, "config", None)
    if config is not None and hasattr(config, "reference_compile"):
        setattr(config, "reference_compile", False)
    model = _TransformerWindowMilModel(torch_module, encoder)
    setattr(model, "w1_attn_implementation_requested", requested_attn_implementation)
    setattr(model, "w1_attn_implementation_observed", observed_attn_implementation)
    return tokenizer, model, model_name


def _load_model_bundle(
    *,
    model_name: str,
    model_bundle_factory: Any | None,
    torch: Any,
    attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
) -> tuple[Any, Any, str]:
    factory = model_bundle_factory or default_model_bundle_factory
    return factory(
        model_name=model_name,
        torch_module=torch,
        attn_implementation=normalize_w1_attn_implementation(attn_implementation),
    )


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("window-MIL training requires torch") from exc
    return torch


def _resolve_device(torch: Any, *, device: str, require_cuda: bool) -> Any:
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this window-MIL run but is not available")
    if require_cuda and not resolved.startswith("cuda"):
        raise RuntimeError("CUDA is required for this window-MIL run but device resolved to CPU")
    return torch.device(resolved)


def _make_grad_scaler(torch: Any, *, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    def factory() -> Any:
        if not enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)

    return factory


def _resolve_document_microbatch_size(value: int | None, total_documents: int) -> int:
    if int(total_documents) < 1:
        raise ValueError("training batch must contain at least one document")
    if value is None:
        return int(total_documents)
    size = int(value)
    if size < 1:
        raise ValueError("document_microbatch_size must be positive")
    return min(size, int(total_documents))


def _validate_train_batch_lengths(texts: Sequence[str], labels: Sequence[int], sample_weights: Sequence[float]) -> None:
    if len(texts) != len(labels) or len(texts) != len(sample_weights):
        raise ValueError("training inputs must have matching lengths")
    if not texts:
        raise ValueError("training batch must contain at least one document")


def _train_one_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scaler: Any,
    autocast: Any,
    texts: list[str],
    labels: list[int],
    sample_weights: list[float],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    document_microbatch_size: int | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _validate_train_batch_lengths(texts, labels, sample_weights)
    total_documents = len(texts)
    microbatch_size = _resolve_document_microbatch_size(document_microbatch_size, total_documents)
    loss_total = 0.0
    for start in range(0, total_documents, microbatch_size):
        end = min(total_documents, start + microbatch_size)
        with autocast():
            doc_logits = _document_logits_for_texts(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                texts=texts[start:end],
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
                checkpoint_window_chunks=True,
            )
            targets = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
            weights = torch.tensor(sample_weights[start:end], dtype=torch.float32, device=device)
            loss_values = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, targets, reduction="none")
            loss = (loss_values * weights).sum() / float(total_documents)
        loss_total += float(loss.detach().cpu().item())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss_total)


def _train_split_halo_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scaler: Any,
    autocast: Any,
    core_texts: list[str],
    halo_texts: list[str],
    labels: list[int],
    sample_weights: list[float],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_TOPK_MEAN,
    pooling_top_k: int = 3,
    halo_weight: float = 0.25,
    halo_logit_clamp: float = 2.0,
    document_microbatch_size: int | None = None,
) -> float:
    if len(core_texts) != len(halo_texts) or len(core_texts) != len(labels) or len(core_texts) != len(sample_weights):
        raise ValueError("split-halo training inputs must have matching lengths")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_documents = len(core_texts)
    microbatch_size = _resolve_document_microbatch_size(document_microbatch_size, total_documents)
    loss_total = 0.0
    for start in range(0, total_documents, microbatch_size):
        end = min(total_documents, start + microbatch_size)
        with autocast():
            doc_logits = _split_halo_document_logits(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                core_texts=core_texts[start:end],
                halo_texts=halo_texts[start:end],
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
                halo_weight=halo_weight,
                halo_logit_clamp=halo_logit_clamp,
                checkpoint_window_chunks=True,
            )
            targets = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
            weights = torch.tensor(sample_weights[start:end], dtype=torch.float32, device=device)
            loss_values = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, targets, reduction="none")
            loss = (loss_values * weights).sum() / float(total_documents)
        loss_total += float(loss.detach().cpu().item())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss_total)


def _score_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    telemetry: MutableMapping[str, Any] | None = None,
) -> list[float]:
    doc_logits = _document_logits_for_texts(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    postprocess_at = time.monotonic()
    scores = [float(value) for value in torch.sigmoid(doc_logits).detach().cpu().tolist()]
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
    return scores


def _score_split_halo_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    core_texts: list[str],
    halo_texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_TOPK_MEAN,
    pooling_top_k: int = 3,
    halo_weight: float = 0.25,
    halo_logit_clamp: float = 2.0,
    telemetry: MutableMapping[str, Any] | None = None,
) -> list[float]:
    doc_logits = _split_halo_document_logits(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        core_texts=core_texts,
        halo_texts=halo_texts,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        halo_weight=halo_weight,
        halo_logit_clamp=halo_logit_clamp,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    postprocess_at = time.monotonic()
    scores = [float(value) for value in torch.sigmoid(doc_logits).detach().cpu().tolist()]
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
    return scores


def _telemetry_add(telemetry: MutableMapping[str, Any] | None, key: str, value: float | int) -> None:
    if telemetry is None:
        return
    telemetry[key] = telemetry.get(key, 0) + value


def _telemetry_max(telemetry: MutableMapping[str, Any] | None, key: str, value: int) -> None:
    if telemetry is None:
        return
    telemetry[key] = max(int(telemetry.get(key, 0)), int(value))


def _telemetry_merge_prepared_payload(
    telemetry: MutableMapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> None:
    if telemetry is None or not isinstance(payload, Mapping):
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            telemetry[key] = bool(telemetry.get(key, False)) or value
        elif isinstance(value, int) and key.startswith("max_"):
            _telemetry_max(telemetry, key, value)
        elif isinstance(value, (int, float)):
            _telemetry_add(telemetry, key, value)
        elif value is not None:
            telemetry[key] = value


def _device_type_name(device: Any) -> str:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type)
    return str(device).split(":", 1)[0]


def _synchronize_cuda_for_telemetry(*, torch: Any, device: Any, telemetry: MutableMapping[str, Any] | None) -> None:
    if telemetry is None or not bool(telemetry.get("synchronize_cuda_timing")):
        return
    if _device_type_name(device) != "cuda":
        return
    cuda_module = getattr(torch, "cuda", None)
    synchronize = getattr(cuda_module, "synchronize", None)
    if not callable(synchronize):
        return
    try:
        synchronize(device)
    except TypeError:
        synchronize()
    telemetry["cuda_synchronized_timing"] = True
    _telemetry_add(telemetry, "cuda_synchronization_count", 1)


def _validated_cpu_document_mapping_counts(
    torch: Any,
    mapping: Any,
    *,
    document_count: int,
) -> tuple[Any, Any, int]:
    if int(document_count) < 1:
        raise ValueError("document_count must be positive")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    mapping = mapping.to(dtype=torch.long)
    if _device_type_name(getattr(mapping, "device", "cpu")) != "cpu":
        raise ValueError("document mapping validation requires CPU tokenizer output")
    if int(mapping.numel()) < int(document_count):
        raise ValueError("tokenizer returned fewer windows than documents")
    valid_mask = (mapping >= 0) & (mapping < int(document_count))
    if not bool(valid_mask.all()):
        raise ValueError("tokenizer returned window mapping outside document range")
    counts = torch.bincount(mapping, minlength=int(document_count))
    if bool((counts == 0).any()):
        raise ValueError("document has no tokenized windows")
    return mapping, counts, int(counts.max().item())


def _record_tokenized_window_shape_telemetry(
    torch: Any,
    *,
    encoded: Mapping[str, Any],
    mapping: Any,
    document_count: int,
    telemetry: MutableMapping[str, Any] | None,
) -> int:
    _mapping, _counts, max_windows_per_document = _validated_cpu_document_mapping_counts(
        torch,
        mapping,
        document_count=document_count,
    )
    if telemetry is None:
        return max_windows_per_document

    tensor_values = [value for value in encoded.values() if torch.is_tensor(value)]
    first_tensor = tensor_values[0] if tensor_values else None
    input_cell_count = int(first_tensor.numel()) if first_tensor is not None else 0
    max_sequence_length = 0
    attention_token_count = 0
    padding_cell_count = 0
    attention_mask = encoded.get("attention_mask")
    if torch.is_tensor(attention_mask):
        attention = attention_mask.to(dtype=torch.long)
        input_cell_count = int(attention.numel())
        attention_token_count = int(attention.sum().item())
        padding_cell_count = max(0, input_cell_count - attention_token_count)
        if int(attention.numel()) > 0:
            if getattr(attention, "ndim", 1) >= 2:
                max_sequence_length = int(attention.sum(dim=1).max().item())
            else:
                max_sequence_length = int(attention.sum().item())
    elif first_tensor is not None:
        attention_token_count = input_cell_count
        if getattr(first_tensor, "ndim", 1) >= 2:
            max_sequence_length = int(first_tensor.shape[-1])
        else:
            max_sequence_length = input_cell_count

    _telemetry_add(telemetry, "tokenized_input_cell_count", input_cell_count)
    _telemetry_add(telemetry, "tokenized_attention_token_count", attention_token_count)
    _telemetry_add(telemetry, "tokenized_padding_cell_count", padding_cell_count)
    _telemetry_max(telemetry, "max_tokenized_sequence_length", max_sequence_length)
    _telemetry_max(telemetry, "max_windows_per_document", max_windows_per_document)
    return max_windows_per_document


def _pin_tokenized_windows_for_cuda(
    torch: Any,
    *,
    encoded: Mapping[str, Any],
    mapping: Any,
    pin_memory: bool,
) -> tuple[dict[str, Any], Any, bool]:
    if not bool(pin_memory):
        return dict(encoded), mapping, False
    try:
        pinned_mapping = mapping.pin_memory() if hasattr(mapping, "pin_memory") else mapping
        pinned_encoded = {
            key: value.pin_memory() if torch.is_tensor(value) and hasattr(value, "pin_memory") else value
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
    except Exception:
        return dict(encoded), mapping, False
    if not pinned_encoded:
        return dict(encoded), mapping, False
    return pinned_encoded, pinned_mapping, True


def _prepare_document_logits_batch(
    *,
    torch: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    stride: int,
    pooling_mode: str,
    pooling_top_k: int,
    pin_memory: bool = False,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> PreparedWindowMilBatch:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if not texts:
        raise ValueError("prepared W1 batch requires at least one document")
    prepare_telemetry: dict[str, Any] = {}
    tokenize_at = time.monotonic()
    encoded, mapping = _tokenize_windows_cpu(
        torch=torch,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        stride=stride,
    )
    _telemetry_add(prepare_telemetry, "tokenizer_call_count", 1)
    _telemetry_add(prepare_telemetry, "tokenization_seconds", time.monotonic() - tokenize_at)
    window_count = int(mapping.numel())
    _telemetry_add(prepare_telemetry, "tokenized_window_count", window_count)
    _telemetry_max(prepare_telemetry, "max_tokenized_windows_per_document_batch", window_count)
    max_windows_per_document = _record_tokenized_window_shape_telemetry(
        torch,
        encoded=encoded,
        mapping=mapping,
        document_count=len(texts),
        telemetry=prepare_telemetry,
    )
    pin_at = time.monotonic()
    encoded, mapping, pin_memory_effective = _pin_tokenized_windows_for_cuda(
        torch,
        encoded=encoded,
        mapping=mapping,
        pin_memory=pin_memory,
    )
    if pin_memory:
        _telemetry_add(prepare_telemetry, "pin_memory_seconds", time.monotonic() - pin_at)
    prepare_telemetry["pin_memory_requested"] = bool(pin_memory)
    prepare_telemetry["pin_memory_effective"] = bool(pin_memory_effective)
    first_tensor = next((value for value in encoded.values() if torch.is_tensor(value)), None)
    padded_token_count = int(first_tensor.numel()) if first_tensor is not None else window_count
    prepared = PreparedWindowMilBatch(
        encoded=encoded,
        mapping=mapping,
        document_count=len(texts),
        window_count=window_count,
        max_windows_per_document=max_windows_per_document,
        padded_token_count=padded_token_count,
        pooling_mode=pooling_mode,
        pooling_top_k=int(pooling_top_k),
        pin_memory_requested=bool(pin_memory),
        pin_memory_effective=bool(pin_memory_effective),
        telemetry=dict(prepare_telemetry),
        metadata=tuple(dict(item) for item in (metadata or ())),
        candidate_ids=tuple(str(item.get("candidate_id_hash")) for item in (metadata or ()) if item.get("candidate_id_hash") is not None),
    )
    _telemetry_merge_prepared_payload(telemetry, prepared.telemetry)
    return prepared


def _document_logits_for_texts(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if int(max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    prepared = _prepare_document_logits_batch(
        torch=torch,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        stride=stride,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        pin_memory=False,
        telemetry=None,
    )
    return _document_logits_from_prepared_batch(
        torch=torch,
        model=model,
        prepared=prepared,
        device=device,
        max_windows_per_device_batch=max_windows_per_device_batch,
        checkpoint_window_chunks=checkpoint_window_chunks,
        telemetry=telemetry,
    )


def _document_logits_from_prepared_batch(
    *,
    torch: Any,
    model: Any,
    prepared: PreparedWindowMilBatch,
    device: Any,
    max_windows_per_device_batch: int,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    if int(max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    _validate_pooling_config(pooling_mode=prepared.pooling_mode, pooling_top_k=prepared.pooling_top_k)
    _telemetry_merge_prepared_payload(telemetry, prepared.telemetry)
    encoded = prepared.encoded
    mapping = prepared.mapping
    window_count = int(prepared.window_count)
    if window_count != int(mapping.numel()):
        raise ValueError("prepared W1 batch window count mismatch")
    if window_count > int(max_windows_per_device_batch):
        return _document_logits_from_encoded_window_chunks(
            torch=torch,
            model=model,
            encoded=encoded,
            mapping=mapping,
            document_count=int(prepared.document_count),
            device=device,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=prepared.pooling_mode,
            pooling_top_k=int(prepared.pooling_top_k),
            checkpoint_window_chunks=checkpoint_window_chunks,
            max_windows_per_document=int(prepared.max_windows_per_document),
            telemetry=telemetry,
            non_blocking=bool(prepared.pin_memory_effective),
        )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    transfer_at = time.monotonic()
    encoded, mapping = _move_tokenized_windows_to_device(
        torch=torch,
        encoded=encoded,
        mapping=mapping,
        device=device,
        non_blocking=bool(prepared.pin_memory_effective),
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "device_transfer_seconds", time.monotonic() - transfer_at)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    forward_at = time.monotonic()
    window_logits = _forward_window_logits(model, encoded)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "window_forward_call_count", 1)
    _telemetry_add(telemetry, "window_forward_window_count", window_count)
    _telemetry_add(telemetry, "window_forward_seconds", time.monotonic() - forward_at)
    _telemetry_max(telemetry, "max_windows_per_forward_call", window_count)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    pooling_at = time.monotonic()
    doc_logits = _pool_window_logits(
        torch,
        window_logits=window_logits,
        mapping=mapping,
        document_count=int(prepared.document_count),
        pooling_mode=prepared.pooling_mode,
        pooling_top_k=int(prepared.pooling_top_k),
        max_windows_per_document=int(prepared.max_windows_per_document),
        mapping_validated=True,
        use_ragged_topk_pooling=False,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "pooling_seconds", time.monotonic() - pooling_at)
    return doc_logits


def _document_logits_from_encoded_window_chunks(
    *,
    torch: Any,
    model: Any,
    encoded: Mapping[str, Any],
    mapping: Any,
    document_count: int,
    device: Any,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    checkpoint_window_chunks: bool = False,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
    non_blocking: bool = False,
) -> Any:
    window_count = int(mapping.numel())
    if window_count < 1:
        raise ValueError("tokenizer returned no windows")
    chunk_logits = []
    chunk_mappings = []
    for start in range(0, window_count, int(max_windows_per_device_batch)):
        end = min(window_count, start + int(max_windows_per_device_batch))
        encoded_chunk = {
            key: value[start:end] if torch.is_tensor(value) and int(value.shape[0]) == window_count else value
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        transfer_at = time.monotonic()
        moved_chunk, moved_mapping = _move_tokenized_windows_to_device(
            torch=torch,
            encoded=encoded_chunk,
            mapping=mapping[start:end],
            device=device,
            non_blocking=bool(non_blocking),
        )
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        _telemetry_add(telemetry, "device_transfer_seconds", time.monotonic() - transfer_at)
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        forward_at = time.monotonic()
        logits = _forward_window_logits_for_training_chunk(
            torch=torch,
            model=model,
            encoded=moved_chunk,
            checkpoint_window_chunks=checkpoint_window_chunks,
        )
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        chunk_logits.append(logits)
        _telemetry_add(telemetry, "window_forward_call_count", 1)
        _telemetry_add(telemetry, "window_forward_window_count", end - start)
        _telemetry_add(telemetry, "window_forward_seconds", time.monotonic() - forward_at)
        _telemetry_max(telemetry, "max_windows_per_forward_call", end - start)
        chunk_mappings.append(moved_mapping)
    if not chunk_logits:
        raise ValueError("tokenizer returned no windows")
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    pooling_at = time.monotonic()
    doc_logits = _pool_window_logits(
        torch,
        window_logits=torch.cat(chunk_logits, dim=0),
        mapping=torch.cat(chunk_mappings, dim=0),
        document_count=document_count,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        max_windows_per_document=max_windows_per_document,
        mapping_validated=True,
        use_ragged_topk_pooling=False,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "pooling_seconds", time.monotonic() - pooling_at)
    return doc_logits


def _split_halo_document_logits(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    core_texts: list[str],
    halo_texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    halo_weight: float,
    halo_logit_clamp: float,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    if len(core_texts) != len(halo_texts):
        raise ValueError("split-halo scoring inputs must have matching lengths")
    if not math.isfinite(float(halo_weight)) or float(halo_weight) < 0:
        raise ValueError("halo_weight must be a finite nonnegative value")
    if not math.isfinite(float(halo_logit_clamp)) or float(halo_logit_clamp) <= 0:
        raise ValueError("halo_logit_clamp must be a finite positive value")
    core_logits = _document_logits_for_texts(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        texts=core_texts,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        checkpoint_window_chunks=checkpoint_window_chunks,
        telemetry=telemetry,
    )
    halo_logits = torch.zeros(len(core_texts), dtype=core_logits.dtype, device=device)
    halo_inputs = [(index, text) for index, text in enumerate(halo_texts) if str(text).strip()]
    if halo_inputs:
        indexes = [index for index, _text in halo_inputs]
        scored_halos = _document_logits_for_texts(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            texts=[text for _index, text in halo_inputs],
            device=device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            checkpoint_window_chunks=checkpoint_window_chunks,
            telemetry=telemetry,
        )
        for position, index in enumerate(indexes):
            halo_logits[index] = scored_halos[position]
    return core_logits + float(halo_weight) * torch.clamp(
        halo_logits,
        min=-float(halo_logit_clamp),
        max=float(halo_logit_clamp),
    )


def _forward_window_logits_for_training_chunk(
    *,
    torch: Any,
    model: Any,
    encoded: Mapping[str, Any],
    checkpoint_window_chunks: bool,
) -> Any:
    if not checkpoint_window_chunks or not torch.is_grad_enabled():
        return _forward_window_logits(model, encoded)
    try:
        from torch.utils.checkpoint import checkpoint as torch_checkpoint
    except ImportError as exc:  # pragma: no cover - torch builds normally include checkpointing.
        raise RuntimeError("torch.utils.checkpoint is required for checkpointed window-MIL training") from exc
    keys = tuple(sorted(encoded))
    tensors = tuple(encoded[key] for key in keys)
    if not tensors:
        raise ValueError("tokenizer returned no tensor inputs")

    def checkpointed_forward(*values: Any) -> Any:
        return _forward_window_logits(model, dict(zip(keys, values, strict=True)))

    return torch_checkpoint(
        checkpointed_forward,
        *tensors,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def _tokenize_windows_cpu(
    *,
    torch: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    stride: int,
) -> tuple[dict[str, Any], Any]:
    encoded = tokenizer(
        texts,
        max_length=int(max_length),
        stride=int(stride),
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    if "overflow_to_sample_mapping" not in encoded:
        raise ValueError("tokenizer did not return overflow_to_sample_mapping")
    mapping = encoded.pop("overflow_to_sample_mapping")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    encoded_tensors = {
        key: value
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }
    if not encoded_tensors:
        raise ValueError("tokenizer returned no tensor inputs")
    if int(mapping.numel()) < len(texts):
        raise ValueError("tokenizer returned fewer windows than documents")
    return encoded_tensors, mapping


def _move_tokenized_windows_to_device(
    *,
    torch: Any,
    encoded: Mapping[str, Any],
    mapping: Any,
    device: Any,
    non_blocking: bool = False,
) -> tuple[dict[str, Any], Any]:
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    moved_mapping = mapping.to(device, non_blocking=bool(non_blocking))
    moved_encoded = {
        key: value.to(device, non_blocking=bool(non_blocking))
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }
    if not moved_encoded:
        raise ValueError("tokenizer returned no tensor inputs")
    return moved_encoded, moved_mapping


def _forward_window_logits(model: Any, encoded: Mapping[str, Any]) -> Any:
    outputs = model(**encoded)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if getattr(logits, "ndim", 1) == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if getattr(logits, "ndim", 1) != 1:
        raise ValueError("window-MIL model must return one logit per window")
    return logits


def _pool_window_logits(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    document_count: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    max_windows_per_document: int | None = None,
    mapping_validated: bool = False,
    use_ragged_topk_pooling: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if int(document_count) < 1:
        raise ValueError("document_count must be positive")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long, device=window_logits.device)
    else:
        if getattr(mapping, "device", None) != window_logits.device or getattr(mapping, "dtype", None) != torch.long:
            mapping = mapping.to(device=window_logits.device, dtype=torch.long)
    if int(window_logits.numel()) != int(mapping.numel()):
        raise ValueError("window logits and mapping must have matching lengths")
    if int(mapping.numel()) < int(document_count):
        raise ValueError("tokenizer returned fewer windows than documents")
    if not bool(mapping_validated):
        valid_mask = (mapping >= 0) & (mapping < int(document_count))
        if not bool(valid_mask.all()):
            mapping = mapping[valid_mask]
            window_logits = window_logits[valid_mask]
    counts = torch.bincount(mapping, minlength=int(document_count))
    if not bool(mapping_validated):
        if bool((counts == 0).any()):
            raise ValueError("document has no tokenized windows")
    if pooling_mode == WINDOW_POOLING_MAX and hasattr(torch.Tensor, "scatter_reduce_"):
        doc_logits = torch.full(
            (int(document_count),),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        return doc_logits.scatter_reduce_(0, mapping, window_logits, reduce="amax", include_self=True)
    if pooling_mode == WINDOW_POOLING_TOPK_MEAN:
        if bool(use_ragged_topk_pooling) and hasattr(torch.Tensor, "scatter_reduce_"):
            return _pool_window_logits_topk_mean_ragged(
                torch,
                window_logits=window_logits,
                mapping=mapping,
                counts=counts,
                document_count=int(document_count),
                pooling_top_k=int(pooling_top_k),
                max_windows_per_document=max_windows_per_document,
                telemetry=telemetry,
            )
        return _pool_window_logits_topk_mean_vectorized(
            torch,
            window_logits=window_logits,
            mapping=mapping,
            counts=counts,
            document_count=int(document_count),
            pooling_top_k=int(pooling_top_k),
            max_windows_per_document=max_windows_per_document,
            telemetry=telemetry,
        )
    return _pool_window_logits_reference(
        torch,
        window_logits=window_logits,
        mapping=mapping,
        document_count=document_count,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
    )


def _pool_window_logits_reference(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    document_count: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    doc_logits = []
    for document_index in range(int(document_count)):
        mask = mapping == document_index
        if not bool(mask.any()):
            raise ValueError("document has no tokenized windows")
        doc_logits.append(
            _pool_logits_for_one_document(
                torch,
                window_logits[mask],
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
            )
        )
    return torch.stack(doc_logits)


def _pool_window_logits_topk_mean_ragged(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = (
        int(counts.max().detach().cpu().item())
        if max_windows_per_document is None
        else int(max_windows_per_document)
    )
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    k = min(int(pooling_top_k), max_windows)
    window_count = int(window_logits.numel())
    _telemetry_add(telemetry, "pooling_ragged_topk_call_count", 1)
    _telemetry_add(telemetry, "pooling_ragged_topk_iteration_count", k)
    _telemetry_add(telemetry, "pooling_ragged_topk_input_window_count", window_count)
    _telemetry_add(telemetry, "pooling_ragged_topk_document_count", int(document_count))
    _telemetry_add(telemetry, "pooling_dense_padded_cell_count_avoided", int(document_count) * max_windows)
    if telemetry is not None:
        telemetry["pooling_backend"] = "ragged_topk"

    active = torch.ones((window_count,), dtype=torch.bool, device=window_logits.device)
    selected_sum = torch.zeros((int(document_count),), dtype=window_logits.dtype, device=window_logits.device)
    positions = torch.arange(window_count, dtype=torch.long, device=window_logits.device)
    sentinel_position = int(window_count)
    is_nan = torch.isnan(window_logits)
    zeros = torch.zeros_like(window_logits)

    for _iteration in range(k):
        active_nan = active & is_nan
        doc_has_nan = torch.zeros((int(document_count),), dtype=torch.long, device=window_logits.device)
        doc_has_nan.scatter_reduce_(
            0,
            mapping,
            active_nan.to(dtype=torch.long),
            reduce="amax",
            include_self=True,
        )
        use_nan_for_doc = doc_has_nan > 0
        finite_candidates = torch.where(
            active & ~is_nan,
            window_logits,
            torch.full_like(window_logits, -float("inf")),
        )
        max_per_doc = torch.full(
            (int(document_count),),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        max_per_doc.scatter_reduce_(0, mapping, finite_candidates, reduce="amax", include_self=True)
        winner_mask = (active_nan & use_nan_for_doc[mapping]) | (
            active
            & ~is_nan
            & ~use_nan_for_doc[mapping]
            & (window_logits == max_per_doc[mapping])
        )
        candidate_positions = torch.where(
            winner_mask,
            positions,
            torch.full_like(positions, sentinel_position),
        )
        winner_positions = torch.full(
            (int(document_count),),
            sentinel_position,
            dtype=torch.long,
            device=window_logits.device,
        )
        winner_positions.scatter_reduce_(
            0,
            mapping,
            candidate_positions,
            reduce="amin",
            include_self=True,
        )
        selected = winner_mask & (positions == winner_positions[mapping])
        selected_sum.scatter_add_(0, mapping, torch.where(selected, window_logits, zeros))
        active = active & ~selected

    selected_counts = torch.clamp(counts, max=k).to(dtype=window_logits.dtype)
    return selected_sum / selected_counts


def _pool_window_logits_topk_mean_vectorized(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = (
        int(counts.max().detach().cpu().item())
        if max_windows_per_document is None
        else int(max_windows_per_document)
    )
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    padded_cell_count = int(document_count) * max_windows
    if telemetry is not None:
        telemetry["pooling_backend"] = "vectorized_topk_mean"
    _telemetry_add(telemetry, "pooling_vectorized_topk_call_count", 1)
    _telemetry_add(telemetry, "pooling_vectorized_topk_input_window_count", int(window_logits.numel()))
    _telemetry_add(telemetry, "pooling_vectorized_topk_document_count", int(document_count))
    _telemetry_add(telemetry, "pooling_vectorized_padded_cell_count", padded_cell_count)
    if padded_cell_count > 1_000_000:
        return _pool_window_logits_topk_mean_padded_chunks(
            torch,
            window_logits=window_logits,
            mapping=mapping,
            counts=counts,
            document_count=document_count,
            pooling_top_k=pooling_top_k,
            max_windows_per_document=max_windows,
            telemetry=telemetry,
        )
    order = torch.argsort(mapping, stable=True)
    sorted_mapping = mapping[order]
    sorted_logits = window_logits[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(int(sorted_logits.numel()), device=window_logits.device) - starts[sorted_mapping]
    padded = torch.full(
        (int(document_count), max_windows),
        -float("inf"),
        dtype=window_logits.dtype,
        device=window_logits.device,
    )
    padded[sorted_mapping, positions] = sorted_logits
    k = min(int(pooling_top_k), max_windows)
    top_values = torch.topk(padded, k=k, dim=1).values
    selected_counts = torch.clamp(counts, max=k).to(dtype=window_logits.dtype)
    selected_mask = torch.arange(k, device=window_logits.device).unsqueeze(0) < selected_counts.unsqueeze(1)
    selected = torch.where(selected_mask, top_values, torch.zeros_like(top_values))
    return selected.sum(dim=1) / selected_counts


def _pool_window_logits_topk_mean_padded_chunks(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int,
    max_padded_cell_count: int = 1_000_000,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = int(max_windows_per_document)
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    order = torch.argsort(mapping, stable=True)
    sorted_mapping = mapping[order]
    sorted_logits = window_logits[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(int(sorted_logits.numel()), device=window_logits.device) - starts[sorted_mapping]
    k = min(int(pooling_top_k), max_windows)
    docs_per_chunk = max(1, int(max_padded_cell_count) // max_windows)
    if telemetry is not None:
        telemetry["pooling_backend"] = "vectorized_topk_mean_padded_chunks"
    _telemetry_add(telemetry, "pooling_vectorized_padded_chunk_call_count", 1)
    output = torch.empty((int(document_count),), dtype=window_logits.dtype, device=window_logits.device)
    for doc_start in range(0, int(document_count), docs_per_chunk):
        doc_end = min(int(document_count), doc_start + docs_per_chunk)
        row_mask = (sorted_mapping >= doc_start) & (sorted_mapping < doc_end)
        local_mapping = sorted_mapping[row_mask] - doc_start
        local_positions = positions[row_mask]
        local_logits = sorted_logits[row_mask]
        padded = torch.full(
            (doc_end - doc_start, max_windows),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        padded[local_mapping, local_positions] = local_logits
        top_values = torch.topk(padded, k=k, dim=1).values
        selected_counts = torch.clamp(counts[doc_start:doc_end], max=k).to(dtype=window_logits.dtype)
        selected_mask = torch.arange(k, device=window_logits.device).unsqueeze(0) < selected_counts.unsqueeze(1)
        selected = torch.where(selected_mask, top_values, torch.zeros_like(top_values))
        output[doc_start:doc_end] = selected.sum(dim=1) / selected_counts
    return output


def _pool_logits_for_one_document(torch: Any, logits: Any, *, pooling_mode: str, pooling_top_k: int) -> Any:
    if int(logits.numel()) < 1:
        raise ValueError("document has no tokenized windows")
    if pooling_mode == WINDOW_POOLING_MAX:
        return logits.max()
    if pooling_mode == WINDOW_POOLING_TOPK_MEAN:
        k = min(int(pooling_top_k), int(logits.numel()))
        return torch.topk(logits, k=k).values.mean()
    raise ValueError(f"unsupported pooling mode: {pooling_mode}")


def _validate_pooling_config(*, pooling_mode: str, pooling_top_k: int) -> None:
    if pooling_mode not in SUPPORTED_WINDOW_POOLING_MODES:
        raise ValueError(f"unsupported pooling mode: {pooling_mode}")
    if int(pooling_top_k) < 1:
        raise ValueError("pooling_top_k must be positive")
    if pooling_mode == WINDOW_POOLING_MAX and int(pooling_top_k) != 1:
        raise ValueError("max pooling requires pooling_top_k=1")


def _device_observations(torch: Any, device: Any) -> dict[str, Any]:
    payload = {
        "device_type": str(device.type),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        payload.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": str(torch.cuda.get_device_name(index)),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return payload


def _model_numeric_hash(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


PreparedWindowMilBatch.__module__ = "twc.classifiers.window_mil"
# Preserve the established pickle lookup path while window_mil re-exports the same class.
