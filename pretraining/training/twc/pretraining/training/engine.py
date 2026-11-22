"""Single-device engine with exact token-weighted accumulation semantics."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn

from twc.pretraining.training.model import LMOutput, PackedCausalBatch


@dataclass(frozen=True, slots=True)
class EngineConfig:
    learning_rate: float
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    precision: str
    attention_backend: str
    compile_mode: str
    world_size: int
    grad_clip_norm: float
    warmup_tokens: int
    stable_until_tokens: int
    target_tokens: int
    attention_num_splits: int | str = 1
    optimizer_backend: str = "reference"
    allow_tf32: bool = False

    @classmethod
    def from_contract(cls, contract: Any) -> "EngineConfig":
        return cls(
            learning_rate=contract.learning_rate,
            adam_betas=contract.adam_betas,
            adam_eps=contract.adam_eps,
            weight_decay=contract.weight_decay,
            precision=contract.precision,
            attention_backend=contract.attention_backend,
            compile_mode=contract.compile_mode,
            world_size=1,
            grad_clip_norm=contract.grad_clip_norm,
            warmup_tokens=contract.warmup_loss_tokens,
            stable_until_tokens=contract.warmup_loss_tokens + contract.stable_loss_tokens,
            target_tokens=contract.target_loss_tokens,
            attention_num_splits=contract.attention_num_splits,
            optimizer_backend=contract.optimizer_backend,
            allow_tf32=contract.allow_tf32,
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    loss_tokens: int
    learning_rate: float
    grad_norm: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    loss_tokens: int
    batch_count: int
    component_losses: tuple[float, ...]
    component_loss_tokens: tuple[int, ...]


def wsd_token_multiplier(
    consumed_before_step: int,
    step_loss_tokens: int,
    *,
    warmup_tokens: int,
    stable_until_tokens: int,
    target_tokens: int,
) -> float:
    """WSD learning-rate multiplier sampled at the optimizer step midpoint."""

    values = (consumed_before_step, step_loss_tokens, warmup_tokens, stable_until_tokens, target_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("token schedule values must be integers")
    if consumed_before_step < 0 or step_loss_tokens <= 0:
        raise ValueError("token positions must be nonnegative with positive step work")
    if not 0 < warmup_tokens <= stable_until_tokens < target_tokens:
        raise ValueError("token schedule boundaries are incoherent")
    if consumed_before_step + step_loss_tokens > target_tokens:
        raise ValueError("optimizer step exceeds the target token budget")
    midpoint = consumed_before_step + step_loss_tokens / 2
    if midpoint < warmup_tokens:
        return midpoint / warmup_tokens
    if midpoint < stable_until_tokens:
        return 1.0
    progress = (midpoint - stable_until_tokens) / (target_tokens - stable_until_tokens)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


Microbatch = tuple[torch.Tensor, torch.Tensor] | PackedCausalBatch


class TrainingEngine:
    def __init__(self, model: nn.Module, config: EngineConfig, *, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.config = config
        if config.precision == "bf16" and self.device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 requires a CPU or CUDA device")
        if config.attention_backend == "flash" and self.device.type != "cuda":
            raise ValueError("flash attention requires CUDA")
        if config.world_size != 1:
            raise ValueError("R7 supports exactly one training device")
        if config.optimizer_backend == "fused" and self.device.type != "cuda":
            raise ValueError("fused AdamW requires CUDA")
        if config.optimizer_backend not in {"reference", "fused"}:
            raise ValueError("unsupported optimizer backend")
        if config.allow_tf32 and self.device.type != "cuda":
            raise ValueError("TF32 requires CUDA")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.fp32_precision = "tf32" if config.allow_tf32 else "ieee"
            torch.backends.cudnn.conv.fp32_precision = "tf32" if config.allow_tf32 else "ieee"
        prepared_model = model.to(self.device)
        configure_attention = getattr(prepared_model, "configure_attention", None)
        if configure_attention is not None:
            configure_attention(config.attention_backend, config.attention_num_splits)
        if config.compile_mode != "disabled":
            prepared_model = torch.compile(prepared_model, mode=config.compile_mode)
        self.model = prepared_model
        decay, no_decay = _optimizer_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_eps,
            fused=config.optimizer_backend == "fused",
        )
        self.step = 0
        self.consumed_loss_tokens = 0

    @property
    def canonical_model(self) -> nn.Module:
        """Return the stable unwrapped module used by checkpoint state dictionaries."""

        return getattr(self.model, "_orig_mod", self.model)

    def snapshot(self) -> dict[str, object]:
        return {
            "model": {
                name: tensor.detach().to(device="cpu", copy=True)
                for name, tensor in self.canonical_model.state_dict().items()
            },
            "optimizer": _clone_to_cpu(self.optimizer.state_dict()),
            "step": self.step,
            "consumed_loss_tokens": self.consumed_loss_tokens,
        }

    def restore(self, state: dict[str, object]) -> None:
        required = {"model", "optimizer", "step", "consumed_loss_tokens"}
        if set(state) != required:
            raise ValueError("engine checkpoint state has missing or unknown fields")
        step = state["step"]
        consumed = state["consumed_loss_tokens"]
        if type(step) is not int or step < 0 or type(consumed) is not int or consumed < 0:
            raise ValueError("engine checkpoint cursor is invalid")
        self.canonical_model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = step
        self.consumed_loss_tokens = consumed

    def train_step(self, microbatches: Iterable[Microbatch], *, expected_loss_tokens: int | None = None) -> StepResult:
        if expected_loss_tokens is None:
            batches: Iterable[Microbatch] = list(microbatches)
            total_tokens = sum(_microbatch_loss_tokens(batch) for batch in batches)
        else:
            if type(expected_loss_tokens) is not int or expected_loss_tokens <= 0:
                raise ValueError("expected_loss_tokens must be a positive integer")
            batches = microbatches
            total_tokens = expected_loss_tokens
        if total_tokens <= 0:
            raise ValueError("training step requires at least one loss token")
        multiplier = self._schedule_multiplier(total_tokens)
        lr = self.config.learning_rate * multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        try:
            observed_tokens = 0
            observed_batches = 0
            for batch in batches:
                expected_tokens = _microbatch_loss_tokens(batch)
                observed_tokens += expected_tokens
                observed_batches += 1
                with self._forward_context():
                    output = self._forward(batch)
                if output.loss_token_count != expected_tokens:
                    raise ValueError("model loss-token count disagrees with the microbatch")
                if not torch.isfinite(output.loss_sum):
                    raise FloatingPointError("non-finite training loss")
                (output.loss_sum / total_tokens).backward()
                total_loss_sum += float(output.loss_sum.detach())
            if observed_batches == 0 or observed_tokens != total_tokens:
                raise ValueError("streamed microbatches do not match the expected loss-token total")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.step += 1
        self.consumed_loss_tokens += total_tokens
        return StepResult(
            loss=total_loss_sum / total_tokens,
            loss_tokens=total_tokens,
            learning_rate=lr,
            grad_norm=float(grad_norm),
        )

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        tokens = 0
        count = 0
        component_losses: list[float] = []
        component_tokens: list[int] = []
        try:
            for batch in batches:
                expected_tokens = _microbatch_loss_tokens(batch)
                with self._forward_context():
                    output = self._forward(batch)
                if output.loss_token_count != expected_tokens:
                    raise ValueError("model loss-token count disagrees with the evaluation batch")
                if not torch.isfinite(output.loss_sum):
                    raise FloatingPointError("non-finite evaluation loss")
                loss_sum += float(output.loss_sum)
                tokens += output.loss_token_count
                count += 1
                component_losses.append(float(output.loss_sum) / output.loss_token_count)
                component_tokens.append(output.loss_token_count)
        finally:
            self.model.train(was_training)
        if count == 0 or tokens == 0:
            raise ValueError("evaluation requires at least one nonempty batch")
        return EvaluationResult(
            loss=loss_sum / tokens,
            loss_tokens=tokens,
            batch_count=count,
            component_losses=tuple(component_losses),
            component_loss_tokens=tuple(component_tokens),
        )

    def _schedule_multiplier(self, step_tokens: int) -> float:
        return wsd_token_multiplier(
            self.consumed_loss_tokens,
            step_tokens,
            warmup_tokens=self.config.warmup_tokens,
            stable_until_tokens=self.config.stable_until_tokens,
            target_tokens=self.config.target_tokens,
        )

    def _forward(self, batch: Microbatch) -> LMOutput:
        if isinstance(batch, PackedCausalBatch):
            output = self.model(batch.to(self.device))
        else:
            inputs, labels = batch
            output = self.model(inputs.to(self.device), labels.to(self.device))
        if not isinstance(output, LMOutput):
            raise TypeError("training models must return LMOutput")
        return output

    @contextmanager
    def _forward_context(self):
        with ExitStack() as stack:
            stack.enter_context(
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.config.precision == "bf16",
                )
            )
            if self.config.attention_backend == "math":
                stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH))
            elif self.config.attention_backend == "flash":
                stack.enter_context(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION))
            yield


def _microbatch_loss_tokens(batch: Microbatch) -> int:
    if isinstance(batch, PackedCausalBatch):
        return batch.loss_token_count
    _, labels = batch
    if labels.numel() == 0:
        raise ValueError("microbatch labels cannot be empty")
    return int((labels != -100).sum().item())


def _optimizer_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not seen:
        raise ValueError("training model has no parameters")
    return decay, no_decay


def _clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value
