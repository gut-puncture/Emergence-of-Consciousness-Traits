"""The supported decoder architecture and its flat, sequence-isolated batch seam."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    vocab_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.context_length,
            self.hidden_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.intermediate_size,
        ) <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size must be at least hidden_size")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


R7_HISTORICAL_CONTROL_MODEL_CONFIG = DecoderConfig(
    vocab_size=32_768,
    context_length=4_096,
    hidden_size=512,
    num_layers=26,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1_664,
    dropout=0.0,
)

# Compatibility for historical control fixtures and preserved artifacts. Active
# production configuration is compiled from configs/model_100m.yaml.
R7_BASE_MODEL_CONFIG = R7_HISTORICAL_CONTROL_MODEL_CONFIG


@dataclass(frozen=True, slots=True)
class PackedCausalBatch:
    """A flat token batch whose cumulative offsets are hard attention boundaries."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    loss_token_count: int

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        cu_seqlens: tuple[int, ...] | list[int] | torch.Tensor,
    ) -> "PackedCausalBatch":
        if input_ids.ndim != 1 or target_ids.ndim != 1 or input_ids.shape != target_ids.shape:
            raise ValueError("packed input_ids and target_ids must be equal-length flat tensors")
        if input_ids.numel() == 0:
            raise ValueError("packed batch must contain at least one loss token")
        if input_ids.dtype == torch.bool or target_ids.dtype == torch.bool:
            raise ValueError("token tensors cannot use boolean dtype")
        offsets = torch.as_tensor(cu_seqlens, dtype=torch.int64, device="cpu")
        if offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError("cu_seqlens must contain at least one sequence")
        values = tuple(int(item) for item in offsets.tolist())
        if values[0] != 0:
            raise ValueError("cu_seqlens must start at zero")
        if values[-1] != input_ids.numel():
            raise ValueError("cu_seqlens final offset must equal flat token count")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("cu_seqlens must be strictly increasing")
        if values[-1] > torch.iinfo(torch.int32).max:
            raise ValueError("cu_seqlens exceed int32 capacity")
        lengths = [right - left for left, right in zip(values, values[1:])]
        return cls(
            input_ids=input_ids,
            target_ids=target_ids,
            cu_seqlens=offsets.to(dtype=torch.int32),
            max_seqlen=max(lengths),
            loss_token_count=input_ids.numel(),
        )

    def to(self, device: torch.device | str) -> "PackedCausalBatch":
        return PackedCausalBatch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            loss_token_count=self.loss_token_count,
        )


@dataclass(frozen=True, slots=True)
class LMOutput:
    loss_sum: torch.Tensor
    loss_token_count: int
    logits: torch.Tensor | None

    @property
    def loss(self) -> torch.Tensor:
        return self.loss_sum / self.loss_token_count

class DecoderLM(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.num_layers)])
        self.norm = _RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.attention_backend = "reference"
        self.attention_num_splits: int | None = 1
        self._reset_parameters()

    def configure_attention(self, backend: str, num_splits: int | str = 1) -> None:
        if backend not in {"reference", "native_varlen", "math", "sdpa"}:
            raise ValueError("unsupported attention backend")
        if not (num_splits == "auto" or type(num_splits) is int and num_splits > 0):
            raise ValueError("attention num_splits must be a positive integer or auto")
        self.attention_backend = "reference" if backend in {"reference", "math", "sdpa"} else backend
        self.attention_num_splits = None if num_splits == "auto" else num_splits

    def _reset_parameters(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, _RMSNorm):
                nn.init.ones_(module.weight)
        for block in self.blocks:
            nn.init.normal_(block.out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.down.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        batch: PackedCausalBatch,
        *,
        return_logits: bool = False,
    ) -> LMOutput:
        if not isinstance(batch, PackedCausalBatch):
            raise TypeError("DecoderLM accepts only PackedCausalBatch")
        if batch.max_seqlen > self.config.context_length:
            raise ValueError("packed sequence exceeds decoder context")
        if batch.input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input token IDs must use an integer tensor dtype")
        hidden = self.embedding(batch.input_ids)
        for block in self.blocks:
            hidden = block(
                hidden,
                batch.cu_seqlens,
                batch.max_seqlen,
                attention_backend=self.attention_backend,
                attention_num_splits=self.attention_num_splits,
            )
        hidden = self.norm(hidden)
        loss_sum, logits = self._project_loss(hidden, batch.target_ids, return_logits=bool(return_logits))
        return LMOutput(loss_sum=loss_sum, loss_token_count=batch.loss_token_count, logits=logits)

    def _project_loss(
        self, hidden: torch.Tensor, targets: torch.Tensor, *, return_logits: bool, chunk_size: int = 4_096
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        chunks: list[torch.Tensor] = []
        for start in range(0, hidden.shape[0], chunk_size):
            logits = self.head(hidden[start : start + chunk_size])
            loss_sum = loss_sum + F.cross_entropy(
                logits.float(), targets[start : start + chunk_size], reduction="sum"
            )
            if return_logits:
                chunks.append(logits)
        return loss_sum, torch.cat(chunks) if return_logits else None


def initialize_decoder(config: DecoderConfig, *, seed: int) -> DecoderLM:
    """Seed every available device before allocating fresh model parameters."""

    if seed < 0:
        raise ValueError("model initialization seed must be nonnegative")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return DecoderLM(config)


def estimate_parameter_count(config: DecoderConfig) -> int:
    """Return the exact unique parameter count for the tied-embedding decoder."""

    head_dim = config.hidden_size // config.num_attention_heads
    kv_size = config.num_key_value_heads * head_dim
    embedding = config.vocab_size * config.hidden_size
    attention = config.hidden_size * (config.hidden_size + 2 * kv_size) + config.hidden_size**2
    mlp = 3 * config.hidden_size * config.intermediate_size
    block_norms = 2 * config.hidden_size
    return embedding + config.num_layers * (attention + mlp + block_norms) + config.hidden_size


class _RMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        return (self.weight.float() * normalized).to(value.dtype)


class _Block(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = _RMSNorm(config.hidden_size)
        head_dim = config.hidden_size // config.num_attention_heads
        kv_size = config.num_key_value_heads * head_dim
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size + 2 * kv_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp_norm = _RMSNorm(config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        attention_backend: str,
        attention_num_splits: int | None,
    ) -> torch.Tensor:
        value = self.attn_norm(hidden)
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        dim = self.config.hidden_size // heads
        q_size = heads * dim
        kv_size = kv_heads * dim
        projected = self.qkv(value)
        q, k, v = projected.split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(-1, heads, dim)
        k = k.view(-1, kv_heads, dim)
        v = v.view(-1, kv_heads, dim)
        q, k = _rope_flat(q, k, cu_seqlens)
        if attention_backend == "native_varlen":
            if q.device.type != "cuda":
                raise RuntimeError("native variable-length attention requires CUDA")
            from torch.nn.attention.varlen import varlen_attn

            attended = varlen_attn(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                window_size=(-1, 0),
                enable_gqa=True,
                num_splits=attention_num_splits,
            )
        else:
            attended = _reference_varlen_attention(
                q, k, v, cu_seqlens, dropout=self.config.dropout if self.training else 0.0
            )
        hidden = hidden + self.out(attended.reshape(-1, self.config.hidden_size))
        value = self.mlp_norm(hidden)
        return hidden + self.down(F.silu(self.gate(value)) * self.up(value))


def _spans(cu_seqlens: torch.Tensor) -> Iterator[tuple[int, int]]:
    offsets = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    return iter(zip(offsets, offsets[1:]))


def _reference_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    dropout: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    repeat = q.shape[1] // k.shape[1]
    for start, stop in _spans(cu_seqlens):
        q_part = q[start:stop].transpose(0, 1).unsqueeze(0)
        k_part = k[start:stop].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        v_part = v[start:stop].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
        result = F.scaled_dot_product_attention(q_part, k_part, v_part, is_causal=True, dropout_p=dropout)
        outputs.append(result.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _rope_flat(
    q: torch.Tensor, k: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.cat(
        [torch.arange(stop - start, device=q.device, dtype=torch.float32) for start, stop in _spans(cu_seqlens)]
    )
    dim = q.shape[-1]
    inv = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    frequency = torch.outer(positions, inv)
    cos = frequency.cos().to(q.dtype)[:, None]
    sin = frequency.sin().to(q.dtype)[:, None]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
