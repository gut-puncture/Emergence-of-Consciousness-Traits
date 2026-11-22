"""The supported decoder architecture and its flat, sequence-isolated batch seam."""

from __future__ import annotations

from dataclasses import dataclass
import functools
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
            loss_token_count=int((target_ids != -100).sum().item()),
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
        if self.loss_token_count <= 0:
            raise ValueError("loss requires a supervised target; use logits_at for inference")
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
        self.sdpa_kernel = "math"
        self._reset_parameters()

    def configure_attention(self, backend: str, num_splits: int | str = 1, sdpa_kernel: str = "math") -> None:
        """``sdpa_kernel`` picks the scaled-dot-product kernel the reference backend's attention calls run on where
        no gradient is taken (SDPA_KERNELS; training always uses math): the same float32 attention up to the order
        of summation, so the prefix cache and every test against the math kernel hold under either."""
        if backend not in {"reference", "native_varlen", "math", "sdpa"}:
            raise ValueError("unsupported attention backend")
        if not (num_splits == "auto" or type(num_splits) is int and num_splits > 0):
            raise ValueError("attention num_splits must be a positive integer or auto")
        if sdpa_kernel not in SDPA_KERNELS:
            raise ValueError(f"sdpa_kernel must be one of {sorted(SDPA_KERNELS)}")
        self.attention_backend = "reference" if backend in {"reference", "math", "sdpa"} else backend
        self.attention_num_splits = None if num_splits == "auto" else num_splits
        self.sdpa_kernel = sdpa_kernel
        for block in self.blocks:
            block.sdpa_kernel = sdpa_kernel

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
        hidden = self.hidden(batch)
        loss_sum, logits = self._project_loss(hidden, batch.target_ids, return_logits=bool(return_logits))
        return LMOutput(loss_sum=loss_sum, loss_token_count=batch.loss_token_count, logits=logits)

    def hidden(self, batch: PackedCausalBatch) -> torch.Tensor:
        if not isinstance(batch, PackedCausalBatch) or batch.max_seqlen > self.config.context_length:
            raise ValueError("invalid packed decoder input")
        hidden = self.embedding(batch.input_ids)
        for block in self.blocks:
            hidden = block(hidden, batch.cu_seqlens, batch.max_seqlen, attention_backend=self.attention_backend,
                           attention_num_splits=self.attention_num_splits)
        return self.norm(hidden)


    def logits_at(self, batch: PackedCausalBatch, positions: torch.Tensor) -> torch.Tensor:
        positions = torch.as_tensor(positions, dtype=torch.long, device=batch.input_ids.device)
        if positions.ndim != 1 or positions.numel() == 0 or bool((positions < 0).any()) or bool((positions >= batch.input_ids.numel()).any()):
            raise ValueError("projection positions outside packed batch")
        return self.head(self.hidden(batch).index_select(0, positions))

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
        self.sdpa_kernel = "math"    # set for every block by DecoderLM.configure_attention

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
                q, k, v, cu_seqlens, dropout=self.config.dropout if self.training else 0.0, kernel=self.sdpa_kernel
            )
        hidden = hidden + self.out(attended.reshape(-1, self.config.hidden_size))
        value = self.mlp_norm(hidden)
        return hidden + self.down(F.silu(self.gate(value)) * self.up(value))

    def prefix_suffix(self, hidden, positions, spans, previous, *, attention_cells):
        """Inference-only suffix computation, gated by prefix_cache.prefix_logits."""
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        dim = self.config.hidden_size // heads
        q, k, v = self.qkv(self.attn_norm(hidden)).split(
            (heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
        q, k = _rope_positions(q.view(-1, heads, dim), k.view(-1, kv_heads, dim), positions)
        v = v.view(-1, kv_heads, dim)
        keys, values = [], []
        for (start, stop), old in zip(spans, previous, strict=True):
            keys.append(k[start:stop] if old is None else torch.cat((old[0], k[start:stop])))
            values.append(v[start:stop] if old is None else torch.cat((old[1], v[start:stop])))
        attended, cells = suffix_attention_batched(q, keys, values, spans, max_cells=attention_cells,
                                                   kernel=self.sdpa_kernel)
        hidden = hidden + self.out(attended.reshape(-1, self.config.hidden_size))
        value = self.mlp_norm(hidden)
        hidden = hidden + self.down(F.silu(self.gate(value)) * self.up(value))
        return hidden, list(zip(keys, values, strict=True)), cells


def _spans(cu_seqlens: torch.Tensor) -> Iterator[tuple[int, int]]:
    offsets = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    return iter(zip(offsets, offsets[1:]))


# The scaled-dot-product kernels the reference backend's attention calls may run on, by name. "math" is the
# exact kernel alone; "efficient" prefers the memory-efficient kernel (float32 on CUDA, with masks and causal
# attention), then the flash kernel where it can run (float32 on the CPU), then math where neither can. Each is
# the same float32 attention up to the order of summation. The choice applies only where no gradient is taken
# (rollouts, answers, evaluation, the prefix cache's suffix attention): the memory-efficient kernel's backward is
# not deterministic on CUDA, which would break a bit-exact resume, so training's forward and backward always run
# on the math kernel.
SDPA_KERNELS = {"math": ("MATH",), "efficient": ("EFFICIENT_ATTENTION", "FLASH_ATTENTION", "MATH")}


def _sdpa(kernel: str):
    """The context under which the attention calls of ``kernel`` run: ``kernel``'s list under no_grad, math otherwise."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    names = SDPA_KERNELS[kernel] if not torch.is_grad_enabled() else SDPA_KERNELS["math"]
    return sdpa_kernel([getattr(SDPBackend, name) for name in names])


def _reference_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    dropout: float,
    kernel: str = "math",
) -> torch.Tensor:
    spans = list(_spans(cu_seqlens))
    # Nonzero dropout keeps the original call order and RNG consumption.
    order = list(range(len(spans))) if dropout else sorted(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])
    groups: list[list[int]] = []
    for index in order:
        length = spans[index][1] - spans[index][0]
        group = groups[-1] if groups else []
        cells = sum((spans[i][1] - spans[i][0]) ** 2 for i in group) + length ** 2
        if dropout or not group or len(group) == 8 or (len(group) + 1) * length ** 2 > 1.25 * cells:
            groups.append([index])
        else:
            group.append(index)
    outputs: list[torch.Tensor | None] = [None] * len(spans)
    repeat = q.shape[1] // k.shape[1]
    for group in groups:
        def padded(value: torch.Tensor) -> torch.Tensor:
            parts = [value[spans[i][0]:spans[i][1]] for i in group]
            return torch.nn.utils.rnn.pad_sequence(parts, batch_first=True).transpose(1, 2)

        q_part = padded(q)
        k_part = padded(k).repeat_interleave(repeat, dim=1)
        v_part = padded(v).repeat_interleave(repeat, dim=1)
        # Padding is strictly to the right of every real query. The unchanged
        # causal mask excludes those keys; padded query outputs are discarded.
        with _sdpa(kernel):
            result = F.scaled_dot_product_attention(q_part, k_part, v_part, is_causal=True, dropout_p=dropout)
        for row, index in enumerate(group):
            start, stop = spans[index]
            outputs[index] = result[row, :, :stop - start].transpose(0, 1)
    return torch.cat(outputs, dim=0)


def _rope_flat(
    q: torch.Tensor, k: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.cat(
        [torch.arange(stop - start, device=q.device, dtype=torch.float32) for start, stop in _spans(cu_seqlens)]
    )
    return _rope_positions(q, k, positions)


def _rope_positions(q, k, positions):
    dim = q.shape[-1]
    inv = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    frequency = torch.outer(positions, inv)
    cos = frequency.cos().to(q.dtype)[:, None]
    sin = frequency.sin().to(q.dtype)[:, None]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def suffix_attention_batched(q, keys, values, spans, *, max_cells, kernel: str = "math"):
    """``suffix_attention`` for every sequence in one attention call per layer (on ``kernel``, SDPA_KERNELS):
    queries and each sequence's keys and values padded to the longest, the offset causal mask and the padding
    masked out, and the query heads grouped onto their shared key-value head instead of repeating the keys. The
    loop's float32 attention up to the order of float summation (about 1e-6), without its several small operations
    per sequence; the loop is used where the padded batch would exceed the attention cell budget. The mask and the
    real rows' index depend only on the lengths, so they are built once for all the layers of a step
    (`_suffix_layout`)."""
    from torch.nn.utils.rnn import pad_sequence
    heads, dim = q.shape[1], q.shape[2]
    kv_heads = keys[0].shape[1]
    repeat = heads // kv_heads
    lengths = [stop - start for start, stop in spans]
    counts = [len(key) for key in keys]
    batch, longest, widest = len(spans), max(lengths), max(counts)
    cells = batch * heads * longest * widest
    if cells > max_cells:
        return suffix_attention(q, keys, values, spans, max_cells=max_cells, kernel=kernel)
    mask, real = _suffix_layout(tuple(lengths), tuple(counts), repeat, q.device)
    query = pad_sequence([q[start:stop] for start, stop in spans], batch_first=True)      # [batch, longest, heads, dim]
    query = query.view(batch, longest, kv_heads, repeat, dim).permute(0, 2, 3, 1, 4).reshape(
        batch, kv_heads, repeat * longest, dim)                                          # head h = kv * repeat + r
    key = pad_sequence(keys, batch_first=True).transpose(1, 2)                            # [batch, kv, widest, dim]
    value = pad_sequence(values, batch_first=True).transpose(1, 2)
    with _sdpa(kernel):
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
    out = out.view(batch, kv_heads, repeat, longest, dim).permute(0, 3, 1, 2, 4).reshape(batch * longest, heads, dim)
    return out.index_select(0, real), cells


@functools.lru_cache(maxsize=8)
def _suffix_layout(lengths: tuple, counts: tuple, repeat: int, device) -> tuple:
    """The padded batch's attention mask and its real query rows, in span order, for these suffix lengths and key
    counts: a suffix query at row i of a sequence with `count` keys and `length` new tokens sees keys up to its own
    absolute place, i + count - length; padded keys never; a padded query row sees key 0 only (and is dropped)."""
    batch, longest, widest = len(lengths), max(lengths), max(counts)
    row = torch.arange(longest)[None, :, None]
    column = torch.arange(widest)[None, None, :]
    count = torch.tensor(counts)[:, None, None]
    length = torch.tensor(lengths)[:, None, None]
    allowed = ((column <= row + count - length) & (column < count)) | ((row >= length) & (column == 0))
    mask = allowed[:, None, None].expand(batch, 1, repeat, longest, widest).reshape(batch, 1, repeat * longest, widest)
    real = torch.tensor([b * longest + i for b, n in enumerate(lengths) for i in range(n)])
    return mask.to(device), real.to(device)


def suffix_attention(q, keys, values, spans, *, max_cells, kernel: str = "math"):
    """Exact attention (on ``kernel``, SDPA_KERNELS) with offset causal masks, bounded query workspace."""
    heads = q.shape[1]
    outputs, peak = [], 0
    # Keep each sequence isolated. Q/K/V projections above still share one flat
    # forward; small per-sequence query chunks bound attention matrix workspace.
    for (start, stop), key, value in zip(spans, keys, values, strict=True):
        key_count = len(key)
        rows = max_cells // (heads * key_count)
        if rows < 1:
            raise ValueError('attention cell budget cannot fit one query against this prefix')
        repeat = heads // key.shape[1]
        k = key.transpose(0, 1).unsqueeze(0).repeat_interleave(repeat, dim=1)
        v = value.transpose(0, 1).unsqueeze(0).repeat_interleave(repeat, dim=1)
        length = stop - start
        chunks = []
        for left in range(0, length, rows):
            right = min(length, left + rows)
            query = q[start + left:start + right].transpose(0, 1).unsqueeze(0)
            # is_causal=True alone would incorrectly align suffix query zero to
            # key zero. Explicit absolute local positions include the prior KV.
            allowed = torch.arange(key_count, device=q.device)[None, :] <= (
                torch.arange(left, right, device=q.device) + key_count - length)[:, None]
            with _sdpa(kernel):
                result = F.scaled_dot_product_attention(query, k, v, attn_mask=allowed, dropout_p=0.0)
            chunks.append(result[0].transpose(0, 1))
            peak = max(peak, heads * (right - left) * key_count)
        outputs.append(torch.cat(chunks))
    return torch.cat(outputs), peak
