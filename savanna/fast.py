"""The base decoder's forward on a padded [batch, length] tensor: the same computation as
twc.pretraining.training.model.DecoderLM.hidden (RMSNorm, RoPE theta 1e4, GQA attention, SwiGLU, tied head), with
RoPE tables built once per call and one batched causal attention per layer. Right padding never reaches real
tokens under the causal mask."""
import os, sys, torch
import torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
# the base model's code: training/ beside this file (a GPU bundle) or at the repository root
sys.path.insert(0, os.path.join(_HERE, "training") if os.path.isdir(os.path.join(_HERE, "training")) else os.path.join(_HERE, "..", "training"))
from twc.pretraining.training.model import DecoderConfig, DecoderLM, PackedCausalBatch

def rms(x, w):
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5))

def hidden(m, ids):  # ids [B, L]
    c = m.config
    B, L = ids.shape
    H, KV = c.num_attention_heads, c.num_key_value_heads
    D = c.hidden_size // H
    inv = 1.0 / (10_000 ** (torch.arange(0, D, 2, device=ids.device, dtype=torch.float32) / D))
    f = torch.outer(torch.arange(L, device=ids.device, dtype=torch.float32), inv)
    cos, sin = f.cos()[None, None], f.sin()[None, None]          # [1,1,L,D/2]
    def rot(v):  # v [B, h, L, D]
        e, o = v[..., 0::2], v[..., 1::2]
        return torch.stack((e * cos - o * sin, e * sin + o * cos), dim=-1).flatten(-2)
    x = m.embedding(ids)
    for blk in m.blocks:
        v = rms(x, blk.attn_norm.weight)
        q, k, vv = blk.qkv(v).split((H * D, KV * D, KV * D), dim=-1)
        q = rot(q.view(B, L, H, D).transpose(1, 2))
        k = rot(k.view(B, L, KV, D).transpose(1, 2)).repeat_interleave(H // KV, dim=1)
        vv = vv.view(B, L, KV, D).transpose(1, 2).repeat_interleave(H // KV, dim=1)
        a = F.scaled_dot_product_attention(q, k, vv, is_causal=True)
        x = x + blk.out(a.transpose(1, 2).reshape(B, L, H * D))
        v = rms(x, blk.mlp_norm.weight)
        x = x + blk.down(F.silu(blk.gate(v)) * blk.up(v))
    return rms(x, m.norm.weight)

class Cache:
    """Keys and values of the tokens fed so far, per layer, for incremental decoding of a batch of separate texts.
    Each text keeps its own positions; padded slots are stored but never attended to. Without a capacity the cache
    grows by concatenation (a copy per step); with one (cap, in slots), buffers of that size are allocated on the first
    step, keys and values stored already repeated over the query heads, and each step writes its tokens in place at
    the n slots filled so far and attends to the filled part (a view: no copy)."""
    def __init__(self, cap=None):
        self.k, self.v, self.pos, self.valid = None, None, None, None
        self.cap, self.n = cap, 0


def hidden_step(m, ids, pos, valid, cache):
    """Hidden states of new tokens ids [B, T] at RoPE positions pos [B, T] (valid [B, T] marks real tokens),
    attending to everything in the cache and causally to each other; the cache is extended in place."""
    c = m.config
    B, T = ids.shape
    H, KV = c.num_attention_heads, c.num_key_value_heads
    D = c.hidden_size // H
    inv = 1.0 / (10_000 ** (torch.arange(0, D, 2, device=ids.device, dtype=torch.float32) / D))
    f = pos.float()[..., None] * inv                                 # [B, T, D/2]
    cos, sin = f.cos()[:, None], f.sin()[:, None]                    # [B, 1, T, D/2]
    def rot(v):
        e, o = v[..., 0::2], v[..., 1::2]
        return torch.stack((e * cos - o * sin, e * sin + o * cos), dim=-1).flatten(-2)
    if cache.cap is None:
        kpos = pos if cache.pos is None else torch.cat([cache.pos, pos], 1)
        kval = valid if cache.valid is None else torch.cat([cache.valid, valid], 1)
    else:
        n, E = cache.n, cache.n + T
        assert E <= cache.cap, (E, cache.cap)
        if cache.k is None:
            dt = m.embedding.weight.dtype
            cache.k = [torch.empty(B, H, cache.cap, D, dtype=dt, device=ids.device) for _ in m.blocks]
            cache.v = [torch.empty(B, H, cache.cap, D, dtype=dt, device=ids.device) for _ in m.blocks]
            cache.pos = torch.empty(B, cache.cap, dtype=pos.dtype, device=ids.device)
            cache.valid = torch.empty(B, cache.cap, dtype=torch.bool, device=ids.device)
        cache.pos[:, n:E] = pos; cache.valid[:, n:E] = valid
        kpos, kval = cache.pos[:, :E], cache.valid[:, :E]
    mask = kval[:, None, None, :] & (kpos[:, None, None, :] <= pos[:, None, :, None])   # [B, 1, T, K]
    new_k, new_v = [], []
    x = m.embedding(ids)
    for li, blk in enumerate(m.blocks):
        v = rms(x, blk.attn_norm.weight)
        q, k, vv = blk.qkv(v).split((H * D, KV * D, KV * D), dim=-1)
        q = rot(q.view(B, T, H, D).transpose(1, 2))
        k = rot(k.view(B, T, KV, D).transpose(1, 2))
        vv = vv.view(B, T, KV, D).transpose(1, 2)
        if cache.cap is None:
            if cache.k is not None:
                k = torch.cat([cache.k[li], k], 2); vv = torch.cat([cache.v[li], vv], 2)
            new_k.append(k); new_v.append(vv)
            k, vv = k.repeat_interleave(H // KV, dim=1), vv.repeat_interleave(H // KV, dim=1)
        else:   # written in place, per query head; the filled slots are attended to as a view
            cache.k[li][:, :, n:E] = k.repeat_interleave(H // KV, dim=1); cache.v[li][:, :, n:E] = vv.repeat_interleave(H // KV, dim=1)
            k, vv = cache.k[li][:, :, :E], cache.v[li][:, :, :E]
        a = F.scaled_dot_product_attention(q, k, vv, attn_mask=mask)
        x = x + blk.out(a.transpose(1, 2).reshape(B, T, H * D))
        v = rms(x, blk.mlp_norm.weight)
        x = x + blk.down(F.silu(blk.gate(v)) * blk.up(v))
    if cache.cap is None:
        cache.k, cache.v, cache.pos, cache.valid = new_k, new_v, kpos, kval
    else:
        cache.n = E
    return rms(x, m.norm.weight)


def load_state(base_path, weights_path, dev):
    """A model with the base config and the weights of a training checkpoint (a plain state dict)."""
    m = load(base_path, "cpu")
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        m.load_state_dict(sd["model"] if "model" in sd else sd)   # a weights_{u}.pt or a resume ckpt.pt
    return m.to(dev)


def load(path, dev, random_init=False, seed=0):
    d = torch.load(path, map_location="cpu")
    torch.manual_seed(seed)
    m = DecoderLM(DecoderConfig(**d["config"]))
    if not random_init:
        m.load_state_dict(d["weights"])
    return m.to(dev)

if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "mps"
    m = load("base_weights.pt", dev); m.eval()
    B, L = 3, 700
    ids = torch.randint(0, 32000, (B, L))
    lens = [700, 650, 500]
    with torch.no_grad():
        ref = m.hidden(PackedCausalBatch.create(torch.cat([ids[i, :n] for i, n in enumerate(lens)]),
                                                torch.cat([ids[i, :n] for i, n in enumerate(lens)]),
                                                [0, 700, 1350, 1850]).to(dev))
        mine = hidden(m, ids.to(dev))
        mine = torch.cat([mine[i, :n] for i, n in enumerate(lens)])
        print("max abs diff", (ref - mine).abs().max().item(), "ref scale", ref.abs().max().item())
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=6e-5, betas=(0.9, 0.95), weight_decay=0.1)
    ids = torch.randint(0, 32000, (12, 750)).to(dev)
    for it in range(3):
        torch.mps.synchronize(); t = time.time()
        h = hidden(m, ids); loss = F.cross_entropy(m.head(h[:, ::3]).flatten(0, 1), ids[:, ::3].flatten())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        torch.mps.synchronize(); print(f"update {time.time()-t:.2f}s", flush=True)
