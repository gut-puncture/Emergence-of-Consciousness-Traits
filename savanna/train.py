"""Fine-tune the base model (or a random-init copy) on a text world; evaluate on a fixed held-out set.

Loss as in post-training: forecast (cross-entropy of the true answer over the question's option tokens, mean over
questions) + observation (full-vocabulary next-token loss on printed world lines, mean over their tokens).
Commands and question text carry no loss; the answer is never written into the text (the token after
"Answer:" is the newline), so no answer leaks into later questions.
AdamW (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, lr 6e-5 after 100 warm-up updates, held; FP32.

usage: train.py OUT --updates N --batch 8 [--random-init] [--eval-at 200,500,...] [--smoke]
"""
import argparse, json, math, os, pickle, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from fast import load, hidden
import importlib
# the world module is chosen per run (--world NAME, read before anything else is imported)
WORLD = sys.argv[sys.argv.index("--world") + 1] if "--world" in sys.argv else "world"
W = importlib.import_module(WORLD)
TAG = WORLD + ("_" + os.environ["STRESS"] if os.environ.get("STRESS") else "")   # names the held-out cache
simulate, render_episode, tag_question, LEVELS = W.simulate, W.render_episode, W.tag_question, W.LEVELS

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = Tokenizer.from_str(open(os.path.join(HERE, "tokenizer.json")).read())
OPT_IDS = [TOK.encode(" " + w).ids[0] for w in LEVELS]
NL = TOK.encode("\n").ids
assert len(NL) == 1
OBS_KINDS = getattr(W, "OBS_KINDS", ("morning", "hour", "power"))
NOPT = 3   # most options a question has; a question with fewer has its spare slots masked


def options(ln):
    """The question's answer words (a world may give each question its own set; otherwise LEVELS)."""
    return tuple(W.options(ln)) if hasattr(W, "options") else tuple(LEVELS)


def option_ids(ln):
    ids = [TOK.encode(" " + w).ids[0] for w in options(ln)]
    assert len(set(ids)) == len(ids), options(ln)   # first tokens must tell the options apart
    return ids + [-1] * (NOPT - len(ids))


def truth_index(ln):
    w = W.to_word(ln.q["module"], ln.q["truth"])
    return w if isinstance(w, int) else options(ln).index(w)   # older worlds give the level index itself


def encode(lines):
    """ids, obs mask, [(position of the ':' after Answer, question index, option token ids)]."""
    ids, obs, ans = [], [], []
    texts = render_episode(lines)
    for i, (ln, t) in enumerate(zip(lines, texts)):
        e = TOK.encode(t).ids + NL
        if ln.kind == "question":
            assert TOK.decode(e[-3:-1]) == " Answer:", TOK.decode(e[-3:])
            ans.append((len(ids) + len(e) - 2, i, option_ids(ln)))
        ids += e
        obs += [ln.kind in OBS_KINDS] * len(e)
    return ids, obs, ans


QMAX = 40   # most questions an episode can ask (2 an hour x 16 hours = 32)


def batch_tensors(eps, dev):
    L = -(-max(len(e[0]) for e in eps) // 128) * 128   # padded to a multiple of 128: few distinct shapes
    ids = torch.zeros(len(eps), L, dtype=torch.long)
    tgt = torch.full((len(eps), L), -100, dtype=torch.long)
    apos, atru, aopt = [], [], []
    for b, (x, obs, ans, truths) in enumerate(eps):
        ids[b, :len(x)] = torch.tensor(x)
        nxt = torch.tensor(x[1:] + [NL[0]])
        m = torch.tensor(obs)
        # loss on predicting each printed-line token from the token before it
        mprev = torch.zeros(len(x), dtype=torch.bool); mprev[:-1] = m[1:]
        tgt[b, :len(x)] = torch.where(mprev, nxt, torch.tensor(-100))
        for (p, _, o), tw in zip(ans, truths):
            apos.append(b * L + p); atru.append(tw); aopt.append(o)
    # answer slots padded to a fixed count (position 0, ignored target): every tensor shape depends only on L,
    # because MPS keeps a compiled graph per shape and varying shapes leak memory
    n = len(apos)
    pad = QMAX * len(eps) - n
    apos += [0] * pad; atru += [-100] * pad; aopt += [[0] * NOPT] * pad
    return (ids.to(dev), tgt.to(dev), torch.tensor(apos, device=dev), torch.tensor(atru, device=dev), n,
            torch.tensor(aopt, device=dev))


def option_logprobs(m, h_sel, aopt):
    """Log-probabilities over each question's own options (spare slots, id -1, get none)."""
    logits = m.head(h_sel).float().gather(1, aopt.clamp(min=0))
    return torch.log_softmax(logits.masked_fill(aopt < 0, float("-inf")), -1)


def empty_cache():
    if torch.cuda.is_available(): return   # the CUDA caching allocator handles varying shapes
    torch.mps.empty_cache()


def mem_bytes():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else torch.mps.driver_allocated_memory()


def NOBS(eps):
    """Loss-bearing printed-line tokens of a batch (each predicted from the token before it)."""
    return sum(sum(e[1][1:]) for e in eps)


def train_episode(rng):
    lines = simulate(rng)
    x, obs, ans = encode(lines)
    truths = [truth_index(lines[i]) for _, i, _ in ans]
    return x, obs, ans, truths


def build_heldout(path, n_eps, seed=10**6):
    if os.path.exists(path):
        eps, rows = pickle.load(open(path, "rb"))
        if len(eps) == n_eps:
            return eps, rows
    rng = random.Random(seed)
    eps, rows = [], []
    for e in range(n_eps):
        lines = simulate(rng)
        x, obs, ans = encode(lines)
        for p, i, _ in ans:
            r = tag_question(lines, i); r["ep"] = e; r["options"] = options(lines[i]); rows.append(r)
        eps.append((x, obs, ans, [truth_index(lines[i]) for _, i, _ in ans]))
    pickle.dump((eps, rows), open(path, "wb"))
    return eps, rows


@torch.no_grad()
def evaluate(m, eps, dev, bs=8):
    m.eval()
    out = []
    for i in range(0, len(eps), bs):
        ids, _, apos, _, n, aopt = batch_tensors(eps[i:i + bs], dev)
        h = hidden(m, ids).flatten(0, 1).index_select(0, apos)
        out.append(option_logprobs(m, h, aopt).cpu()[:n])
    m.train()
    empty_cache()
    return torch.cat(out).numpy()   # log-probs over each question's options (spare slots -inf), held-out order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--eval-at", default=""); ap.add_argument("--heldout", type=int, default=800)
    ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--world", default="world"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # FP32, as agreed
    log = open(os.path.join(a.out, "log.txt"), "a")
    def say(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    t0 = time.time()
    hold, rows = build_heldout(os.path.join(HERE, f"heldout_{TAG}.pkl"), a.heldout)
    say(f"held-out: {len(hold)} episodes, {len(rows)} questions ({time.time()-t0:.0f}s)")
    m = load(os.path.join(HERE, "base_weights.pt"), dev, random_init=a.random_init, seed=a.seed)
    m.train()
    decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in m.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    ck = os.path.join(a.out, "ckpt.pt")
    start = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu")
        m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); start = s["update"]
        say("resumed at", start)
    evals = sorted(int(x) for x in a.eval_at.split(",") if x)
    if start == 0 and 0 in evals:
        np.save(os.path.join(a.out, "eval_0.npy"), evaluate(m, hold, dev)); say("eval 0 done")
    t_last, acc = time.time(), []
    for u in range(start + 1, a.updates + 1):
        lr = a.lr * min(1.0, u / 100)
        for g in opt.param_groups: g["lr"] = lr
        rng = random.Random(a.seed * 10**7 + u)
        eps = [train_episode(rng) for _ in range(a.batch)]
        # two microbatches of four, gradients summed: the same loss as one batch of eight, in half the memory
        nq = sum(len(e[2]) for e in eps)
        opt.zero_grad(set_to_none=True)
        fsum = osum = 0.0
        for c in range(0, a.batch, 4):
            ids, tgt, apos, atru, _, aopt = batch_tensors(eps[c:c + 4], dev)
            h = hidden(m, ids).flatten(0, 1)
            t = tgt.flatten(); sel = None
            o_s = F.cross_entropy(m.head(h).float(), t, ignore_index=-100, reduction="sum")
            fl = option_logprobs(m, h.index_select(0, apos), aopt)
            f_s = F.nll_loss(fl, atru, ignore_index=-100, reduction="sum")
            (f_s / nq + o_s / NOBS(eps)).backward()
            fsum += f_s.item(); osum += o_s.item()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        del h, ids, tgt, apos, atru, aopt, t, sel, o_s, fl, f_s
        empty_cache()   # batch shapes vary; without this the MPS cache grows until it runs out
        fc_loss, obs_loss = torch.tensor(fsum / nq), torch.tensor(osum / NOBS(eps))
        acc.append((fc_loss.item(), obs_loss.item(), nq))
        if u % 20 == 0 or u == a.updates:
            f, o, q = map(np.mean, zip(*acc)); acc = []
            say(f"update {u} forecast {f:.4f} obs {o:.4f} q/upd {q:.0f} gnorm {gn.item():.2f} lr {lr:.1e} "
                f"mem {mem_bytes()/2**30:.1f}G {(time.time()-t_last)/20:.2f}s/upd elapsed {(time.time()-t0)/60:.1f}m")
            t_last = time.time()
        if u in evals:
            np.save(os.path.join(a.out, f"eval_{u}.npy"), evaluate(m, hold, dev)); say(f"eval {u} done")
            torch.save(m.state_dict(), os.path.join(a.out, f"weights_{u}.pt"))   # kept for the trait tests over training
        if u % 250 == 0 or u == a.updates:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), update=u), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    say("finished", a.updates, f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
