"""Stages 2 and 3: the model acts (decisions.md).

Stage 2 continues a stage-1 model on a mix of days: two of eight scripted (the routine acts; the model also learns
to write the routine's commands) and six closed-loop (the model writes every command and the world answers; one
command in five is drawn evenly from the legal ones). The world-line and forecast losses are stage 1's, on every
day; the command loss is on the scripted days' commands only, never on the model's own.

Stage 3 continues a stage-2 model on closed-loop days only and adds a reward for its commands. At every command
point h the day follows a command a, and an independent second draw a' is taken from the same policy. Each is
scored on a branch built the same way: the day's own commands before h, a (or a') at h, then the routine (the same
world draws in both, so the tails differ only through the place and level each command leads to). The real day
stays the training text. Both branches are scored on a fixed set of questions that no command chooses: the four
stores, asked at hours h+1 to h+3, each about the next hour. R = the model's own log-probability of the truth,
summed over that set; the advantage of a is R(a) - R(a') - beta * (log pi(a) - log pi_stage2(a)), and the loss
adds -advantage * log pi(a) (REINFORCE).

usage: train_act.py OUT --init WEIGHTS --stage 2 --updates 1000 --eval-at 250,500,1000 --seed S
       train_act.py OUT --init STAGE2_WEIGHTS --stage 3 --ref STAGE2_WEIGHTS --updates 1500 --eval-at ... --seed S
"""
import argparse, json, os, random, sys, time
sys.argv += [] if "--world" in sys.argv else ["--world", "world_i1"]   # train.py reads the world module from argv
import numpy as np
import torch
import torch.nn.functional as F
import train as T                      # stage 1's encoding, batching, losses and held-out evaluation
from fast import load_state, hidden, hidden_step, Cache
from act import Commands, run_days, legal_logprobs

W = T.W
STORES = ("water", "grass", "well", "herd")
FIXED = {t: [(s, t + 1) for s in STORES] for t in range(W.FIRST + 1, W.LAST)}   # the reward's question set


def encode_cmds(lines):
    """Stage 1's encoding plus, per command line, (position of its first token after 'Command:', place, tokens)."""
    x, obs, ans = T.encode(lines)
    cpos, n = [], 0
    for ln, t in zip(lines, W.render_episode(lines)):
        e = T.TOK.encode(t).ids + T.NL
        if ln.kind == "command":
            cpos.append((n + len(CMDS.p0), ln.place, e[len(CMDS.p0):]))
        n += len(e)
    truths = [T.truth_index(lines[i]) for _, i, _ in ans]
    return x, obs, ans, truths, cpos


def command_logprobs(m, h, eps, L):
    """log pi(command) of every command line of every day: a list per day."""
    out = []
    for b, e in enumerate(eps):
        out.append([legal_logprobs(m.head(h[b * L + p - 1: b * L + p - 1 + len(toks)]).float(), place, toks, CMDS).sum()
                    for p, place, toks in e[4]])
    return out


def cut_after(lines, hour):
    """The lines up to and including hour `hour`'s questions."""
    for i, ln in enumerate(lines):
        if ln.kind == "hour" and ln.hour == hour + 1:
            return lines[:i]
    return lines


def branch(sc, cmds, h, a, window):
    """A compared branch of the day, with the fixed questions: the day's own commands before hour h, a at h, then
    the routine (its chance draws from the script, at the place and level this branch reaches), cut after hour
    min(h + window, LAST - 1)'s questions. Every command is legal where it is given."""
    g = W.day(sc, FIXED)
    try:
        lines, place, t, lev = next(g)
        while True:
            c = cmds[t] if t < h else a if t == h else W.pick(sc["act_u"][t], W.policy(place, t, lev))
            lines, place, t, lev = g.send(c)
    except StopIteration as e:
        return cut_after(e.value, min(h + window, W.LAST - 1))


@torch.no_grad()
def question_logp(m, days, dev):
    """Per day: [(asked hour, log-probability of the truth)] for every question, by the model."""
    bs = 16 if dev == "cuda" else 4
    out = [None] * len(days)
    enc = [T.encode(L) for L in days]
    order = sorted(range(len(days)), key=lambda i: len(enc[i][0]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        Lmax = -(-max(len(enc[i][0]) for i in idx) // 256) * 256   # few distinct shapes (MPS keeps a graph per shape)
        ids = torch.zeros(bs, Lmax, dtype=torch.long)
        pos, opt, tru, who = [], [], [], []
        for b, i in enumerate(idx):
            x, _, ans = enc[i]
            ids[b, :len(x)] = torch.tensor(x)
            for p, qi, o in ans:
                pos.append(b * Lmax + p); opt.append(o); tru.append(T.truth_index(days[i][qi])); who.append((i, days[i][qi].q["asked"]))
        n = len(pos)
        if n == 0:   # a group of days without questions (a cut branch can end before its window)
            continue
        pad = -(-n // 64) * 64 - n
        pos += [0] * pad; opt += [opt[0]] * pad; tru += [0] * pad
        h = hidden(m, ids.to(dev)).flatten(0, 1).index_select(0, torch.tensor(pos, device=dev))
        lp = T.option_logprobs(m, h, torch.tensor(opt, device=dev))
        lp = lp.gather(1, torch.tensor(tru, device=dev)[:, None])[:, 0].float().cpu().numpy()[:n]
        for (i, t), v in zip(who, lp):
            out[i] = out[i] or []
            out[i].append((t, float(v)))
        T.empty_cache()
    return [o or [] for o in out]


@torch.no_grad()
def tail_logp(m, base, tails, dev):
    """Per tail (a branch, the hour of its command): [(asked hour, log-probability of the truth)] for its questions
    from that command line on. Every branch's lines before its command line at h are the base day's (the day on its
    own commands with the fixed questions), so the base is run once and its keys and values kept; each tail is then
    scored against the cached positions before its own start, batched."""
    off, n = {}, 0
    for ln, t in zip(base, W.render_episode(base)):
        if ln.kind == "command":
            off[ln.hour] = n
        n += len(T.TOK.encode(t).ids) + 1
    x0 = T.encode(base)[0]
    enc = []   # per tail: (start in the base, its token ids from there, [(position from the start, question index)])
    for L, h in tails:
        x, _, ans = T.encode(L)
        enc.append((off[h], x[off[h]:], [(p - off[h], qi) for p, qi, _ in ans if p >= off[h]]))
    P = -(-max(s for s, _, _ in enc) // 128) * 128   # padded to a multiple of 128: few distinct shapes (MPS keeps a graph per shape)
    ids0 = torch.zeros(1, P, dtype=torch.long); ids0[0, :len(x0[:P])] = torch.tensor(x0[:P])
    cache = Cache()
    hidden_step(m, ids0.to(dev), torch.arange(P, device=dev)[None], (torch.arange(P) < len(x0))[None].to(dev), cache)
    bs = 32 if dev == "cuda" else 8
    out = [[] for _ in tails]
    order = sorted(range(len(tails)), key=lambda i: enc[i][0])   # by start: a chunk needs the prefix only up to its last
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        n = len(idx)
        Tmax = -(-max(len(enc[i][1]) for i in idx) // 64) * 64
        ids = torch.zeros(n, Tmax, dtype=torch.long)
        starts = torch.tensor([enc[i][0] for i in idx])
        lens = torch.tensor([len(enc[i][1]) for i in idx])
        Pc = min(P, -(-int(starts.max()) // 128) * 128)
        pos, opt, tru, who = [], [], [], []
        for b, i in enumerate(idx):
            ids[b, :lens[b]] = torch.tensor(enc[i][1])
            for p, qi in enc[i][2]:
                L = tails[i][0]
                pos.append(b * Tmax + p); opt.append(T.option_ids(L[qi])); tru.append(T.truth_index(L[qi])); who.append((i, L[qi].q["asked"]))
        c = Cache()   # the prefix, shared by the rows: each row attends to the positions before its own start
        c.k = [k[:, :, :Pc].expand(n, -1, -1, -1) for k in cache.k]; c.v = [v[:, :, :Pc].expand(n, -1, -1, -1) for v in cache.v]
        c.pos = cache.pos[:, :Pc].expand(n, -1); c.valid = (torch.arange(Pc)[None] < starts[:, None]).to(dev)
        h = hidden_step(m, ids.to(dev), (starts[:, None] + torch.arange(Tmax)[None]).to(dev),
                        (torch.arange(Tmax)[None] < lens[:, None]).to(dev), c)
        nq = len(pos)
        pad = -(-nq // 64) * 64 - nq
        pos += [0] * pad; opt += [opt[0]] * pad; tru += [0] * pad
        h = h.flatten(0, 1).index_select(0, torch.tensor(pos, device=dev))
        lp = T.option_logprobs(m, h, torch.tensor(opt, device=dev))
        lp = lp.gather(1, torch.tensor(tru, device=dev)[:, None])[:, 0].float().cpu().numpy()[:nq]
        for (i, t), v in zip(who, lp):
            out[i].append((t, float(v)))
        del c, h
        T.empty_cache()
    return out


def rewards(m, scripts, dec, dev, window):
    """R(a) - R(a') at every command point of every day, on the fixed question set (None where the window is empty)."""
    adv = [[None] * len(D) for D in dec]
    for b, (sc, D) in enumerate(zip(scripts, dec)):
        cm = {d["hour"]: d["action"] for d in D}
        tails, owner = [], []   # the a branch then the a' branch of every compared pair
        for j, d in enumerate(D):
            h = d["hour"]
            if h + 1 > W.LAST - 1:
                continue
            if d["alt"] == d["action"]:   # the same branch: no difference to compute
                adv[b][j] = 0.0
                continue
            tails += [(branch(sc, cm, h, d["action"], window), h), (branch(sc, cm, h, d["alt"], window), h)]; owner.append(j)
        if not tails:
            continue
        lp = tail_logp(m, W.replay(sc, FIXED, cm), tails, dev)
        for i, j in enumerate(owner):
            h = dec[b][j]["hour"]
            win = range(h + 1, min(h + window, W.LAST - 1) + 1)
            adv[b][j] = sum(v for t, v in lp[2 * i] if t in win) - sum(v for t, v in lp[2 * i + 1] if t in win)
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--init", required=True); ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--closed", type=int, default=6)
    ap.add_argument("--explore", type=float, default=0.2)
    ap.add_argument("--ref", default=None, help="stage 3: the frozen stage-2 weights the KL term pulls toward")
    ap.add_argument("--beta", type=float, default=0.5); ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--eval-at", default=""); ap.add_argument("--heldout", type=int, default=800)
    ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--world", default="world_i1")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()
    if a.stage == 3:
        a.closed = a.batch
        assert a.ref, "stage 3 needs --ref"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # FP32, as agreed
    log = open(os.path.join(a.out, "log.txt"), "a")
    def say(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    # the run's invariants live in run.json: a resume refuses if one of them changed; any other change is logged
    ck = os.path.join(a.out, "ckpt.pt")
    rj = os.path.join(a.out, "run.json")
    args = dict(vars(a), init=os.path.abspath(a.init), ref=a.ref and os.path.abspath(a.ref))
    inv = ("stage", "init", "ref", "seed", "beta", "explore", "window", "closed", "batch", "lr", "world")
    if os.path.exists(ck) and os.path.exists(rj):
        old = json.load(open(rj))
        changed = {k: (old[k], args[k]) for k in inv if old.get(k) != args[k]}
        if changed:
            sys.exit(f"resume refused: {a.out} was started with different settings: " +
                     ", ".join(f"{k} {v[0]!r} -> {v[1]!r}" for k, v in changed.items()))
        for k in args:
            if k not in inv and old.get(k) != args[k]:
                say(f"argument changed on resume: {k} {old.get(k)!r} -> {args[k]!r}")
    json.dump(args, open(rj, "w"), indent=1)
    t0 = time.time()
    hold, rows = T.build_heldout(os.path.join(T.HERE, f"heldout_{T.TAG}.pkl"), a.heldout)
    say(f"stage {a.stage}; held-out: {len(hold)} episodes ({time.time()-t0:.0f}s); init {a.init}; " + str(vars(a)))
    m = load_state(os.path.join(T.HERE, "base_weights.pt"), a.init, dev)
    m.train()
    ref = load_state(os.path.join(T.HERE, "base_weights.pt"), a.ref, dev).eval() if a.stage == 3 else None
    decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in m.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    start = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu")
        m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); start = s["update"]
        say("resumed at", start)
    evals = sorted(int(x) for x in a.eval_at.split(",") if x)
    if start == 0 and 0 in evals:
        np.save(os.path.join(a.out, "eval_0.npy"), T.evaluate(m, hold, dev)); say("eval 0 done")
    gen = torch.Generator()
    t_last, acc = time.time(), []
    for u in range(start + 1, a.updates + 1):
        lr = a.lr * min(1.0, u / 100)
        for g in opt.param_groups: g["lr"] = lr
        base = a.seed * 10**7 + (5 if a.stage == 2 else 7) * 10**6 + u   # stage 1 used seed*1e7 + u; these days are new
        rng = random.Random(base); gen.manual_seed(base)
        scripted = [encode_cmds(W.simulate(rng)) for _ in range(a.batch - a.closed)]
        scripts = [W.draw_script(rng) for _ in range(a.closed)]
        tm = [time.time()]
        m.eval()
        days, dec = run_days(m, CMDS, scripts, dev, gen=gen, explore=a.explore, second=a.stage == 3)
        T.empty_cache(); tm.append(time.time())
        adv = rewards(m, scripts, dec, dev, a.window) if a.stage == 3 else None
        m.train(); tm.append(time.time())
        closed = [encode_cmds(L) for L in days]
        eps = scripted + closed
        nq = sum(len(e[2]) for e in eps)
        nobs = T.NOBS(eps)
        ncmd = sum(len(e[4]) for e in scripted)
        nrl = sum(x is not None for A in adv for x in A) if adv else 0
        if adv:
            ids_r = T.batch_tensors([e[:4] for e in closed], dev)[0]
            with torch.no_grad():
                lp_ref = command_logprobs(ref, hidden(ref, ids_r).flatten(0, 1), closed, ids_r.shape[1])
            del ids_r
        opt.zero_grad(set_to_none=True)
        fsum = osum = csum = rsum = klsum = 0.0
        split = {True: [], False: []}   # R(a) - R(a') by whether a' leads to a different next place than a
        for part, is_scr in ((scripted, True), (closed, False)):
            if not part:
                continue
            ids, tgt, apos, atru, _, aopt = T.batch_tensors([e[:4] for e in part], dev)
            L = ids.shape[1]
            h = hidden(m, ids).flatten(0, 1)
            o_s = F.cross_entropy(m.head(h).float(), tgt.flatten(), ignore_index=-100, reduction="sum")
            f_s = F.nll_loss(T.option_logprobs(m, h.index_select(0, apos), aopt), atru, ignore_index=-100, reduction="sum")
            loss = f_s / nq + o_s / nobs
            if is_scr:
                c_s = -sum(sum(x) for x in command_logprobs(m, h, part, L))
                loss = loss + c_s / ncmd
                csum += c_s.item()
            elif adv:
                lp = command_logprobs(m, h, part, L)
                pg = 0.0
                for b in range(len(part)):
                    for j, A in enumerate(adv[b]):
                        if A is None:
                            continue
                        kl = lp[b][j].item() - lp_ref[b][j].item()
                        At = A - a.beta * kl
                        pg = pg - At * lp[b][j]
                        rsum += A; klsum += kl
                        d = dec[b][j]
                        if d["alt"] != d["action"]:
                            split[W.next_place(d["alt"], d["place"]) != W.next_place(d["action"], d["place"])].append(A)
                loss = loss + pg / max(nrl, 1)
            loss.backward()
            fsum += f_s.item(); osum += o_s.item()
            del h, ids, tgt, apos, atru, aopt, o_s, f_s, loss
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        T.empty_cache(); tm.append(time.time())
        kinds = [d["action"][0] for D in dec for d in D if not d["even"]]
        n = max(len(kinds), 1)
        acc.append((fsum / nq, osum / nobs, csum / max(ncmd, 1), rsum / max(nrl, 1), klsum / max(nrl, 1),
                    np.mean(split[True] or [0.0]), np.mean(split[False] or [0.0]),
                    kinds.count("listen") / n, kinds.count("look_reeds") / n,
                    sum(k in ("act", "act_go", "go_act") for k in kinds) / n, tm[1] - tm[0], tm[2] - tm[1], tm[3] - tm[2]))
        if u % a.log_every == 0 or u == a.updates:
            f, o, c, r, kl, rm, rp, li, lo, ac, ta, tr, tt = map(np.mean, zip(*acc)); acc = []
            say(f"update {u} forecast {f:.4f} obs {o:.4f} command {c:.3f} adv {r:+.4f} (a' elsewhere {rm:+.4f}, "
                f"same place {rp:+.4f}) kl {kl:+.4f} | own choices: "
                f"listen {li:.3f} look {lo:.3f} acts {ac:.3f} | gnorm {gn.item():.2f} lr {lr:.1e} "
                f"mem {T.mem_bytes()/2**30:.1f}G {(time.time()-t_last)/a.log_every:.2f}s/upd (acting {ta:.1f} reward {tr:.1f} "
                f"train {tt:.1f}) elapsed {(time.time()-t0)/60:.1f}m")
            t_last = time.time()
        if u in evals:
            np.save(os.path.join(a.out, f"eval_{u}.npy"), T.evaluate(m, hold, dev)); say(f"eval {u} done")
            torch.save(m.state_dict(), os.path.join(a.out, f"weights_{u}.pt"))
        if u % 100 == 0 or u == a.updates:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), update=u), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    say("finished", a.updates, f"{(time.time()-t0)/60:.1f} min")


CMDS = Commands(T.TOK)

if __name__ == "__main__":
    main()
