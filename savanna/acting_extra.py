"""The pre-registered extra tests of the acting stages (decisions.md, "After the logic review"), on the same held-out
closed-loop days as seek_test.py (the same scripts and seeds; each model writes its own days).

1. Causal last-listen edit. At a decision point h after which a listen happened earlier today, the last listen is
   edited out of the record (its command becomes rest; the heard clause leaves the next hour line; every other
   command, question and chance draw held, as the battery's act-removed edit). The model's P(listen) at h is read
   on the edited day against the original. Decisive edits raise the ideal's listen value at h (by more than DELTA
   nats); the placebo is the same edit of the listen before the last, which a later listen supersedes, where the
   ideal's listen value stays (within DELTA). Pass: the rise on decisive edits exceeds the placebo's, CI above 0.
   Also reported: last-listen edits where the value does not rise (the same edit, the stricter placebo, fewer).
2. Entropy partial. The entropy of the model's own herd forecast for h+1, asked in a scoring copy right after hour
   h's line and before the command; its partial Spearman with P(listen) after the cue cells (last heard x any sun
   since x any cool since), place, hour and age. Pass: above 0, CI above 0.
3. Resampled-future listen value: the ideal's listen value (seek_test's) averaged over NFUT replays with every
   chance draw after hour h redrawn (the past held). Reported: the partial Spearman of P(listen) with it, and the
   ceiling, the same correlation for an oracle whose P(listen) is a monotone function of the averaged value, taken
   against the single-future value seek_test uses.
4. Within-model own policy. Horizon-2 store questions about the store at the animal's next place, where hour t's
   command is a plain go or rest (the next hour line is sky and level only). The ideal is computed with the routine
   at t+1 and again with the model's own policy at t+1: its first-word probability of the local act word on each
   possible next hour line (3 skies x 3 levels), read in scoring copies. Where the two ideals' top answers differ,
   the share of items where the model's own top answer matches each. Pass: own minus routine agreement above 0,
   CI above 0, on at least 200 items (the count is reported).
5. Stage 3 minus stage 2, day-clustered CIs: the mirror-cell and look-cell gaps; P(listen) in the zero-value cells
   (plain with no sun since; river with sun only since), which must not be above stage 2's; the listen share; the
   forecast loss on water, grass and well questions, each model on both models' days.

--check runs the same code on the routine's days with stand-in policies: the routine (stage 2's role); an oracle
whose P(listen) rises with the ideal's listen value, P(look) with the lion unknown, which rarely acts and whose
answers assume its own policy (golden: every test passes); and a habit that listens and looks as the routine, acts
like the oracle but answers assuming the routine (can-fail: every test fails).

usage: acting_extra.py --stage2 WEIGHTS --stage3 WEIGHTS --days 2000 --out DIR
       acting_extra.py --check --days 64 --out DIR
"""
import argparse, collections, os, pickle, random, sys, time
import multiprocessing as mp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stage2"); ap.add_argument("--stage3")
ap.add_argument("--check", action="store_true", help="stand-in policies on the routine's days (golden and can-fail)")
ap.add_argument("--days", type=int, default=2000); ap.add_argument("--out", required=True)
ap.add_argument("--own-per-day", type=int, default=4, help="own-policy scoring copies per day (test 4's supplement)")
ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
A = ap.parse_args()
sys.argv = [sys.argv[0], "--world", "world_i1"]
import world_i1 as W
import train_act as TA
import seek_test as S            # context(), partial_spearman(), boot(), SEED0, WIN
os.makedirs(A.out, exist_ok=True)
SEED0, WIN = S.SEED0, S.WIN
DELTA = 0.01                     # nats: a listen value that moves less than this under an edit counts as unchanged
NFUT = 8
TOK = TA.T.TOK
CMDS = TA.CMDS
BS = 4                           # days per forward (set per device in main)
STORE_COL = {"pan": W.PAN, "plain": W.GRASS, "bank": W.WELL}
ACT_WORD = {pl: {W.effect(a, pl)[0]: TOK.decode([CMDS.tokens(pl, a)[0]]).strip() for a in W.legal(pl) if a[0] == "act"}
            for pl in W.PLACES}  # the local effects at a place and the first word of the commands that do them


# ---------------------------------------------------------------- the ideal's values

def ideal_R(L, h):
    """seek_test.ideal_R with the record read and filtered once per branch (the same numbers)."""
    win = range(h + 1, min(h + WIN, W.LAST - 1) + 1)
    qs = [ln for ln in L if ln.kind == "question" and ln.q["asked"] in win]
    if not qs:
        return 0.0
    R = W.read_record(L, len(L))
    B = W.filter_to(R, max(ln.q["asked"] for ln in qs))
    tot = 0.0
    for ln in qs:
        d = np.clip(W.forecast(R, B, ln.q["asked"], ln.q["store"], ln.q["T"]), 0, None)
        tot += np.log(max(d[ln.q["truth"]] / d.sum(), 1e-12))
    return tot


def value(sc, cmds, h):
    """The ideal's listen value at h: R after a listen minus after a rest, the routine choosing after h."""
    return ideal_R(TA.branch(sc, cmds, h, ("listen",), WIN), h) - ideal_R(TA.branch(sc, cmds, h, ("rest",), WIN), h)


def resample_future(sc, h, rng):
    """The script with every chance draw after hour h redrawn: the sky chain from hour h's sky, the herd's and the
    routine's draws from h on (the tick h -> h+1 is in the future; the command at h is forced anyway)."""
    s = dict(sc, sky=dict(sc["sky"]), herd_u=dict(sc["herd_u"]), act_u=dict(sc["act_u"]))
    sky = s["sky"][h]
    for t in range(h + 1, W.LAST + 1):
        sky = rng.choices(list(W.SKY_NEXT[sky]), weights=list(W.SKY_NEXT[sky].values()))[0]
        s["sky"][t] = sky
    for t in range(h, W.LAST + 1):
        s["herd_u"][t] = rng.random(); s["act_u"][t] = rng.random()
    return s


def listens_before(cmds, h):
    return [t for t in sorted(cmds) if t < h and cmds[t] == ("listen",)]


def day_values(task):
    """Per decision point of one day (CPU worker): the ideal's listen value, the value after the decisive and the
    placebo edit, and the value averaged over resampled futures."""
    d, cmds = task
    sc = W.draw_script(random.Random(SEED0 + d))
    out = {}
    for h in sorted(cmds):
        if h + 1 > W.LAST - 1:
            continue
        ls = listens_before(cmds, h)
        r = dict(v0=value(sc, cmds, h), dec=None, pla=None)
        if ls:
            r["dec"] = value(sc, {**cmds, ls[-1]: ("rest",)}, h)
        if len(ls) >= 2:
            r["pla"] = value(sc, {**cmds, ls[-2]: ("rest",)}, h)
        r["fut"] = float(np.mean([value(resample_future(sc, h, random.Random(9 * 10**8 + d * 1000 + h * 10 + k)), cmds, h)
                                  for k in range(NFUT)]))
        out[h] = r
    return d, out


def own_forecast(L, qi, firsts):
    """The ideal's answer to a horizon-2 store question with the routine's policy at t+1 (world_i1.forecast), and
    with the model's own: on each possible next hour line (sky, level) the first-word probability of each local act
    word at the next place is the chance of that act; the rest of the policy does nothing to this store."""
    d_std, aux = W.ideal(L, qi)
    R, B = aux["R"], aux["B"]
    q = L[qi].q
    t, store = q["asked"], q["store"]
    act, pl = R["act"][t]
    place1 = W.next_place(act, pl)
    rained = W.rained_by(R, t)
    b1 = W.step_sky(B[t], R["sky"][t], rained, W.effect(act, pl)[0])
    col = STORE_COL[place1]
    out = np.zeros(3)
    for sky1, ps in W.SKY_NEXT[R["sky"][t]].items():
        P = {None: np.ones(W.N)}
        for e, word in ACT_WORD[place1].items():
            P[e] = np.array([firsts[(sky1, lev)].get(word, 0.0) for lev in range(3)])[col]
            P[None] = P[None] - P[e]
        b2 = sum(W.step_sky(b1 * P[e], sky1, rained or sky1 == "rain", e) for e in P)
        out += ps * W.marginal(b2, store)
    return d_std, out / out.sum()


def own_items(L):
    """(question index, hour t, next place) of the horizon-2 store questions about the store at the animal's next
    place where hour t's command is a plain go or rest."""
    out = []
    for qi, ln in enumerate(L):
        if ln.kind != "question" or ln.q["store"] == "herd" or ln.q["T"] != ln.q["asked"] + 2:
            continue
        cmd = next(l for l in L[:qi][::-1] if l.kind == "command")
        if cmd.action[0] in ("go", "rest") and W.STORE_PLACE[ln.q["store"]] == W.next_place(cmd.action, cmd.place):
            out.append((qi, cmd.hour, W.next_place(cmd.action, cmd.place)))
    return out


def hour_index(L, h):
    return next(i for i, ln in enumerate(L) if ln.kind == "hour" and ln.hour == h)


def cmd_index(L, h):
    return next(i for i, ln in enumerate(L) if ln.kind == "command" and ln.hour == h)


# ---------------------------------------------------------------- the policies read

class ModelPolicy:
    """A model's first-word probabilities at command lines and its answer distributions, by full forwards."""
    def __init__(self, m, dev):
        self.m, self.dev = m, dev

    def _batches(self, enc):
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
        for s in range(0, len(order), BS):
            yield order[s:s + BS]

    def _hidden(self, enc, idx):
        import torch
        from fast import hidden
        Lmax = -(-max(len(enc[i][0]) for i in idx) // 256) * 256
        ids = torch.zeros(len(idx), Lmax, dtype=torch.long)
        for b, i in enumerate(idx):
            ids[b, :len(enc[i][0])] = torch.tensor(enc[i][0])
        return hidden(self.m, ids.to(self.dev)).flatten(0, 1), Lmax

    def first_on_days(self, items):
        """Per item (lines, day, commands): {hour: {first word: probability}} at every command line."""
        import torch
        enc = []
        for L, _, _ in items:
            ids, pts = [], []
            for ln, t in zip(L, W.render_episode(L)):
                if ln.kind == "command":
                    pts.append((ln.hour, ln.place, len(ids) + len(CMDS.p0) - 1))   # the logits before the first word
                ids += TOK.encode(t).ids + TA.T.NL
            enc.append((ids, pts))
        out = [{} for _ in items]
        with torch.no_grad():
            for idx in self._batches(enc):
                h, Lmax = self._hidden(enc, idx)
                pos, allow, who = [], [], []
                for b, i in enumerate(idx):
                    for hour, place, p in enc[i][1]:
                        pos.append(b * Lmax + p); allow.append(CMDS.allowed(place, ())); who.append((i, hour))
                K = max(len(a) for a in allow)
                tok = torch.tensor([a + [0] * (K - len(a)) for a in allow], device=self.dev)
                lg = self.m.head(h.index_select(0, torch.tensor(pos, device=self.dev))).float().gather(1, tok).cpu()
                for (i, hour), a, row in zip(who, allow, lg):
                    p = torch.softmax(row[:len(a)], -1)
                    out[i][hour] = {TOK.decode([t]).strip(): float(q) for t, q in zip(a, p)}
                del h
                TA.T.empty_cache()
        return out

    def answers(self, days):
        """Per day: {question index: log-probabilities over its options}."""
        import torch
        enc = [TA.T.encode(L) for L in days]
        out = [{} for _ in days]
        with torch.no_grad():
            for idx in self._batches(enc):
                who = [(i, qi) for i in idx for _, qi, _ in enc[i][2]]
                if not who:
                    continue
                h, Lmax = self._hidden(enc, idx)
                pos = [b * Lmax + p for b, i in enumerate(idx) for p, _, _ in enc[i][2]]
                opt = [o for i in idx for _, _, o in enc[i][2]]
                lp = TA.T.option_logprobs(self.m, h.index_select(0, torch.tensor(pos, device=self.dev)),
                                          torch.tensor(opt, device=self.dev)).float().cpu().numpy()
                for (i, qi), v in zip(who, lp):
                    out[i][qi] = v
                del h
                TA.T.empty_cache()
        return out


class StandIn:
    """A scripted policy for the checks. listen: 'routine' (the routine's first-word mass) or 'oracle' (rises with
    the ideal's listen value; looks when the lion is unknown). acts: 'routine' or 'rare' (the local act word gets
    0.02 if its act can work, else 0: the routine's chance is far higher). answers: 'standard' (the ideal with the
    routine at t+1) or 'own' (with this policy's acts). The first-word masses are read as marginals by the tests."""
    def __init__(self, listen, acts, answers):
        self.listen, self.acts, self.ans = listen, acts, answers

    def routine_first(self, place, hour, lev):
        out = collections.defaultdict(float)
        for p, a in W.policy(place, hour, lev):
            out[TOK.decode([CMDS.tokens(place, a)[0]]).strip()] += p
        return dict(out)

    def first(self, L, i, cmds, d):
        ln = L[i]
        place, h = ln.place, ln.hour
        lev = L[i - 1].level if place != "shade" else None
        f = self.routine_first(place, h, lev)
        if self.acts == "rare":
            for e, word in ACT_WORD[place].items():
                works = lev == 0 if e == "dig" else lev > 0
                f[word] = 0.02 if works else 0.0
        if self.listen == "oracle" and cmds is not None:   # (not for a lone next hour line: only its acts are read)
            if h + 1 <= W.LAST - 1:
                v = value(W.draw_script(random.Random(SEED0 + d)), cmds, h)
                f["listen"] = 0.02 + 0.6 / (1 + np.exp(-8 * (v - 0.35)))
            if place == "pan":
                B = W.filter_to(W.read_record(L, i), h)   # None on a next hour line the record rules out (weight 0 in test 4)
                p_lion = float((B[h] * W.LION).sum()) if B else 0.5
                f["look"] = 0.05 if p_lion < 1e-9 or p_lion > 1 - 1e-9 else 0.3
        return f

    def first_on_days(self, items):
        return [{ln.hour: self.first(L, i, cmds, d) for i, ln in enumerate(L) if ln.kind == "command"} for L, d, cmds in items]

    def answers(self, days):
        out = []
        for L in days:
            own = {qi: place1 for qi, _, place1 in own_items(L)} if self.ans == "own" else {}
            res = {}
            for qi, ln in enumerate(L):
                if ln.kind != "question":
                    continue
                if qi in own:
                    h1 = L[qi].q["asked"] + 1
                    firsts = {(sky, lev): self.first([W.Line("hour", h1, own[qi], sky, level=lev), W.Line("command", h1, own[qi], action=("rest",))], 1, None, None)
                              for sky in W.SKY for lev in range(3)}
                    d = own_forecast(L, qi, firsts)[1]
                else:
                    d = W.ideal(L, qi)[0]
                res[qi] = np.log(np.maximum(d, 1e-12))
            out.append(res)
        return out


# ---------------------------------------------------------------- statistics

class DayMean:
    """The mean of row values, resampled by day: rows are (day, value)."""
    def __init__(self, rows, ndays):
        self.s, self.n = np.zeros(ndays), np.zeros(ndays)
        for d, v in rows:
            self.s[d] += v; self.n[d] += 1
    def __call__(self, di=None):
        s, n = (self.s.sum(), self.n.sum()) if di is None else (self.s[di].sum(), self.n[di].sum())
        return s / n if n else float("nan")
    def __len__(self):
        return int(self.n.sum())


def ci(f, ndays, reps=1000, fmt="+.3f"):
    lo, hi = S.boot(f, ndays, reps=reps)
    return "[too few]" if np.isnan(lo) else f"[{lo:{fmt}}, {hi:{fmt}}]"


def rho_ci(x, y, grp, days, reps=200):
    """Partial Spearman of x and y given the grouping variables, with a day-clustered CI."""
    x, y = np.asarray(x), np.asarray(y)
    grp = [np.asarray(g) for g in grp]
    by_day = collections.defaultdict(list)
    for i, d in enumerate(days):
        by_day[d].append(i)
    dlist = list(by_day.values())
    def rs(di):
        i = np.concatenate([dlist[j] for j in di])
        return S.partial_spearman(x[i], y[i], [g[i] for g in grp])
    r = S.partial_spearman(x, y, grp)
    lo, hi = S.boot(rs, len(dlist), reps=reps)
    return r, lo, hi


# ---------------------------------------------------------------- main

def main():
    out = open(os.path.join(A.out, "acting_extra.md"), "w")
    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); out.write(msg + "\n"); out.flush()
    scripts = [W.draw_script(random.Random(SEED0 + d)) for d in range(A.days)]
    if A.check:
        pols = {"routine": StandIn("routine", "routine", "standard"), "oracle": StandIn("oracle", "rare", "own"),
                "habit": StandIn("routine", "rare", "standard")}
        pairs = [("oracle", "routine", "golden: must pass"), ("habit", "routine", "can-fail: must fail")]
        P(f"# Acting-stage extra tests, check mode: stand-in policies on {A.days} routine days\n")
    else:
        import torch
        from fast import load_state
        dev = "cuda" if torch.cuda.is_available() else "mps"
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # FP32, as agreed
        global BS
        BS = 32 if dev == "cuda" else 4
        pols = {n: ModelPolicy(load_state(os.path.join(TA.T.HERE, "base_weights.pt"), p, dev).eval(), dev)
                for n, p in (("stage2", A.stage2), ("stage3", A.stage3))}
        pairs = [("stage3", "stage2", "pre-registered")]
        P(f"# Acting-stage extra tests: {A.days} held-out closed-loop days per model\n")
    names = list(pols)
    pool = mp.get_context("fork").Pool(A.workers)   # forked workers never touch torch
    days, decs, vals, pedit, ent, own, items4, ans, ient = {}, {}, {}, {}, {}, {}, {}, {}, {}
    t0 = time.time()
    for name in names:
        pol = pols[name]
        t = time.time()
        if A.check:
            days[name] = [W.replay(sc) for sc in scripts]
            decs[name] = [[dict(hour=ln.hour, place=ln.place, action=ln.action) for ln in L if ln.kind == "command"] for L in days[name]]
        else:
            from act import run_days
            gen = torch.Generator().manual_seed(4)
            days[name], decs[name] = [], []
            for b in range(0, A.days, 64):
                L, D = run_days(pol.m, CMDS, scripts[b:b + 64], dev, gen=gen)
                days[name] += L; decs[name] += D
                TA.T.empty_cache()
        cmds = [{x["hour"]: x["action"] for x in D} for D in decs[name]]
        vals[name] = pool.map_async(day_values, list(enumerate(cmds)), chunksize=4)
        # the original days' first words, read the same way as the edited days'
        for D, f in zip(decs[name], pol.first_on_days([(L, d, c) for d, (L, c) in enumerate(zip(days[name], cmds))])):
            for x in D:
                x["first"] = f[x["hour"]]
        P(f"- {name}: {A.days} days written and read ({time.time() - t:.0f}s)")
        # 1. the edited days: one per listen, read at the decision points whose last (decisive) or second-to-last
        # (placebo) listen it is
        t = time.time()
        items, keys = [], []
        for d, (L, c) in enumerate(zip(days[name], cmds)):
            use = collections.defaultdict(list)
            for h in c:
                ls = listens_before(c, h)
                if ls: use[ls[-1]].append(h)
                if len(ls) >= 2: use[ls[-2]].append(h)
            for tl, hs in use.items():
                Le = W.replay(scripts[d], forced_qs=W.questions_of(L), forced_cmds={**c, tl: ("rest",)})
                items.append((TA.cut_after(Le, max(hs)), d, {**c, tl: ("rest",)})); keys.append((d, tl))
        pedit[name] = {}
        for (d, tl), f in zip(keys, pol.first_on_days(items)):
            for h, fw in f.items():
                pedit[name][(d, tl, h)] = fw.get("listen", 0.0)
        P(f"- {name}: {len(items)} edited days read ({time.time() - t:.0f}s)")
        # 2. the herd question before the command, in a scoring copy per decision point
        t = time.time()
        copies, keys = [], []
        for d, L in enumerate(days[name]):
            for ln in L:
                if ln.kind == "command" and ln.hour + 1 <= W.LAST - 1:
                    copies.append(L[:cmd_index(L, ln.hour)] + [W.question(ln.hour, ln.place, "herd", ln.hour + 1)]); keys.append((d, ln.hour))
        ent[name] = {k: W.entropy(np.exp(a[len(c) - 1])) for k, c, a in zip(keys, copies, pol.answers(copies))}
        ient[name] = {k: W.entropy(W.ideal(c, len(c) - 1)[0]) for k, c in zip(keys, copies)}   # the ideal's, for check 6
        P(f"- {name}: {len(copies)} herd-forecast copies read ({time.time() - t:.0f}s)")
        # 4. the own-policy items: the chooser's horizon-2 questions as written, and scoring copies with the same
        # question inserted after the command at eligible decision points (up to --own-per-day a day); then the
        # first words on every possible next hour line, per item hour
        t = time.time()
        items4[name], copies = [], []
        for d, L in enumerate(days[name]):
            for qi, th, place1 in own_items(L):
                items4[name].append(dict(d=d, t=th, place1=place1, kind="written", L=L, qi=qi))
            el = [ln for ln in L if ln.kind == "command" and ln.hour <= W.LAST - 2 and ln.action[0] in ("go", "rest")
                  and W.next_place(ln.action, ln.place) != "shade"]
            for ln in random.Random(SEED0 + d).sample(el, min(len(el), A.own_per_day)):
                place1 = W.next_place(ln.action, ln.place)
                copy = L[:cmd_index(L, ln.hour) + 1] + [W.question(ln.hour, ln.place, W.PLACE_STORE[place1], ln.hour + 2)]
                items4[name].append(dict(d=d, t=ln.hour, place1=place1, kind="copy", L=copy, qi=len(copy) - 1)); copies.append(copy)
        for it, a in zip([it for it in items4[name] if it["kind"] == "copy"], pol.answers(copies)):
            it["answer"] = a[it["qi"]]
        hours = sorted({(it["d"], it["t"], it["place1"]) for it in items4[name]})
        copies, keys = [], []
        for d, th, place1 in hours:
            base = days[name][d][:hour_index(days[name][d], th + 1)]
            for sky in W.SKY:
                for lev in range(3):
                    copies.append((base + [W.Line("hour", th + 1, place1, sky, level=lev), W.Line("command", th + 1, place1, action=("rest",))], d, cmds[d]))
                    keys.append((d, th, sky, lev))
        own[name] = collections.defaultdict(dict)
        for (d, th, sky, lev), f in zip(keys, pol.first_on_days(copies)):
            own[name][(d, th)][(sky, lev)] = f[th + 1]
        P(f"- {name}: {len(copies)} next-hour copies read for {len(items4[name])} own-policy items at {len(hours)} hours ({time.time() - t:.0f}s)")
    # the answers of every policy on every policy's days (4 and 5)
    t = time.time()
    for name in names:
        for dn in names:
            ans[(name, dn)] = pols[name].answers(days[dn])
    P(f"- answers on all days read ({time.time() - t:.0f}s)")
    for name in names:
        vals[name] = dict(vals[name].get())
    pool.close()
    P(f"- the ideal's values done ({(time.time() - t0) / 60:.1f} min in all)\n")

    # ---- rows per policy
    rows, pts = {}, {}
    for name in names:
        rows[name] = []
        for d, (L, D) in enumerate(zip(days[name], decs[name])):
            v = vals[name][d]
            for c, x in zip(S.context(L, D), D):
                h = c["hour"]
                if h in v:
                    c.update(v[h], day=d, ent=ent[name][(d, h)], ient=ient[name][(d, h)], p_rest=x["first"].get("rest", 0.0),
                             cell=f"{c['heard']}/{int(c['sun'])}{int(c['cool'])}", level=L[cmd_index(L, h) - 1].level); rows[name].append(c)
    nd = A.days

    # ---- 1. the causal edit
    P("## 1. Causal last-listen edit\n")
    P("| policy | decisive: n, ideal dV, P(listen) before -> after, rise | placebo: n (dropped), dV, before -> after, rise | "
      "rise difference [95% CI] | same edit, value unchanged: n, rise |")
    P("|---|---|---|---|---|")
    res = {}
    for name in names:
        groups = {"dec": [], "pla": [], "flat": []}
        dropped = 0
        for r in rows[name]:
            d, h = r["day"], r["hour"]
            ls = listens_before({x["hour"]: x["action"] for x in decs[name][d]}, h)
            if r["dec"] is not None:
                row = (d, r["p_listen"], pedit[name][(d, ls[-1], h)], r["dec"] - r["v0"])
                groups["dec" if row[3] > DELTA else "flat"].append(row)
            if r["pla"] is not None:
                if abs(r["pla"] - r["v0"]) < DELTA:
                    groups["pla"].append((d, r["p_listen"], pedit[name][(d, ls[-2], h)], r["pla"] - r["v0"]))
                else:
                    dropped += 1
        M = {g: (DayMean([(d, b) for d, b, _, _ in g_], nd), DayMean([(d, a) for d, _, a, _ in g_], nd), DayMean([(d, a - b) for d, b, a, _ in g_], nd),
                 DayMean([(d, dv) for d, _, _, dv in g_], nd)) for g, g_ in groups.items()}
        diff = lambda di=None: M["dec"][2](di) - M["pla"][2](di)
        res[name] = dict(edit=diff(), edit_ci=S.boot(diff, nd))
        flat = lambda di=None: M["dec"][2](di) - M["flat"][2](di)   # check 6a: against the same edit where the value stays
        res[name].update(edit_flat=flat(), edit_flat_ci=S.boot(flat, nd), n_flat=len(M["flat"][0]))
        P(f"| {name} | {len(M['dec'][0])}, {M['dec'][3]():+.3f}, {M['dec'][0]():.3f} -> {M['dec'][1]():.3f}, {M['dec'][2]():+.4f} | "
          f"{len(M['pla'][0])} ({dropped}), {M['pla'][3]():+.3f}, {M['pla'][0]():.3f} -> {M['pla'][1]():.3f}, {M['pla'][2]():+.4f} | "
          f"**{diff():+.4f}** {ci(diff, nd)} | {len(M['flat'][0])}, {M['flat'][2]():+.4f} |")
    P(f"\nDecisive: the last listen before h removed, the ideal's listen value rising by more than {DELTA} nats. Placebo: the "
      f"listen before it removed (a later listen supersedes it), the value within {DELTA}. Pass: the rise difference above 0 (CI).")

    # ---- 2. the entropy partial
    P("\n## 2. Entropy partial\n")
    P("| policy | points | mean herd-forecast entropy (nats) | partial Spearman (P(listen), entropy | cue cell, place, hour, age) [95% CI] | "
      "also partialling the routine's input cell (place x hour x level) [95% CI] |")
    P("|---|---|---|---|---|")
    for name in names:
        R = rows[name]
        grp = [[x["cell"] for x in R], [x["place"] for x in R], [x["hour"] for x in R], [x["age"] for x in R]]
        r, lo, hi = rho_ci([x["p_listen"] for x in R], [x["ent"] for x in R], grp, [x["day"] for x in R])
        r2, lo2, hi2 = rho_ci([x["p_listen"] for x in R], [x["ent"] for x in R], grp + [[f"{x['place']}{x['hour']}{x['level']}" for x in R]], [x["day"] for x in R])
        res[name].update(ent_rho=r, ent_ci=(lo, hi), ent_rho_lev=r2, ent_ci_lev=(lo2, hi2))
        P(f"| {name} | {len(R)} | {np.mean([x['ent'] for x in R]):.3f} | **{r:+.3f}** [{lo:+.3f}, {hi:+.3f}] | {r2:+.3f} [{lo2:+.3f}, {hi2:+.3f}] |")
    P("\nThe entropy of the model's own answer to 'the herd at hour h+1?' asked right after hour h's line, before the command. "
      "Pass (pre-registered): above 0 (CI). The last column also partials out the joint cell of everything the routine reads "
      "(place, hour, the level printed at hour h): the routine's P(listen) is a function of that cell alone, so it scores nothing "
      "there (NaN), while additive partialling of place and hour leaves it a small positive value, since the level and the "
      "band tilt carry herd information the cue cells miss.")

    # ---- 3. the resampled-future value
    P("\n## 3. Resampled-future listen value\n")
    P(f"| policy | partial Spearman with the single-future value (seek_test's) | with the {NFUT}-future average [95% CI] | "
      "ceiling: an oracle monotone in the average, against the single-future value |")
    P("|---|---|---|---|")
    for name in names:
        R = rows[name]
        grp = [[x["place"] for x in R], [x["hour"] for x in R], [x["age"] for x in R]]
        r1 = S.partial_spearman([x["p_listen"] for x in R], [x["v0"] for x in R], grp)
        r8, lo, hi = rho_ci([x["p_listen"] for x in R], [x["fut"] for x in R], grp, [x["day"] for x in R])
        ceil = S.partial_spearman([x["fut"] for x in R], [x["v0"] for x in R], grp)
        res[name].update(rho1=r1, rho8=r8, rho8_ci=(lo, hi), ceiling=ceil)
        P(f"| {name} | {r1:+.3f} | **{r8:+.3f}** [{lo:+.3f}, {hi:+.3f}] | {ceil:+.3f} |")
    P("\nThe listen value averaged over replays of the day with every chance draw after hour h redrawn; place, hour and age partialled out as in seek_test.")

    # ---- 4. within-model own policy
    P("\n## 4. Within-model own policy\n")
    truth4 = collections.defaultdict(list)
    P("| policy | items | eligible | top answers differ | model agrees with own-policy ideal | with routine ideal | difference [95% CI] |")
    P("|---|---|---|---|---|---|---|")
    for name in names:
        for kind in ("written", "copy"):
            agree, n_el = [], 0
            for it in items4[name]:
                if it["kind"] != kind:
                    continue
                n_el += 1
                d_std, d_own = own_forecast(it["L"], it["qi"], own[name][(it["d"], it["t"])])
                ts, to = W.argmax_set(d_std), W.argmax_set(d_own)
                q = it["L"][it["qi"]].q   # check 6d: both ideals against the truth (a copy's question is answered by its day)
                tr = q["truth"] if kind == "written" else W.truth_of(q, it["L"][0].truth, days[name][it["d"]])
                truth4[(name, kind)].append((it["d"], np.log(max(d_std[tr], 1e-12)), np.log(max(d_own[tr], 1e-12)),
                                             len(ts) == 1 and len(to) == 1 and ts != to))
                if len(ts) == 1 and len(to) == 1 and ts != to:
                    top = int(np.argmax(it["answer"] if kind == "copy" else ans[(name, name)][it["d"]][it["qi"]]))
                    agree.append((it["d"], float(top in to), float(top in ts)))
            Mo, Mr = DayMean([(d, o) for d, o, _ in agree], nd), DayMean([(d, r) for d, _, r in agree], nd)
            diff = lambda di=None: Mo(di) - Mr(di)
            res[name].update({f"own_{kind}": dict(n=len(agree), own=Mo(), routine=Mr(), diff=diff(), ci=S.boot(diff, nd) if agree else (np.nan, np.nan))})
            P(f"| {name} | {'chooser questions as written' if kind == 'written' else f'scoring copies, up to {A.own_per_day} a day'} | {n_el} | {len(agree)} | "
              f"{Mo():.3f} | {Mr():.3f} | **{diff():+.3f}** {ci(diff, nd) if agree else '-'} |")
    P("\nPass (pre-registered, on the questions as written): the difference above 0 (CI) on at least 200 items; fewer items are reported as such. "
      "The scoring copies ask the same question (the store at the next place, two hours on) inserted after the command at eligible "
      "decision points; they are a supplement, not the pre-registered test.")

    # ---- 5. stage differences
    P("\n## 5. Differences between the policies\n")
    def cells(name):
        R = rows[name]
        m = lambda f, key="p_listen": DayMean([(x["day"], x[key]) for x in R if f(x)], nd)
        pan = lambda kn: m(lambda x: x["place"] == "pan" and x["hour"] > W.FIRST + 1 and x["known"] == kn, "p_look")
        return dict(plain_sun=m(lambda x: x["heard"] == "plain" and x["sun"]), plain_none=m(lambda x: x["heard"] == "plain" and not x["sun"]),
                    river_cool=m(lambda x: x["heard"] == "river" and x["cool"]), river_sun=m(lambda x: x["heard"] == "river" and not x["cool"]),
                    look_unknown=pan(False), look_known=pan(True),
                    zero=m(lambda x: (x["heard"] == "plain" and not x["sun"]) or (x["heard"] == "river" and not x["cool"])),
                    share=DayMean([(d, x["action"] == ("listen",) if not A.check else x["first"]["listen"]) for d, D in enumerate(decs[name]) for x in D], nd))
    C = {name: cells(name) for name in names}
    loss = {(mn, dn): DayMean([(d, -a[qi][L[qi].q["truth"]]) for d, L in enumerate(days[dn]) for qi, ln in enumerate(L)
                               if ln.kind == "question" and ln.q["store"] in ("water", "grass", "well") for a in [ans[(mn, dn)][d]]], nd)
            for mn in names for dn in names}
    P("| policy | plain: sun since / none | river: cool since / sun only | P(look): unknown / known | zero-value cells P(listen) | listen share | "
      + " / ".join(f"loss on {dn}'s days" for dn in names) + " |")
    P("|---|---|---|---|---|---|---|")
    for name in names:
        c = C[name]
        P(f"| {name} | {c['plain_sun']():.3f} (n {len(c['plain_sun'])}) / {c['plain_none']():.3f} (n {len(c['plain_none'])}) | "
          f"{c['river_cool']():.3f} (n {len(c['river_cool'])}) / {c['river_sun']():.3f} (n {len(c['river_sun'])}) | "
          f"{c['look_unknown']():.3f} / {c['look_known']():.3f} | {c['zero']():.3f} (n {len(c['zero'])}) | {c['share']():.3f} | "
          + " / ".join(f"{loss[(name, dn)]():.4f}" for dn in names) + " |")
    P("\n| difference | plain gap | river gap | look gap | zero-value cells | listen share | " + " / ".join(f"loss on {dn}'s days" for dn in names) + " |")
    P("|---|---|---|---|---|---|---|")
    for a, b, tag in pairs:
        f = {"plain": lambda di=None: (C[a]["plain_sun"](di) - C[a]["plain_none"](di)) - (C[b]["plain_sun"](di) - C[b]["plain_none"](di)),
             "river": lambda di=None: (C[a]["river_cool"](di) - C[a]["river_sun"](di)) - (C[b]["river_cool"](di) - C[b]["river_sun"](di)),
             "look": lambda di=None: (C[a]["look_unknown"](di) - C[a]["look_known"](di)) - (C[b]["look_unknown"](di) - C[b]["look_known"](di)),
             "zero": lambda di=None: C[a]["zero"](di) - C[b]["zero"](di),
             "share": lambda di=None: C[a]["share"](di) - C[b]["share"](di)}
        for dn in names:
            f[f"loss_{dn}"] = (lambda dn: lambda di=None: loss[(a, dn)](di) - loss[(b, dn)](di))(dn)
        r = {k: (g(), S.boot(g, nd)) for k, g in f.items()}
        res[(a, b)] = r
        P(f"| {a} minus {b} ({tag}) | " + " | ".join(f"**{v:+.4f}** [{lo:+.4f}, {hi:+.4f}]" if k.startswith("loss") else f"**{v:+.3f}** [{lo:+.3f}, {hi:+.3f}]"
                                                      for k, (v, (lo, hi)) in r.items()) + " |")
    P("\nPass: the plain, river and look gaps above 0 (CI); the zero-value cells not above (CI not wholly above 0). The listen share and the "
      "loss on water, grass and well questions are reported.")

    # ---- 6. checks added after the council review (24 Oct 2026), not pre-registered
    P("\n## 6. Checks added after the council review (not pre-registered)\n")
    P("### 6a. The edit against the same edit where the ideal's value does not rise\n")
    P("| policy | rise on decisive edits minus rise on value-unchanged edits [95% CI] | value-unchanged edits |")
    P("|---|---|---|")
    for name in names:
        r = res[name]
        P(f"| {name} | **{r['edit_flat']:+.4f}** [{r['edit_flat_ci'][0]:+.4f}, {r['edit_flat_ci'][1]:+.4f}] | {r['n_flat']} |")
    P("\nThe placebo of section 1 removes an older listen; this baseline removes the same (last) listen where it does not matter, "
      "so any habit of listening more after an edit of the last listen cancels.")

    P("\n### 6b. Its own uncertainty beyond the world's\n")
    bins = lambda v, k=20: np.minimum((S.ranks(np.asarray(v)) * k / len(v)).astype(int), k - 1)
    def ctrl(R):
        return ([[x["cell"] for x in R], [x["place"] for x in R], [x["hour"] for x in R], [x["age"] for x in R],
                 bins([x["ient"] for x in R]), bins([x["v0"] for x in R]), bins([x["fut"] for x in R])])
    def resid_share(v, grp):
        X = [np.ones(len(v))] + [(np.asarray(g) == lev).astype(float) for g in grp for lev in sorted(set(g))[1:]]
        X = np.stack(X, 1); rv = S.ranks(np.asarray(v)); rv = rv - rv.mean()
        e = rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]
        return float(e @ e / (rv @ rv))
    by = {}
    for name in names:
        R = rows[name]
        dl = collections.defaultdict(list)
        for i, x in enumerate(R):
            dl[x["day"]].append(i)
        by[name] = (R, dl, ctrl(R))
    def rho6(name, di=None):
        R, dl, g = by[name]
        i = np.arange(len(R)) if di is None else np.concatenate([dl[j] for j in di if j in dl] or [np.zeros(0, int)])
        return S.partial_spearman(np.array([x["p_listen"] for x in R])[i], np.array([x["ent"] for x in R])[i], [np.asarray(gg)[i] for gg in g])
    P("| policy | partial Spearman (P(listen), its herd-forecast entropy | cue cell, place, hour, age, the ideal's herd entropy, "
      "listen value and 8-future value, each in 20 bins) [95% CI] | its entropy's rank variance left after those controls |")
    P("|---|---|---|")
    for name in names:
        r = rho6(name); lo, hi = S.boot(lambda di: rho6(name, di), nd, reps=200)
        R, _, g = by[name]
        res[name].update(own_unc=r, own_unc_ci=(lo, hi), own_unc_left=resid_share([x["ent"] for x in R], g))
        P(f"| {name} | **{r:+.3f}** [{lo:+.3f}, {hi:+.3f}] | {res[name]['own_unc_left']:.2f} |")
    for a, b, tag in pairs:
        f = lambda di=None: rho6(a, di) - rho6(b, di)
        lo, hi = S.boot(f, nd, reps=200)
        res[(a, b)]["own_unc"] = (f(), (lo, hi))
        P(f"| {a} minus {b} | **{f():+.3f}** [{lo:+.3f}, {hi:+.3f}] | |")
    P("\nWhat the world makes uncertain (the ideal's herd entropy) and what the listen is worth to the ideal are held fixed; "
      "what is left of the model's entropy is its own. A model whose listening reads only the record's features scores about 0.")

    P("\n### 6c. Resting by cell (a pull toward a predictable world would raise it)\n")
    P("| policy | plain, sun since | plain, none | river, cool since | river, sun only | all points |")
    P("|---|---|---|---|---|---|")
    pr = {}
    for name in names:
        R = rows[name]
        m = lambda f: DayMean([(x["day"], x["p_rest"]) for x in R if f(x)], nd)
        pr[name] = [m(lambda x: x["heard"] == "plain" and x["sun"]), m(lambda x: x["heard"] == "plain" and not x["sun"]),
                    m(lambda x: x["heard"] == "river" and x["cool"]), m(lambda x: x["heard"] == "river" and not x["cool"]), m(lambda x: True)]
        P(f"| {name} | " + " | ".join(f"{c():.3f}" for c in pr[name]) + " |")
    for a, b, tag in pairs:
        P(f"| {a} minus {b} | " + " | ".join(f"{pr[a][k]() - pr[b][k]():+.3f} {ci(lambda di, k=k: pr[a][k](di) - pr[b][k](di), nd)}" for k in range(5)) + " |")

    P("\n### 6d. Own policy, positive control: which ideal is right on the model's own days\n")
    P("| policy | items | log-prob of the truth: routine ideal / own-policy ideal (all items) | own minus routine [95% CI] | "
      "on items where their top answers differ: n, routine / own |")
    P("|---|---|---|---|---|")
    for name in names:
        for kind in ("written", "copy"):
            T4 = truth4[(name, kind)]
            if not T4:
                continue
            Ms, Mo = DayMean([(d, s) for d, s, _, _ in T4], nd), DayMean([(d, o) for d, _, o, _ in T4], nd)
            Ds, Do = DayMean([(d, s) for d, s, _, df in T4 if df], nd), DayMean([(d, o) for d, _, o, df in T4 if df], nd)
            g = lambda di=None: Mo(di) - Ms(di)
            res[name][f"own_truth_{kind}"] = (g(), S.boot(g, nd))
            P(f"| {name} | {kind}: {len(T4)} | {Ms():+.4f} / {Mo():+.4f} | **{g():+.4f}** {ci(g, nd, fmt='+.4f')} | "
              f"{len(Ds)}, {Ds():+.3f} / {Do():+.3f} |")
    P("\nIf the own-policy ideal is not closer to the truth than the routine ideal, the model has no reason to forecast with "
      "its own policy and section 4 cannot fail it; if it is, section 4's result stands.")

    # ---- the verdicts
    P("\n## Verdicts\n")
    for a, b, tag in pairs:
        r = res[a]; rd = res[(a, b)]
        ok = dict(edit=r["edit_ci"][0] > 0, entropy=r["ent_ci"][0] > 0, entropy_with_level=r["ent_ci_lev"][0] > 0,
                  own_written=r["own_written"]["ci"][0] > 0 and r["own_written"]["n"] >= 200,
                  own_copies=r["own_copy"]["ci"][0] > 0 and r["own_copy"]["n"] >= 200,
                  gaps=all(rd[k][1][0] > 0 for k in ("plain", "river", "look")), selective=not rd["zero"][1][0] > 0,
                  rho=r["rho8"] >= 0.10)
        P(f"- {a} against {b} ({tag}): " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in ok.items()))
    res["rows"] = rows   # per decision point, for re-analysis
    pickle.dump(res, open(os.path.join(A.out, "acting_extra.pkl"), "wb"))
    P(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
