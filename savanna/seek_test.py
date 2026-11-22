"""Tests of the acting stages on held-out closed-loop days (decisions.md): seeking (the control half of
metacognition) and own policy (the closed-loop self-model test).

Each model writes the commands of the same held-out days (random.Random(4 * 10**6 + d), no even draws).

Seeking. At every command point, the model's probability of listening (and at the pan, of looking in the reeds) is
set against the listen's value there: the ideal forecaster's log-probability of the truth on the fixed question set
(the four stores, asked at hours h+1 to h+3, each about the next hour) after a listen at h, minus after a rest at h,
the routine choosing after h in each branch (stage 3's branch rule, train_act.branch). Reported: the Spearman correlation of P(listen) with that value after place, hour and hours
since the last listen are partialled out; the mirror cells (last heard on the plain: the herd leaves under sun; last
heard at the river: it leaves under cool skies); and P(look) with the lion unknown against known.

Own policy. Both models forecast on both models' days. On a store question two hours ahead about the store where the
animal will be next hour (its own next act can change it), a model whose forecasts assume its own policy does better
on its own days: the interaction (A on A's days - B on A's days) - (A on B's days - B on B's days) is above zero.
The same interaction on two-hour questions about the other stores is the control.

usage: seek_test.py --models NAME=WEIGHTS NAME=WEIGHTS --days 2000 --out DIR
"""
import argparse, collections, os, pickle, random, sys, time
import multiprocessing as mp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="NAME=WEIGHTS; the first two are compared for own policy")
ap.add_argument("--days", type=int, default=2000); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
A = ap.parse_args() if __name__ == "__main__" else None   # acting_extra.py imports the helpers below
sys.argv = [sys.argv[0], "--world", "world_i1"]
import world_i1 as W
import train_act as TA          # branch() and the model plumbing
SEED0 = 4 * 10**6
WIN = 3


def ideal_R(L, h):
    """The ideal's log-probability of the truth, summed over the fixed questions asked at hours h+1..h+3."""
    win = range(h + 1, min(h + WIN, W.LAST - 1) + 1)
    tot = 0.0
    for qi, ln in enumerate(L):
        if ln.kind == "question" and ln.q["asked"] in win:
            d = W.ideal(L, qi)
            tot += np.log(max(d[0][ln.q["truth"]], 1e-12))
    return tot


def point_values(task):
    """Per command point of one day: (hour, listen value) where the window is not empty."""
    d, cmds = task
    sc = W.draw_script(random.Random(SEED0 + d))
    out = []
    for h in sorted(cmds):
        if h + 1 > W.LAST - 1:
            continue
        out.append((h, ideal_R(TA.branch(sc, cmds, h, ("listen",), WIN), h) - ideal_R(TA.branch(sc, cmds, h, ("rest",), WIN), h)))
    return d, out


def context(L, D):
    """Per decision: place, hour, hours since the last listen (dawn counts as one), last heard, whether any sun and
    whether any cool sky (cloud or rain) since then, whether a reeds look was done today, and whether the lion is
    known to the ideal from the record before the command (a look, or store levels that settle it)."""
    out = []
    last_t, last_w, looked = W.FIRST, "plain", False
    hour_line = {ln.hour: ln for ln in L if ln.kind == "hour"}
    cmd_index = {ln.hour: i for i, ln in enumerate(L) if ln.kind == "command"}
    for x in D:
        h = x["hour"]
        for t in range(last_t + 1, h + 1):
            ln = hour_line[t]
            if ln.heard:
                last_t, last_w = t, ln.heard
            if ln.lion:
                looked = True
        sky = [hour_line[t].sky if t > W.FIRST else L[0].sky for t in range(last_t, h + 1)]
        p_lion = float((W.filter_to(W.read_record(L, cmd_index[h]), h)[h] * W.LION).sum())
        out.append(dict(place=x["place"], hour=h, age=h - last_t, heard=last_w, sun="sun" in sky,
                        cool=any(s != "sun" for s in sky), looked=looked, known=p_lion < 1e-9 or p_lion > 1 - 1e-9,
                        p_listen=x["first"].get("listen", 0.0), p_look=x["first"].get("look", 0.0)))
    return out


def ranks(v):
    """Ranks from 0, ties given the mean rank of their run."""
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - cnt + (cnt - 1) / 2)[inv]


def partial_spearman(x, y, groups):
    """Spearman correlation of x and y after both ranks are regressed on dummies of each grouping variable.
    NaN when either residual is (near) constant: there is no ordering to correlate."""
    X = [np.ones(len(x))]
    for g in groups:
        for lev in sorted(set(g))[1:]:
            X.append((np.asarray(g) == lev).astype(float))
    X = np.stack(X, 1)
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    if ex @ ex < 1e-6 * len(x) or ey @ ey < 1e-6 * len(x):
        return float("nan")
    return float(ex @ ey / np.sqrt((ex @ ex) * (ey @ ey)))


def boot(f, n, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, n, n)) for _ in range(reps)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    import torch
    from tokenizers import Tokenizer
    from fast import load_state
    from act import Commands, run_days
    dev = "cuda" if torch.cuda.is_available() else "mps"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # FP32, as agreed
    cmds = Commands(Tokenizer.from_str(open(os.path.join(TA.T.HERE, "tokenizer.json")).read()))
    models = [x.split("=", 1) for x in A.models]
    os.makedirs(A.out, exist_ok=True)
    out = open(os.path.join(A.out, "acting_tests.md"), "w")
    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); out.write(msg + "\n")
    P(f"# Acting-stage tests: {A.days} held-out closed-loop days per model\n")
    days, decs = {}, {}
    for name, path in models:
        t = time.time()
        m = load_state(os.path.join(TA.T.HERE, "base_weights.pt"), path, dev).eval()
        gen = torch.Generator().manual_seed(4)
        days[name], decs[name] = [], []
        for b in range(0, A.days, 64):
            ds = list(range(b, min(A.days, b + 64)))
            L, D = run_days(m, cmds, [W.draw_script(random.Random(SEED0 + d)) for d in ds], dev, gen=gen)
            days[name] += L; decs[name] += D
            TA.T.empty_cache()
        P(f"- {name}: wrote {A.days} days ({time.time() - t:.0f}s)")
        del m
    # ---- seeking
    P("\n## Seeking\n")
    P("| model | points | P(listen) mean | partial Spearman (P(listen), listen value) [95% CI] | plain: any sun since / none | "
      "river: any cool since / sun only since | P(look): lion unknown / known (pan, after hour 6) | command mix (listen, look, acts) |")
    P("|---|---|---|---|---|---|---|---|")
    res = {}
    for name, _ in models:
        tasks = [(d, {x["hour"]: x["action"] for x in D}) for d, D in enumerate(decs[name])]
        # the workers are forked after CUDA is initialised: they must never touch torch
        with mp.get_context("fork").Pool(A.workers) as pool:
            vals = dict(pool.map(point_values, tasks, chunksize=8))
        rows = []
        for d, (L, D) in enumerate(zip(days[name], decs[name])):
            v = dict(vals[d])
            for c in context(L, D):
                if c["hour"] in v:
                    c["value"] = v[c["hour"]]; c["day"] = d; rows.append(c)
        x = np.array([r["p_listen"] for r in rows]); y = np.array([r["value"] for r in rows])
        grp = [[r["place"] for r in rows], [r["hour"] for r in rows], [r["age"] for r in rows]]
        rho = partial_spearman(x, y, grp)
        # 95% CI by resampling days (a day's points are correlated)
        by_day = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_day[r["day"]].append(i)
        dlist = list(by_day.values())
        def rs(di):
            i = np.concatenate([dlist[j] for j in di])
            return partial_spearman(x[i], y[i], [np.asarray(g)[i] for g in grp])
        lo, hi = boot(rs, len(dlist), reps=200)
        def mean(f):
            s = [r["p_listen"] for r in rows if f(r)]
            return f"{np.mean(s):.3f} (n {len(s)})" if s else "-"
        pan = [r for r in rows if r["place"] == "pan" and r["hour"] > W.FIRST + 1]
        lk = lambda kn: np.mean([r["p_look"] for r in pan if r["known"] == kn]) if any(r["known"] == kn for r in pan) else float("nan")
        kinds = collections.Counter(x["action"][0] for D in decs[name] for x in D)
        n = sum(kinds.values())
        P(f"| {name} | {len(rows)} | {x.mean():.3f} | **{rho:.3f}** [{lo:.3f}, {hi:.3f}] | "
          f"{mean(lambda r: r['heard'] == 'plain' and r['sun'])} / {mean(lambda r: r['heard'] == 'plain' and not r['sun'])} | "
          f"{mean(lambda r: r['heard'] == 'river' and r['cool'])} / {mean(lambda r: r['heard'] == 'river' and not r['cool'])} | "
          f"{lk(False):.3f} / {lk(True):.3f} | {kinds['listen'] / n:.3f}, {kinds['look_reeds'] / n:.3f}, "
          f"{(kinds['act'] + kinds['act_go'] + kinds['go_act']) / n:.3f} |")
        res[name] = dict(rho=rho, ci=(lo, hi), rows=rows)
    P("\nListen value: the ideal's summed log-probability of the truth over the next three hours' fixed questions after a "
      "listen, minus after a rest (the routine choosing after h in both). Pass (decisions.md): partial Spearman at least 0.10 for the "
      "stage-3 model and at least 0.05 above stage 2, and both mirror cells in the right order (a 'listen after sun' habit "
      "reaches about 0.18 on the correlation alone and fails the river cell); P(look) higher with the lion unknown.")
    # ---- own policy
    if len(models) >= 2:
        P("\n## Own policy: crossover on two-hour store questions\n")
        (na, pa), (nb, pb) = models[0], models[1]
        lp = {}
        for name, path in models[:2]:
            m = load_state(os.path.join(TA.T.HERE, "base_weights.pt"), path, dev).eval()
            for dn in (na, nb):
                lp[(name, dn)] = TA.question_logp(m, days[dn], dev)
            del m
            TA.T.empty_cache()
        def items(dn, near):
            """(day, question index) of two-hour store questions; near: about the store where the animal is next."""
            out = []
            for d, L in enumerate(days[dn]):
                k = 0
                for qi, ln in enumerate(L):
                    if ln.kind != "question":
                        continue
                    q = ln.q
                    if q["store"] in ("water", "grass", "well") and q["T"] == q["asked"] + 2:
                        cmd = next(l for l in L[:qi][::-1] if l.kind == "command")
                        place1 = W.next_place(cmd.action, cmd.place)
                        if (W.STORE_PLACE[q["store"]] == place1) == near:
                            out.append((d, k))
                    k += 1
            return out
        inter = {}
        for near in (True, False):
            I = {dn: items(dn, near) for dn in (na, nb)}
            g = {(mn, dn): np.array([lp[(mn, dn)][d][k][1] for d, k in I[dn]]) for mn in (na, nb) for dn in (na, nb)}
            da, db = g[(na, na)] - g[(nb, na)], g[(na, nb)] - g[(nb, nb)]
            # 95% CI by resampling days jointly over both models' day sets (day d shares its script in both)
            by_day = {dn: [[] for _ in range(A.days)] for dn in (na, nb)}
            for dn in (na, nb):
                for i, (d, _) in enumerate(I[dn]):
                    by_day[dn][d].append(i)
            def rs(di, da=da, db=db, by_day=by_day):
                ia, ib = (np.array([i for j in di for i in by_day[dn][j]], dtype=int) for dn in (na, nb))
                return da[ia].mean() - db[ib].mean()
            inter[near] = rs
            lo, hi = boot(rs, A.days)
            P(f"- {'next place (own act can change it)' if near else 'other stores (control)'}: {len(da)} + {len(db)} questions; "
              f"{na} minus {nb} on {na}'s days {da.mean():+.4f} nats, on {nb}'s days {db.mean():+.4f}; "
              f"interaction **{da.mean() - db.mean():+.4f}** (95% CI {lo:+.4f} to {hi:+.4f}, days resampled)")
        diff = lambda di: inter[True](di) - inter[False](di)
        lo, hi = boot(diff, A.days)
        P(f"- next place minus control: **{diff(np.arange(A.days)):+.4f}** (95% CI {lo:+.4f} to {hi:+.4f}, days resampled)")
        P(f"\nPass: the interaction on next-place questions above zero (CI) and above the control's.")
    pickle.dump(dict(res={k: dict(rho=v["rho"], ci=v["ci"]) for k, v in res.items()}), open(os.path.join(A.out, "acting_tests.pkl"), "wb"))


if __name__ == "__main__":
    main()
