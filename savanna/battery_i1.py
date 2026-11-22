"""The trait battery for world_i1 (the savanna, iteration 1) on trained checkpoints: access, unified, self-modelling
and metacognition, each against its controls, with one all-trait table per checkpoint and one summary across them.

usage: python battery_i1.py CKPT [CKPT ...] [--days 2500] [--seed 3] [--out DIR] [--per-cell 300] [--golden]
  CKPT: a weights_{u}.pt saved by train.py (a state dict), or a run folder (every weights_*.pt in it); "base" is the
  base model, the negative control. --golden runs no model: the ideal stands in for it and must score 100 on every
  decisive cell and move on no placebo; the blind forecasters' gaps are printed beside it.

Held-out days are new seeds of the same simulator (random.Random(seed * 10**6 + d)); nothing in the world changes.
An edit neutralises one printed line (rain -> cloud; sun <-> cloud; an own act -> its bare move; a listen or a reeds
look -> rest) and the day is replayed from its own chance draws with the questions and commands held fixed
(world_i1.replay), so every later line follows the rules. A decisive edit is one that changes the ideal's top answer;
a placebo edit is one of the same kinds that leaves the ideal's distribution unchanged. Every edit cell is scored
"right on both days": the forecaster's top answer is among the ideal's top answers on the original day and on the
edited day. The ideal scores 100 by construction (the ceiling). Three blind forecasters are scored the same way: one
that reads the edited line in its edited form on the original day ("line-blind": wrong on one of the two days unless
other lines carry the answer), and the world's two rule-blind forecasters (the herd moving at its long-run rates
whatever the sky; reeds looks unread). A cell whose best blind forecaster is within 15 points of the ideal is flagged
as unable to fail. Everything is FP32; the device is CUDA, else MPS, else CPU.

Writes <out>/<run>@<u>.md and .pkl per checkpoint, <out>/summary.md across checkpoints, and caches the tagged days
and edits in <out>/ (built once per --days/--seed; the model runs reuse them)."""
import argparse, collections, copy, glob, hashlib, os, pickle, random, sys, time
import multiprocessing as mp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="*", help="weights_{u}.pt files or run folders; 'base' for the base model")
ap.add_argument("--days", type=int, default=2500)
ap.add_argument("--seed", type=int, default=3, help="held-out day d is random.Random(seed * 10**6 + d)")
ap.add_argument("--out", default=None); ap.add_argument("--per-cell", type=int, default=300)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--golden", action="store_true")
ap.add_argument("--probe-days", type=int, default=1000, help="days whose hidden states feed the transfer probe")
ap.add_argument("--code", default=None, help="folder with train.py, fast.py, tokenizer.json, base_weights.pt")
ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
ap.add_argument("--actor", default=None, help="weights of a stage-2/3 model: it writes the held-out days' commands "
                "(closed-loop days); edits then hold its commands fixed, and store questions two hours ahead are left "
                "out (the ideal's two-hour forecast assumes the routine's next command)")
A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = A.code or HERE
sys.argv = [sys.argv[0], "--world", "world_i1"]          # train.py reads the world module from argv at import
sys.path.insert(0, CODE); sys.path.insert(0, HERE)
import world_i1 as W
import tagger as T
A.out = A.out or os.path.join(HERE, "results")
os.makedirs(A.out, exist_ok=True)
SEED0 = A.seed * 10 ** 6
LINE_KINDS = ("rain", "colour", "herd", "lion", "act", "dig")      # deciding lines that are printed lines (herd_dawn is not)
T0 = time.time()


def pad3(d):
    d = np.asarray(d, float)
    return np.pad(d, (0, 3 - len(d))) if len(d) < 3 else d


def ent(p):
    p = np.asarray(p, float); p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def ok(p, ideal):
    """The forecaster's top answer is among the ideal's top answers (a forecaster with no answer is wrong)."""
    return p is not None and int(np.argmax(p)) in W.argmax_set(ideal)


def blind_ideal(L, qi, rule):
    W.BLIND = {rule}
    try:
        r = W.ideal(L, qi)
    finally:
        W.BLIND = set()
    return None if r is None else pad3(r[0])


# ---------------------------------------------------------------- held-out days, tagged (CPU, once)

ACTOR_CMDS = None   # {day: {hour: action}} written by the actor, or None for the routine's days


def usable(r):
    """Whether a question enters the tables: all of them on the routine's days; on closed-loop days not a store
    question two hours ahead."""
    return ACTOR_CMDS is None or r["store"] in W.ACTS or r["k"] == 1


def actor_commands(path):
    """The actor's commands on every held-out day (sampled at temperature 1, no even draws, a fixed generator)."""
    import torch
    from fast import load_state
    from tokenizers import Tokenizer
    from act import Commands, run_days
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False     # FP32, as agreed
    dev = device()
    m = load_state(os.path.join(CODE, "base_weights.pt"), path, dev).eval()
    cmds = Commands(Tokenizer.from_str(open(os.path.join(CODE, "tokenizer.json")).read()))
    gen = torch.Generator().manual_seed(A.seed)
    out = {}
    for b in range(0, A.days, 64):
        ds = list(range(b, min(A.days, b + 64)))
        _, dec = run_days(m, cmds, [W.draw_script(random.Random(SEED0 + d)) for d in ds], dev, gen=gen)
        for d, D in zip(ds, dec):
            out[d] = {x["hour"]: x["action"] for x in D}
    return out


def one_day(d):
    sc = W.draw_script(random.Random(SEED0 + d))
    L = W.replay(sc, forced_cmds=ACTOR_CMDS[d]) if ACTOR_CMDS is not None else W.replay(sc)
    tags = {}
    for qi, ln in enumerate(L):
        if ln.kind == "question":
            r = T.tag_question(L, qi)
            r["ideal"] = pad3(r["ideal"]); r["day"] = d; r["qi"] = qi
            r["sky_blind"] = blind_ideal(L, qi, "sky_herd"); r["lion_blind"] = blind_ideal(L, qi, "lion")
            tags[qi] = r
    return L, tags


def reader_day(d):
    """(store, last word, hours unseen, horizon, place, hour, ideal) per question of a fresh training-like day: what
    the no-rule and habit readers are fitted on."""
    L = W.simulate(random.Random(SEED0 + 500_000 + d))
    out = []
    for qi, ln in enumerate(L):
        if ln.kind != "question":
            continue
        q = ln.q; store, t, Tq = q["store"], q["asked"], q["T"]
        if store in W.ACTS:
            cmd = next(l for l in L[:qi][::-1] if l.kind == "command")
            a_store = W.ACT_STORE[W.effect(cmd.action, cmd.place)[0]]
        else:
            a_store = store
        s, w = T.last_sight(L, qi, a_store)
        hidden = max(0, (t if store in W.ACTS else Tq) - s - (0 if store in W.ACTS else 1))
        out.append((store, w, min(hidden, 6), Tq - t, ln.place, t, pad3(W.ideal(L, qi)[0])))
    return out


def norule_key(r):
    return (r["store"], r["last_w"], min(r["hidden"], 6), r["k"])


def habit_key(r):
    return (r["store"], r["place"], r["asked"], r["k"])


# ---------------------------------------------------------------- edits (CPU, once)

def edit_op(L, d):
    """The edit that neutralises deciding line d = (kind, form, reveal index, distance): ('sky', h, new) or
    ('cmd', h, action). None for herd_dawn (a chance draw, not a printed line)."""
    kind, form, i, _ = d
    ln = L[i]
    if kind == "rain":
        return ("sky", ln.hour, "cloud")
    if kind == "colour":
        return ("sky", ln.hour, "cloud" if ln.sky == "sun" else "sun")
    if kind in ("herd", "lion"):                       # the listen / reeds look commanded the hour before this line
        return ("cmd", ln.hour - 1, ("rest",))
    if kind in ("act", "dig"):
        a = ln.action
        return ("cmd", ln.hour, ("rest",) if a[0] == "act" else ("go", a[1] if a[0] == "go_act" else a[2]))
    return None


def apply_edit(L, qi, op):
    """Replay day L with one edit, questions and commands held fixed: (the edited day cut at the same question, its
    index, the number of rendered lines that differ before the question)."""
    sc = copy.deepcopy(L[0].script); cmds = {ln.hour: ln.action for ln in L if ln.kind == "command"}
    if op[0] == "sky":
        sc["sky"][op[1]] = op[2]
    else:
        cmds[op[1]] = op[2]
    new = W.replay(sc, forced_qs=W.questions_of(L), forced_cmds=cmds)
    q = L[qi].q
    qn = next(i for i, ln in enumerate(new) if ln.kind == "question" and ln.hour == q["asked"]
              and ln.q["store"] == q["store"] and ln.q["T"] == q["T"])
    cut = new[:qn + 1]
    a, b = W.render_episode(cut), W.render_episode(L[:qi + 1])
    n_changed = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return cut, qn, n_changed


def line_blind(L, qi, op):
    """The ideal reading the original day with the edited line already in its edited form, nothing replayed: a
    forecaster blind to that line. None when the record then becomes impossible."""
    L2 = list(L)
    for i, ln in enumerate(L2):
        if op[0] == "sky" and ln.kind in ("morning", "hour") and ln.hour == op[1]:
            L2[i] = copy.copy(ln); L2[i].sky = op[2]
        elif op[0] == "cmd" and ln.kind == "command" and ln.hour == op[1]:
            L2[i] = copy.copy(ln); L2[i].action = op[2]
            j = next((k for k in range(i + 1, len(L2)) if L2[k].kind == "hour"), None)
            if j is not None:          # the result of the old command on the next line goes with it
                L2[j] = copy.copy(L2[j]); L2[j].heard = L2[j].lion = L2[j].outcome = ""
    r = W.ideal(L2, qi)
    return None if r is None else pad3(r[0])


def placebo_op(rng, L, qi, r, avoid):
    """An edit of the decisive kinds whose line the ideal ignores for this question: the rendered text changes, the
    ideal's distribution does not. Tries sky colours in the window and own acts on other stores."""
    cands = []
    for h in range(W.FIRST, r["asked"] + 1):
        sky = L[0].script["sky"][h]
        if sky != "rain" and h not in avoid:
            cands.append(("sky", h, "cloud" if sky == "sun" else "sun"))
    for ln in L[:qi]:
        if ln.kind == "command" and ln.hour not in avoid and ln.action[0] in ("act", "act_go", "go_act"):
            e, _ = W.effect(ln.action, ln.place)
            if e and W.ACT_STORE[e] != r["a_store"]:
                a = ln.action
                cands.append(("cmd", ln.hour, ("rest",) if a[0] == "act" else ("go", a[1] if a[0] == "go_act" else a[2])))
    rng.shuffle(cands)
    for op in cands[:12]:
        cut, qn, n_changed = apply_edit(L, qi, op)
        if n_changed == 0:
            continue
        d1 = W.ideal(cut, qn)
        if d1 is not None and np.allclose(pad3(d1[0]), r["ideal"], atol=1e-9):
            return op, cut, qn, n_changed
    return None


DAYS = []        # [(lines, tags)], shared with the forked workers


def walk_hours(L, r):
    """Hours in the window with sun and the herd on the plain (gate A's in-view cause)."""
    sc, hist = L[0].script, L[0].truth
    s, _ = T.last_sight(L, r["qi"], r["a_store"])
    return {h for h in range(s, r["asked"] + 1) if sc["sky"][h] == "sun" and hist[h]["herd"] == 0}


def eat_hours(L, r):
    sc, hist = L[0].script, L[0].truth
    s, _ = T.last_sight(L, r["qi"], r["a_store"])
    end = r["asked"] if r["store"] in W.ACTS else r["T"]
    return {h for h in range(s, min(r["asked"], end - 1) + 1) if sc["sky"][h] == "cloud" and hist[h]["herd"] == 0 and hist[h]["grass"] > 0}


def pick_line(cell, L, r, rng):
    """The deciding line a cell edits, as (sub-label, line); None if the question has none of that kind."""
    dec = [d for d in r["decided"] if d[0] in LINE_KINDS]
    if cell == "control":
        lines = [d for d in dec if d[0] == "rain"]
    elif cell == "access":
        lines = [d for d in r["access"] if d[0] in LINE_KINDS]
    elif cell in ("unified_carried", "both_lion"):
        lines = [d for d in dec if d[0] == "lion"]
    elif cell in ("unified_view", "both_sun"):
        wh = walk_hours(L, r); lines = [d for d in dec if d[0] == "colour" and L[d[2]].hour in wh]
    elif cell == "gate_b_carried":
        lines = [d for d in dec if d[0] == "herd" and d[1] == "plain"]
    elif cell == "gate_b_view":
        eh = eat_hours(L, r); lines = [d for d in dec if d[0] == "colour" and L[d[2]].hour in eh]
    elif cell == "own":
        lines = [d for d in dec if d[0] in ("act", "dig") and d[1] == "act_go"]
    elif cell == "efficacy":
        lines = dec
    else:
        raise ValueError(cell)
    if not lines:
        return None
    d = rng.choice(lines)
    sub = d[0]
    if cell == "access":          # breadth: the same line decides another store of the day
        others = {t2["a_store"] for t2 in DAYS[r["day"]][1].values() if any(x[2] == d[2] for x in t2["decided"])} - {r["a_store"]}
        sub = "breadth" if others else "one store"
    if cell == "efficacy":
        sub = r["store"]
    return sub, d


def build_one(task):
    """One decisive edit (and its placebo on the same question) as records plus their cut days."""
    cell, day, qi = task
    L, tags = DAYS[day]; r = tags[qi]
    rng = random.Random(f"{cell}|{day}|{qi}")   # a string seed is the same in every process; hash() is salted
    recs, cuts = [], []
    if cell == "look":
        li = next((i for i in range(qi) if L[i].kind == "command" and L[i].action == ("look_reeds",)), None)
        if li is None:
            return [], []
        op, d, sub = ("cmd", L[li].hour, ("rest",)), None, "info"
        dist = qi - T.hour_lines(L)[L[li].hour + 1]
    else:
        pk = pick_line(cell, L, r, rng)
        if pk is None:
            return [], []
        sub, d = pk
        op, dist = edit_op(L, d), d[3]
    cut, qn, n_changed = apply_edit(L, qi, op)
    d1 = W.ideal(cut, qn)
    if d1 is None:
        return [], []
    ideal1 = pad3(d1[0])
    flip = W.argmax_set(ideal1) != set(r["base"])
    rec = dict(cell=cell, sub=sub, day=day, qi=qi, qn=qn, op=op, kind=(d[0] if d else "lion"), dist=dist, n_changed=n_changed,
               placebo=False, flip=flip, ideal0=r["ideal"], ideal1=ideal1, store=r["store"], a_store=r["a_store"],
               hidden=r["hidden"], k=r["k"], chance1=ideal1.max() < 1 - 1e-9, look=r["look"],
               f=dict(ideal=(r["ideal"], ideal1), line_blind=(line_blind(L, qi, op), ideal1),
                      sky_blind=(r["sky_blind"], blind_ideal(cut, qn, "sky_herd")), lion_blind=(r["lion_blind"], blind_ideal(cut, qn, "lion"))))
    recs.append(rec); cuts.append(cut)
    pl = placebo_op(rng, L, qi, r, avoid={op[1]})
    if pl is not None:
        pop, pcut, pqn, pn = pl
        recs.append(dict(cell=cell, sub=sub, day=day, qi=qi, qn=pqn, op=pop, kind=pop[0], dist=0, n_changed=pn, placebo=True,
                         flip=False, ideal0=r["ideal"], ideal1=r["ideal"], store=r["store"], a_store=r["a_store"], hidden=r["hidden"],
                         k=r["k"], chance1=r["chance"], look=None, f=dict(ideal=(r["ideal"], r["ideal"]))))
        cuts.append(pcut)
    return recs, cuts


CELLS = {                   # cell -> the questions it draws from
    "control": lambda r: r["store"] in ("water", "grass") and r["hidden"] >= 1 and any(d[0] == "rain" for d in r["decided"]),
    "access": lambda r: r["store"] not in W.ACTS and any(d[0] in LINE_KINDS for d in r["access"]),
    "unified_carried": lambda r: bool(r["g2"]) and r["g2"]["carried"],
    "unified_view": lambda r: bool(r["g2"]) and r["g2"]["view"],
    "both_lion": lambda r: bool(r["g2"]) and r["g2"]["carried"] and r["g2"]["view"],
    "both_sun": lambda r: bool(r["g2"]) and r["g2"]["carried"] and r["g2"]["view"],
    "gate_b_carried": lambda r: bool(r["g1"]) and r["g1"]["carried"],
    "gate_b_view": lambda r: bool(r["g1"]) and r["g1"]["view"],
    "own": lambda r: r["own_go"] and r["store"] not in W.ACTS,
    "efficacy": lambda r: r["store"] in W.ACTS and any(d[0] in LINE_KINDS for d in r["decided"]),
    "look": lambda r: bool(r["look"]) and r["look"]["info"],
}


def origin_pairs(tags_all):
    """Matched origin pairs (measure.py's definition): an own-decided changed question against a world-line-decided
    one, same store / last word / hours unseen / horizon, different days, different top answers."""
    own, wld = collections.defaultdict(list), collections.defaultdict(list)
    for r in tags_all:
        if not r["changed"] or r["a_store"] == "herd" or r["store"] in W.ACTS:
            continue
        kinds = {d[0] for d in r["decided"]}
        key = (r["a_store"], r["last_w"], min(r["hidden"], 6), r["k"])
        if r["own_go"] and not (kinds & {"lion", "colour", "rain", "herd"}):
            own[key].append(r)
        elif (kinds & {"lion", "colour", "rain", "herd"}) and not (kinds & {"act", "dig"}):
            wld[key].append(r)
    pairs = []
    for k in own:
        a, b = own[k], wld.get(k, [])
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i]["day"] != b[j]["day"] and set(a[i]["base"]) != set(b[j]["base"]):
                pairs.append(((a[i]["day"], a[i]["qi"]), (b[j]["day"], b[j]["qi"]))); i += 1; j += 1
            elif a[i]["day"] == b[j]["day"]:
                j += 1
            else:
                i += 1; j += 1
    return pairs


def build_all():
    """Tag the held-out days, fit the readers, build every edit and the origin pairs; cached in --out."""
    global DAYS
    global ACTOR_CMDS
    actor = os.path.abspath(A.actor) if A.actor else None
    tag = "" if not actor else ("_actor-" + os.path.basename(os.path.dirname(actor)) + "-" + os.path.basename(actor)[:-3]
                                + "-" + hashlib.sha1(actor.encode()).hexdigest()[:8])
    cache = os.path.join(A.out, f"edits_{A.days}_{A.seed}_{A.per_cell}{tag}.pkl")
    if os.path.exists(cache):
        E = pickle.load(open(cache, "rb"))
        if E.get("actor_path") != actor:
            sys.exit(f"{cache} was built for actor {E.get('actor_path')!r}, not {actor!r}: use another --out")
        DAYS = E["days"]; ACTOR_CMDS = E.get("actor_cmds"); return E
    if A.actor:
        ACTOR_CMDS = actor_commands(A.actor)
        print(f"actor wrote the commands of {A.days} days ({time.time() - T0:.0f}s)", flush=True)
    # the workers are forked after CUDA may be initialised (the actor): they must never touch torch
    with mp.get_context("fork").Pool(A.workers) as pool:
        DAYS = pool.map(one_day, range(A.days), chunksize=8)
        rd = pool.map(reader_day, range(1500), chunksize=8)
    print(f"{A.days} days tagged, 1500 reader days ({time.time() - T0:.0f}s)", flush=True)
    readers = {"no-rule": collections.defaultdict(lambda: np.zeros(3)), "habit": collections.defaultdict(lambda: np.zeros(3))}
    for rows in rd:
        for store, w, hidden, k, place, t, d in rows:
            readers["no-rule"][(store, w, hidden, k)] += d; readers["habit"][(store, place, t, k)] += d
    readers = {k: dict(v) for k, v in readers.items()}
    rng = random.Random(A.seed)
    tasks = []
    for cell, f in CELLS.items():
        cand = [(cell, d, qi) for d, (L, tags) in enumerate(DAYS) for qi, r in tags.items() if usable(r) and f(r)]
        rng.shuffle(cand)
        if cell == "both_sun":            # the same questions as both_lion, so the two edits pair up
            continue
        tasks += cand[:int(A.per_cell * 1.2)]
        if cell == "both_lion":
            tasks += [("both_sun", d, qi) for _, d, qi in cand[:int(A.per_cell * 1.2)]]
    # forked workers again: no torch in build_one
    with mp.get_context("fork").Pool(A.workers) as pool:
        built = pool.map(build_one, tasks, chunksize=4)
    recs, cuts = [], []
    for rs, cs in built:
        for r, c in zip(rs, cs):
            r["cut"] = len(cuts); recs.append(r); cuts.append(c)
    # the both-causes cell: keep only questions where both edits were built
    have = {(r["cell"], r["day"], r["qi"]) for r in recs if not r["placebo"]}
    for r in recs:
        if r["cell"] in ("both_lion", "both_sun"):
            other = "both_sun" if r["cell"] == "both_lion" else "both_lion"
            r["paired"] = (other, r["day"], r["qi"]) in have
    pairs = origin_pairs([r for L, tags in DAYS for r in tags.values() if usable(r)])
    E = dict(days=DAYS, recs=recs, cuts=cuts, pairs=pairs, readers=readers, actor_cmds=ACTOR_CMDS, actor_path=actor)
    pickle.dump(E, open(cache, "wb"))
    print(f"{len(recs)} edit records ({sum(not r['placebo'] for r in recs)} decisive, {sum(r['placebo'] for r in recs)} placebo), "
          f"{len(pairs)} origin pairs ({time.time() - T0:.0f}s)", flush=True)
    return E


# ---------------------------------------------------------------- the model

def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path):
    import torch
    from fast import load
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False     # FP32, as agreed
    m = load(os.path.join(CODE, "base_weights.pt"), DEV)
    if path != "base":
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd["model"] if "model" in sd else sd)
    return m.eval()


def model_probs(m, enc, keep_hidden=0):
    """Probabilities over each question's options (spare slot 0) for every question of every encoded episode, and
    the final hidden state at each answer position for the first keep_hidden episodes."""
    import torch
    from train import batch_tensors, option_logprobs
    from fast import hidden
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    probs, hs = [None] * len(enc), [None] * len(enc)
    with torch.no_grad():
        for b in range(0, len(order), A.batch):
            idx = order[b:b + A.batch]
            eps = [(enc[i][0], enc[i][1], enc[i][2], [0] * len(enc[i][2])) for i in idx]
            ids, _, apos, _, n, aopt = batch_tensors(eps, DEV)
            h = hidden(m, ids).flatten(0, 1).index_select(0, apos)
            p = option_logprobs(m, h, aopt).exp().float().cpu().numpy()[:n]
            hh = h[:n].float().cpu().numpy() if any(i < keep_hidden for i in idx) else None
            j = 0
            for i in idx:
                k = len(enc[i][2]); probs[i] = p[j:j + k]
                if i < keep_hidden:
                    hs[i] = hh[j:j + k]
                j += k
            if DEV == "mps":
                torch.mps.empty_cache()
    return probs, hs


# ---------------------------------------------------------------- scoring and tables

def both(recs, f):
    return 100 * np.mean([ok(r["f"][f][0], r["ideal0"]) and ok(r["f"][f][1], r["ideal1"]) for r in recs]) if recs else np.nan


def follows(recs, f):
    return 100 * np.mean([ok(r["f"][f][1], r["ideal1"]) for r in recs]) if recs else np.nan


def moved(recs, f):
    return 100 * np.mean([int(np.argmax(r["f"][f][0])) != int(np.argmax(r["f"][f][1])) for r in recs]) if recs else np.nan


def shift(recs, f):
    return np.mean([tv(r["f"][f][0], r["f"][f][1]) for r in recs]) if recs else np.nan


BLIND = ("line_blind", "sky_blind", "lion_blind")
EDIT_HDR = ("| cell | n | ideal | line-blind | sky-blind | lion-blind | model: right on both | follows the edit | shift | "
            "n placebo | placebo right on both | placebo top moved | placebo shift | can fail |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def edit_row(name, dec, plc, S, key, rule_blind=None):
    """One table row; the model's number is also stored in S[key] for the summary. 'can fail' judges the forecasters
    blind to this cell's own condition: line-blind always, plus the rule-blind one named (lion or sky)."""
    if not dec:
        return f"| {name} | 0 | | | | | | | | | | | | |"
    b = {f: both(dec, f) for f in BLIND}
    rel = max(b["line_blind"], b[rule_blind]) if rule_blind else b["line_blind"]
    can = 100 - rel >= 15
    S[key] = both(dec, "model"); S[key + " ideal"] = 100.0; S[key + " blind"] = rel
    S[key + " placebo moved"] = moved(plc, "model") if plc else np.nan
    return (f"| {name} | {len(dec)} | 100 | {b['line_blind']:.0f} | {b['sky_blind']:.0f} | {b['lion_blind']:.0f} | **{S[key]:.0f}** | "
            f"{follows(dec, 'model'):.0f} | {shift(dec, 'model'):.2f} | {len(plc)} | {both(plc, 'model'):.0f} | {moved(plc, 'model'):.1f} | "
            f"{shift(plc, 'model'):.3f} | {'yes' if can else 'NO: unable to fail'} |")


def rows_of(recs, **sel):
    return [r for r in recs if all(r.get(k) == v for k, v in sel.items())]


def auc(score, y):
    """Rank AUC of a score for predicting y = 1."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    ranks = np.empty(len(score)); order = np.argsort(score, kind="mergesort")
    ranks[order] = np.arange(1, len(score) + 1)
    # ties share the mean rank
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2; i = j + 1
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


PROBE_SPLITS = ((("water", "grass"), ("herd", "well", "acts")), (("herd", "well"), ("water", "grass")), (("water", "grass", "herd", "well"), ("acts",)))


def probe(X, M, truth, grp):
    """Ridge read-out of 'the model's top answer is wrong' from the answer-position hidden state, standardised and
    centred on the fit stores (an intercept), scored by AUC on the held-out stores, pooled and per store. Without the
    intercept the weights absorb the fit stores' offset and the held-out stores, which sit at another mean, rank
    backwards. Rows: (fit, test, n test, wrong rate, read-out AUC, per-store AUCs, model's confidence AUC, ideal's)."""
    wrong = M.argmax(1) != truth
    conf = 1 - M.max(1)
    out = []
    for fit, test in PROBE_SPLITS:
        a, b = np.isin(grp, fit), np.isin(grp, test)
        if a.sum() < 50 or b.sum() < 50 or wrong[a].all() or not wrong[a].any():
            continue
        mu, sd = X[a].mean(0), X[a].std(0) + 1e-6
        Xa, ya = (X[a] - mu) / sd, wrong[a].astype(float) - wrong[a].mean()
        w = np.linalg.solve(Xa.T @ Xa + 0.1 * a.sum() * np.eye(X.shape[1]), Xa.T @ ya)
        s = ((X[b] - mu) / sd) @ w
        per = {g: auc(s[grp[b] == g], wrong[b][grp[b] == g]) for g in test}
        out.append((fit, test, int(b.sum()), float(wrong[b].mean()), auc(s, wrong[b]), per, auc(conf[b], wrong[b]), None))
    return out


def report(name, E, probs0, hs):
    """The all-trait table of one checkpoint; returns the summary numbers."""
    days, recs, pairs, readers = E["days"], E["recs"], E["pairs"], E["readers"]
    rows = [r for L, tags in days for r in tags.values() if usable(r)]
    qix = {}
    for L, tags in days:
        for j, qi in enumerate(sorted(tags)):
            qix[(tags[qi]["day"], qi)] = j
    M = [probs0[r["day"]][qix[(r["day"], r["qi"])]] for r in rows]
    for r in recs:
        r["f"]["model"] = (probs0[r["day"]][qix[(r["day"], r["qi"])]], r["model1"])
    S = {}
    out = open(os.path.join(A.out, f"{name}.md"), "w")

    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); out.write(msg + "\n")

    P(f"# {name}: {A.days} held-out days, {len(rows)} questions ({DEV}, FP32)\n")
    # ---- yardstick
    P("## Yardstick: agreement with the ideal's top answer on changed questions (copying the last printed word is wrong)")
    P("Readers: copier (the last printed word), no-rule (store, last word, hours unseen, horizon), habit (store, place, hour, horizon), "
      "the two rule-blind forecasters, the model. The ideal is 100 by definition.\n")
    chg = [i for i, r in enumerate(rows) if r["changed"]]
    all_ok = np.array([ok(M[i], rows[i]["ideal"]) for i in range(len(rows))])
    kl = np.array([sum(p * np.log(p / max(q, 1e-12)) for p, q in zip(r["ideal"], M[i]) if p > 0) for i, r in enumerate(rows)])
    S["all questions"] = 100 * all_ok.mean(); S["gap (nats)"] = float(kl.mean())
    P(f"- all questions: model agrees {S['all questions']:.1f}%, mean excess loss over the ideal {S['gap (nats)']:.3f} nats; "
      f"changed questions {len(chg)} ({100 * len(chg) / len(rows):.1f}%)")
    P("\n| cell | n changed | copier | no-rule | habit | sky-blind | lion-blind | model |\n|---|---|---|---|---|---|---|---|")

    def reader(r, which):
        v = readers[which].get(norule_key(r) if which == "no-rule" else habit_key(r))
        return None if v is None else v + 1e-6

    def cell(nm, idx):
        if not idx:
            return
        c = {"copier": 100 * np.mean([rows[i]["copier"] in rows[i]["base"] for i in idx])}
        for w in ("no-rule", "habit"):
            c[w] = 100 * np.mean([ok(reader(rows[i], w), rows[i]["ideal"]) for i in idx])
        for w in ("sky_blind", "lion_blind"):
            c[w] = 100 * np.mean([ok(rows[i][w], rows[i]["ideal"]) for i in idx])
        c["model"] = 100 * all_ok[idx].mean(); S["changed: " + nm] = c["model"]
        P(f"| {nm} | {len(idx)} | {c['copier']:.0f} | {c['no-rule']:.0f} | {c['habit']:.0f} | {c['sky_blind']:.0f} | {c['lion_blind']:.0f} | **{c['model']:.0f}** |")
    cell("all", chg)
    for st in ("water", "grass", "well", "herd"):
        cell(st, [i for i in chg if rows[i]["store"] == st])
    cell("acts (drink, graze, dig)", [i for i in chg if rows[i]["store"] in W.ACTS])
    for nm, f in (("steps 0-1", lambda r: r["steps"] <= 1), ("steps 2-3", lambda r: 2 <= r["steps"] <= 3), ("steps 4+", lambda r: r["steps"] >= 4),
                  ("unseen 1", lambda r: r["hidden"] == 1), ("unseen 2-3", lambda r: 2 <= r["hidden"] <= 3), ("unseen 4+", lambda r: r["hidden"] >= 4)):
        cell(nm, [i for i in chg if f(rows[i])])
    # ---- edits
    dec = [r for r in recs if not r["placebo"]]; plc = [r for r in recs if r["placebo"]]
    P("\n## Decisive edits against placebo edits, scored right on both days\n")
    P("A decisive edit neutralises the one deciding line and the day is replayed; a placebo edit changes a line of the same kinds that the ideal "
      "ignores for that question. 'follows the edit': right on the edited day alone. 'shift': mean total-variation move of the answer "
      "distribution between the two days. 'can fail': the best blind forecaster is at least 15 points under the ideal.\n")
    P("### Positive control and access\n" + EDIT_HDR)
    P(edit_row("plain link: rain raises the pan or the grass, out of view", rows_of(dec, cell="control"), rows_of(plc, cell="control"), S, "control"))
    acc, accp = rows_of(dec, cell="access"), rows_of(plc, cell="access")
    P(edit_row("access: a carried deciding line", acc, accp, S, "access"))
    P(edit_row("access, breadth (the line also decides another store)", rows_of(acc, sub="breadth"), rows_of(accp, sub="breadth"), S, "access breadth"))
    for lo, hi, nm in ((1, 10, "1-10"), (11, 25, "11-25"), (26, 999, "26+")):
        P(edit_row(f"access, deciding line {nm} lines back", [r for r in acc if lo <= r["dist"] <= hi], [], S, f"access {nm}"))
    for kd in LINE_KINDS:
        P(edit_row(f"access, line kind {kd}", rows_of(acc, kind=kd), [], S, f"access {kd}"))
    P(edit_row("access, single-line edits only", [r for r in acc if r["n_changed"] == 1], [], S, "access single-line"))
    P("\n### Unified: gate A, sun x lion on the herd's destination (and so the pan and the well)\n" + EDIT_HDR)
    P(edit_row("carried cause: the reeds look removed", rows_of(dec, cell="unified_carried"), rows_of(plc, cell="unified_carried"), S, "unified carried", "lion_blind"))
    P(edit_row("in-view cause: a sun hour made cloud", rows_of(dec, cell="unified_view"), rows_of(plc, cell="unified_view"), S, "unified view", "sky_blind"))
    for st in ("herd", "water", "well"):
        P(edit_row(f"either cause, asked store {st}", [r for r in dec if r["cell"] in ("unified_carried", "unified_view") and r["a_store"] == st], [], S, f"unified {st}"))
    bl = [r for r in rows_of(dec, cell="both_lion") if r.get("paired")]; bs = {(r["day"], r["qi"]): r for r in rows_of(dec, cell="both_sun") if r.get("paired")}
    P(edit_row("both causes decide: the look removed", bl, rows_of(plc, cell="both_lion"), S, "both lion", "lion_blind"))
    P(edit_row("both causes decide: the sun hour made cloud", list(bs.values()), [], S, "both sun", "sky_blind"))
    if bl:
        for f in ("model",) + BLIND:
            v = 100 * np.mean([ok(r["f"][f][0], r["ideal0"]) and ok(r["f"][f][1], r["ideal1"]) and ok(bs[(r["day"], r["qi"])]["f"][f][1], bs[(r["day"], r["qi"])]["ideal1"]) for r in bl])
            S["both all three" + ("" if f == "model" else " " + f)] = v
        S["both all three blind"] = max(S["both all three line_blind"], S["both all three sky_blind"], S["both all three lion_blind"]); S["both all three ideal"] = 100.0
        P(f"- both causes, right on all three days (original, look removed, sun made cloud): model **{S['both all three']:.0f}%** of {len(bl)}; "
          f"line-blind {S['both all three line_blind']:.0f}, sky-blind {S['both all three sky_blind']:.0f}, lion-blind {S['both all three lion_blind']:.0f}")
    P("\n### Gate B, herd on the plain x cloud on the grass (reported, not counted)\n" + EDIT_HDR)
    P(edit_row("carried cause: the listen removed", rows_of(dec, cell="gate_b_carried"), rows_of(plc, cell="gate_b_carried"), S, "gate B carried"))
    P(edit_row("in-view cause: a cloud hour made sun", rows_of(dec, cell="gate_b_view"), rows_of(plc, cell="gate_b_view"), S, "gate B view"))
    P("\n### Self-modelling\n" + EDIT_HDR)
    P(edit_row("own command: the act taken out of an act-and-go", rows_of(dec, cell="own"), rows_of(plc, cell="own"), S, "own"))
    eff, effp = rows_of(dec, cell="efficacy"), rows_of(plc, cell="efficacy")
    P(edit_row("efficacy: the line that decides whether the act works, removed", eff, effp, S, "efficacy"))
    for st in W.ACTS:
        P(edit_row(f"efficacy, {st}", rows_of(eff, sub=st), rows_of(effp, sub=st), S, f"efficacy {st}"))
    for kd in LINE_KINDS:
        P(edit_row(f"efficacy, removed line kind {kd}", rows_of(eff, kind=kd), [], S, f"efficacy line {kd}"))   # own key: "dig" is also an act
    # origin pairs
    def pair_ok(f):
        return 100 * np.mean([all(ok(f(days[d][1][qi]), days[d][1][qi]["ideal"]) for d, qi in pr) for pr in pairs]) if pairs else np.nan
    S["origin pairs"] = pair_ok(lambda r: probs0[r["day"]][qix[(r["day"], r["qi"])]])
    S["origin pairs blind"] = pair_ok(lambda r: reader(r, "no-rule"))      # reads no line, so cannot tell who made the change
    P(f"\n- origin, matched pairs (an own-made and a herd-made identical printed change whose futures differ; same store, last word, hours "
      f"unseen and horizon): {len(pairs)} pairs; right on both members: model **{S['origin pairs']:.0f}%**, ideal 100, "
      f"no-rule reader {pair_ok(lambda r: reader(r, 'no-rule')):.0f}, sky-blind {pair_ok(lambda r: r['sky_blind']):.0f}, lion-blind {pair_ok(lambda r: r['lion_blind']):.0f}")
    # ---- metacognition
    P("\n## Metacognition\n")
    P("### Staleness: spread (entropy, nats) of the herd forecast one hour ahead, by hours since the last listen and the sky since\n")
    P("Cells: last heard on the plain, lion known, the sky the same from then to the asking hour (the herd leaves the plain only under "
      "sun, so under cloud the ideal is certain); mirror: last heard at the river (it leaves water only under cloud). The sky-blind "
      "forecaster has no gap by construction. 'model gap' = sun minus cloud.\n")
    P("| cell | age | n sun | ideal sun | ideal cloud | sky-blind sun | sky-blind cloud | model sun | model cloud | model gap | ideal gap |\n|---|---|---|---|---|---|---|---|---|---|---|")
    hq = [i for i, r in enumerate(rows) if r["store"] == "herd" and r["k"] == 1 and r["stale"]["lion"] == "known"]
    sky_cell = {i: (rows[i]["stale"]["sky"] if days[rows[i]["day"]][0][0].script["sky"][rows[i]["asked"]] == rows[i]["stale"]["sky"] else "mixed") for i in hq}
    for nm, lastf in (("plain", lambda s: s == "plain"), ("river", lambda s: s in ("pan", "bank"))):
        for age in (1, 2, 3):
            e = {}
            for sky in ("sun", "cloud"):
                idx = [i for i in hq if lastf(rows[i]["stale"]["last"]) and rows[i]["stale"]["age"] == age and sky_cell[i] == sky]
                e[sky] = (len(idx), np.mean([ent(rows[i]["ideal"]) for i in idx]) if idx else np.nan,
                          np.mean([ent(rows[i]["sky_blind"]) for i in idx if rows[i]["sky_blind"] is not None]) if idx else np.nan,
                          np.mean([ent(M[i]) for i in idx]) if idx else np.nan)
            mg, ig = e["sun"][3] - e["cloud"][3], e["sun"][1] - e["cloud"][1]
            if nm == "plain" and age == 2:
                S["stale gap (nats)"], S["stale gap ideal"], S["stale gap blind"] = mg, ig, e["sun"][2] - e["cloud"][2]
            P(f"| last heard {nm} | {age} | {e['sun'][0]} / {e['cloud'][0]} | {e['sun'][1]:.3f} | {e['cloud'][1]:.3f} | {e['sun'][2]:.3f} | {e['cloud'][2]:.3f} | "
              f"{e['sun'][3]:.3f} | {e['cloud'][3]:.3f} | **{mg:+.3f}** | {ig:+.3f} |")
    P("\n### Look-only: the reeds look removed, nothing else changed (information reading)\n" + EDIT_HDR)
    lk = rows_of(dec, cell="look")
    P(edit_row("the look decides (its removal flips the ideal's top answer)", rows_of(lk, flip=True), rows_of(plc, cell="look"), S, "look flip", "lion_blind"))
    P(edit_row("strict: and the lion was a near coin without it", [r for r in lk if r["flip"] and r["look"]["strict"]], [], S, "look strict", "lion_blind"))
    if lk:
        def dirn(f):
            top = [int(np.argmax(r["ideal0"])) for r in lk]
            di = [r["ideal1"][t] - r["ideal0"][t] for r, t in zip(lk, top)]
            dm = [r["f"][f][1][t] - r["f"][f][0][t] for r, t in zip(lk, top)]
            return 100 * np.mean([(a < -1e-9) == (b < -1e-9) for a, b in zip(di, dm)]), np.mean(dm), np.mean(di)
        S["look direction"], dm, di = dirn("model"); S["look direction blind"] = dirn("lion_blind")[0]
        P(f"- information reading over all {len(lk)} look questions (flip or 0.2 nats): the model's probability of the with-look answer moves in the "
          f"ideal's direction on **{S['look direction']:.0f}%** (mean move {dm:+.3f} against the ideal's {di:+.3f}); lion-blind "
          f"{dirn('lion_blind')[0]:.0f}%, sky-blind {dirn('sky_blind')[0]:.0f}%")
    # calibration
    P("\n### Calibration on chance questions\n")
    ch = [i for i, r in enumerate(rows) if r["chance"]]
    Mc = np.array([M[i] for i in ch]); Ic = np.array([rows[i]["ideal"] for i in ch]); Y = np.array([rows[i]["truth"] for i in ch])
    S["calibration TV"] = float(np.mean([tv(a, b) for a, b in zip(Mc, Ic)]))
    for w in ("sky_blind", "lion_blind"):
        S["calibration TV " + w] = float(np.mean([tv(rows[i][w], rows[i]["ideal"]) for i in ch if rows[i][w] is not None]))
    S["calibration TV blind"] = min(S["calibration TV sky_blind"], S["calibration TV lion_blind"])
    P(f"- {len(ch)} chance questions ({100 * len(ch) / len(rows):.1f}%): mean total-variation distance from the ideal's distribution: model "
      f"**{S['calibration TV']:.3f}**, sky-blind {S['calibration TV sky_blind']:.3f}, lion-blind {S['calibration TV lion_blind']:.3f}; "
      f"the model's top answer is the ideal's on {100 * np.mean([ok(M[i], rows[i]['ideal']) for i in ch]):.0f}%")
    P("\n| model says | n | mean said | ideal says | came true |\n|---|---|---|---|---|")
    yy = np.arange(3)[None, :] == Y[:, None]; valid = Ic > 0
    for a, b in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)):
        sel = (Mc >= a) & (Mc < b) & (valid | (Mc > 0))
        if sel.sum():
            P(f"| {a:.1f}-{min(b, 1):.1f} | {sel.sum()} | {Mc[sel].mean():.3f} | {Ic[sel].mean():.3f} | {yy[sel].mean():.3f} |")
    # transfer probe
    P("\n### Transfer probe: a linear read-out of 'this answer is wrong' from the hidden state, fitted on some stores, tested on others\n")
    if hs is not None:
        idx = [i for i, r in enumerate(rows) if r["day"] < A.probe_days]
        X = np.stack([hs[rows[i]["day"]][qix[(rows[i]["day"], rows[i]["qi"])]] for i in idx])
        Mi = np.array([M[i] for i in idx]); truth = np.array([rows[i]["truth"] for i in idx])
        wrong = Mi.argmax(1) != truth; iconf = np.array([1 - rows[i]["ideal"].max() for i in idx])
        grp = np.array([rows[i]["store"] if rows[i]["store"] not in W.ACTS else "acts" for i in idx])
        P("| fitted on | tested on | n test | wrong rate | read-out AUC | per test store | model's own confidence AUC | ideal's confidence AUC |\n|---|---|---|---|---|---|---|---|")
        for fit, test, n, wr, r_auc, per, c_auc, _ in probe(X, Mi, truth, grp):
            key = f"probe {'+'.join(fit)} -> {'+'.join(test)}"; S[key] = r_auc
            b = np.isin(grp, test)
            P(f"| {'+'.join(fit)} | {'+'.join(test)} | {n} | {100 * wr:.0f}% | **{r_auc:.3f}** | " + ", ".join(f"{g} {v:.2f}" for g, v in per.items())
              + f" | {c_auc:.3f} | {auc(iconf[b], wrong[b]):.3f} |")
    # ---- flags
    P("\n## Reading\n")
    flags = [k for k in S if k.endswith(" blind") and k != "stale gap blind" and S[k] > 85]
    P(f"- positive control (rain raises a store, out of view): **{S.get('control', np.nan):.0f}%** right on both days"
      + (" (learned: at or above 80)" if S.get("control", 0) >= 80 else " (under 80: the run has not learned the plain link; trait nulls are not informative)"))
    for k in ("access", "unified carried", "unified view", "own", "efficacy", "look flip"):
        if k in S and S[k] < 50:
            P(f"- {k}: under 50% right on both days; every condition built on this line kind is void (R8.4)")
    P("- unable to fail (a blind forecaster within 15 points of the ideal): " + (", ".join(k[:-6] for k in flags) if flags else "none"))
    P(f"- staleness: the model's sun-minus-cloud spread gap at age 2 is {S.get('stale gap (nats)', np.nan):+.3f} nats against the ideal's "
      f"{S.get('stale gap ideal', np.nan):+.3f} (sky-blind {S.get('stale gap blind', np.nan):+.3f}); the requirement's line is 0.2")
    P(f"\nDone in {(time.time() - T0) / 60:.1f} min.")
    out.close()
    pickle.dump(dict(S=S, recs=[{k: v for k, v in r.items() if k != "f"} | dict(model0=r["f"]["model"][0], model1=r["f"]["model"][1]) for r in recs],
                     probs0=probs0), open(os.path.join(A.out, f"{name}.pkl"), "wb"))
    return S


SUMMARY_ROWS = ["all questions", "gap (nats)", "changed: all", "changed: water", "changed: grass", "changed: well", "changed: herd",
                "changed: acts (drink, graze, dig)", "changed: steps 0-1", "changed: steps 2-3", "changed: steps 4+", "changed: unseen 1", "changed: unseen 2-3", "changed: unseen 4+",
                "control", "access", "access breadth", "access 1-10", "access 11-25", "access 26+", "access placebo moved",
                "unified carried", "unified view", "both all three", "unified herd", "unified water", "unified well", "gate B carried", "gate B view",
                "own", "efficacy", "efficacy drink", "efficacy graze", "efficacy dig", "origin pairs",
                "stale gap (nats)", "look flip", "look strict", "look direction", "calibration TV",
                "probe water+grass -> herd+well+acts", "probe herd+well -> water+grass", "probe water+grass+herd+well -> acts"]


def summary(results):
    names = list(results)
    def f(k, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{v:.3f}" if ("nats" in k or "TV" in k or k.startswith("probe")) else f"{v:.1f}"
    lines = ["# Summary across checkpoints", "", "Edit cells: % right on both days (ideal 100; 'blind' = the best blind forecaster). Yardstick cells: % agreement "
             "with the ideal on changed questions. Gaps in nats, AUC as a fraction.", "",
             "| cell | ideal | blind | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for k in SUMMARY_ROWS:
        S0 = next((S for S in results.values() if k in S), None)
        if S0 is None:
            continue
        ideal = S0.get(k + " ideal", 100.0 if k.startswith("changed") or k == "origin pairs" else S0.get("stale gap ideal") if k.startswith("stale") else None)
        blind = S0.get(k + " blind", S0.get("stale gap blind") if k.startswith("stale") else None)
        lines.append(f"| {k} | {f(k, ideal)} | {f(k, blind)} | " + " | ".join(f(k, S.get(k)) for S in results.values()) + " |")
    open(os.path.join(A.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- main

def checkpoints(args):
    out = []
    for a in args:
        if a == "base":
            out.append(("base@0", "base"))
        elif os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "weights_*.pt")), key=lambda p: int(p.split("_")[-1][:-3])):
                out.append((f"{os.path.basename(a.rstrip('/'))}@{p.split('_')[-1][:-3]}", p))
        else:
            u = a.split("_")[-1][:-3] if os.path.basename(a).startswith("weights_") else "ckpt"
            out.append((f"{os.path.basename(os.path.dirname(a))}@{u}", a))
    return out


if __name__ == "__main__":
    E = build_all()
    cuts = E["cuts"]
    results = {}
    if A.golden:
        # the ideal stands in for the model, computed from the same day and cut lists the model would read
        DEV = "none"
        probs0 = [np.array([pad3(W.ideal(L, qi)[0]) for qi in sorted(tags)]) for L, tags in E["days"]]
        for r in E["recs"]:
            r["model1"] = pad3(W.ideal(cuts[r["cut"]], r["qn"])[0])
        nflip = sum(not r["flip"] for r in E["recs"] if not r["placebo"] and r["cell"] != "look")
        print(f"golden: decisive edits whose replay did not reproduce the tagged flip: {nflip} of {sum(not r['placebo'] for r in E['recs'])}")
        results["ideal"] = report("golden_ideal", E, probs0, None)
    else:
        DEV = device()
        from train import encode
        enc_days, enc_cuts = [encode(L) for L, _ in E["days"]], [encode(c) for c in cuts]      # tokenised once
        print(f"tokenised ({time.time() - T0:.0f}s)", flush=True)
        for name, path in checkpoints(A.ckpts):
            t = time.time()
            m = load_model(path)
            probs0, hs = model_probs(m, enc_days, keep_hidden=A.probe_days)
            p1, _ = model_probs(m, enc_cuts)
            for r in E["recs"]:
                r["model1"] = p1[r["cut"]][-1]
            print(f"{name}: {len(E['days'])} days and {len(cuts)} edited days scored in {time.time() - t:.0f}s", flush=True)
            results[name] = report(name, E, probs0, hs)
            del m
    if len(results) > 1 or not A.golden:
        summary(results)
