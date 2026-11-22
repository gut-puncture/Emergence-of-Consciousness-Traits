"""Iteration 1 of the savanna world (design.md, 24 October 2026): simulator, text, routine, chooser and the exact
ideal forecaster, in the interface the training code imports (simulate, render_episode, tag_question, LEVELS,
to_word, options).

One grazing animal; four places: the pan (a clay rain pool ringed by reeds, where a lion may lie), the bank (the
sand of the dry riverbed beside it, where a well can be dug), the plain (grass, over the rise) and the shade
(trees; nothing to read). The sky (sun, cloud, rain) prints on every line. One herd moves between the plain and the
water; a lion may lie in the reeds all day. The herd is never seen; "listen" prints whether it is on the plain or
down at the river (never which water). Rules, each one step a tick (h -> h+1 from the state at h, levels capped):
  sky:   a chain (SKY_NEXT).
  lion:  drawn at first light with chance 0.5, printed only by "look in the reeds".
  pan:   +1 after rain at h; -1 if the herd is at the pan at h or the animal drinks there (and can); rise against
         fall: no change.
  grass: +1 after rain; -1 if the herd is on the plain under cloud at h, -1 if the animal grazes (and can); the
         same rise-against-fall rule.
  well:  undug: dry. A dig at h (the animal's, or the herd standing at the bank) on a day with rain printed at or
         before h digs it: "well low" at h+1. Dug: +1 if nobody drinks at h; -1 if the herd is at the bank or the
         animal drinks there (and can).
  herd:  on the plain at h it walks to water with WALK[sky]: to the pan, or to the bank if a lion lies at the pan.
         At water it goes back to the plain with LEAVE[sky].
  dawn:  hour 5 prints the three places; the pan and grass levels drawn evenly, the well dry, the herd on the
         plain; the animal is at the pan at hour 6 (its dawn drink moves nothing).
The exact ideal is a belief filter over (herd place, lion, pan, grass, well, dug): 324 states.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

LEVELS = ("gone", "low", "high")
WELL_LEVELS = ("dry", "low", "high")
SKY = ("sun", "cloud", "rain")
SKY_NEXT = {"sun": {"sun": 0.80, "cloud": 0.15, "rain": 0.05},
            "cloud": {"sun": 0.35, "cloud": 0.40, "rain": 0.25},
            "rain": {"sun": 0.15, "cloud": 0.35, "rain": 0.50}}
SKY_DAWN = {"sun": 0.57, "cloud": 0.25, "rain": 0.18}
WALK = {"sun": 0.35, "cloud": 0.0, "rain": 0.0}    # herd: plain -> water (one trip a day, in the heat)
LEAVE = {"sun": 0.0, "cloud": 0.35, "rain": 0.35}  # herd: water -> plain (it moves off to graze when it cools)
P_LION = 0.5
P_REEDS = 0.7                                      # the reeds look at the pan at hour 6 (a habit of first light)
PLACES = ("shade", "pan", "plain", "bank")
HERD_PLACES = ("plain", "pan", "bank")             # herd state 0, 1, 2
STORE_PLACE = {"water": "pan", "grass": "plain", "well": "bank"}
PLACE_STORE = {"pan": "water", "plain": "grass", "bank": "well"}
STORES = ("water", "grass", "well", "herd")
ACTS = ("drink", "graze", "dig")
ACT_STORE = {"drink_pan": "water", "graze": "grass", "drink_bank": "well", "dig": "well"}
FIRST, LAST = 5, 22
# the routine, design.md §4: base weights per place (world3's); conditional acts drop when their condition fails
ROUTINE = {
    "shade": [(0.22, ("rest",)), (0.12, ("listen",)), (0.10, ("go", "pan")), (0.14, ("go_act", "pan", "drink")),
              (0.08, ("go", "plain")), (0.14, ("go_act", "plain", "graze")), (0.07, ("go", "bank")),
              (0.06, ("go_act", "bank", "drink")), (0.07, ("go_act", "bank", "dig"))],
    "pan": [(0.08, ("listen",)), (0.08, ("act", "drink")), (0.15, ("act_go", "drink")), (0.08, ("go", "shade")),
            (0.05, ("go", "plain")), (0.08, ("go_act", "plain", "graze")), (0.04, ("go", "bank")),
            (0.04, ("go_act", "bank", "drink")), (0.03, ("go_act", "bank", "dig")), (0.02, ("rest",))],
    "plain": [(0.10, ("act", "graze")), (0.16, ("act_go", "graze")), (0.08, ("listen",)), (0.07, ("go", "pan")),
              (0.14, ("go_act", "pan", "drink")), (0.12, ("go", "shade")), (0.05, ("go", "bank")),
              (0.05, ("go_act", "bank", "drink")), (0.05, ("go_act", "bank", "dig")), (0.18, ("rest",))],
    "bank": [(0.12, ("act", "dig")), (0.08, ("act_go", "dig")), (0.07, ("act", "drink")), (0.10, ("act_go", "drink")),
             (0.08, ("listen",)), (0.09, ("go", "pan")), (0.13, ("go_act", "pan", "drink")), (0.07, ("go", "plain")),
             (0.11, ("go_act", "plain", "graze")), (0.10, ("go", "shade")), (0.05, ("rest",))],
}
# the day's rhythm: the hour's place; moves toward it, every act-and-go, and staying acts there weigh TILT x more
BAND = {6: "plain", 7: "plain", 8: "plain", 9: "plain", 10: "shade", 11: "shade", 12: "shade", 13: "shade",
        14: "pan", 15: "pan", 16: "bank", 17: "bank", 18: "plain", 19: "plain", 20: "shade", 21: "shade"}
TILT = 3.0
# the chooser, design.md §5
Q_STORE_W = {"water": 1.0, "grass": 1.0, "well": 0.7, "herd": 0.8}
Q_HORIZON = {1: 0.9, 2: 0.1}
Q_EFF_W = 2.3
BLIND = set()   # forecasters blind to a rule, for the joint-margin check only: "sky_herd" (the herd walks, leaves
                # and eats at the long-run rates whatever the sky), "lion" (reeds looks are not read)

# ---------------------------------------------------------------- state space
N = 3 * 2 * 3 * 3 * 3 * 2
_i = np.arange(N)
DUG = _i % 2; WELL = (_i // 2) % 3; GRASS = (_i // 6) % 3; PAN = (_i // 18) % 3; LION = (_i // 54) % 2; HERD = _i // 108


def index(herd, lion, pan, grass, well, dug):
    return ((((herd * 2 + lion) * 3 + pan) * 3 + grass) * 3 + well) * 2 + dug


_TRANS = {}


def trans(rain, eat_gate, walk_p, leave_p, rained, eff):
    """One tick as (successor index [N,2], probability [N,2]). rain: rain printed at h; eat_gate: the herd on the
    plain eats this tick (cloud); walk_p / leave_p: the herd's move chances; rained: rain printed at or before h
    (wet sand); eff: the animal's act this tick, None | drink_pan | graze | drink_bank | dig."""
    key = (rain, eat_gate, walk_p, leave_p, rained, eff)
    if key in _TRANS:
        return _TRANS[key]
    herd_pan, herd_plain, herd_bank = HERD == 1, HERD == 0, HERD == 2
    fall = herd_pan | ((eff == "drink_pan") & (PAN > 0))
    pan = np.where(rain & fall, PAN, np.where(rain, PAN + 1, np.where(fall, PAN - 1, PAN)))
    fall = (herd_plain & eat_gate) | ((eff == "graze") & (GRASS > 0))
    grass = np.where(rain & fall, GRASS, np.where(rain, GRASS + 1, np.where(fall, GRASS - 1, GRASS)))
    dig = (DUG == 0) & rained & ((eff == "dig") | herd_bank)
    drink = herd_bank | ((eff == "drink_bank") & (WELL > 0))
    well = np.where(DUG == 1, np.where(drink, WELL - 1, WELL + 1), np.where(dig, 1, 0))
    dug = np.where(dig, 1, DUG)
    pan, grass, well = np.clip(pan, 0, 2), np.clip(grass, 0, 2), np.clip(well, 0, 2)
    stay = index(HERD, LION, pan, grass, well, dug)
    moved = index(np.where(herd_plain, 1 + LION, 0), LION, pan, grass, well, dug)
    p_move = np.where(herd_plain, walk_p, leave_p)
    out = (np.stack([stay, moved], 1), np.stack([1 - p_move, p_move], 1))
    _TRANS[key] = out
    return out


def step(b, t):
    idx, prob = t
    return np.bincount(idx.ravel(), weights=(b[:, None] * prob).ravel(), minlength=N)


def step_sky(b, sky, rained, eff):
    """One tick under a printed sky (or, for a sky-blind forecaster, the long-run rates)."""
    rain = sky == "rain"
    if "sky_herd" in BLIND:
        wk = sum(SKY_DAWN[k] * WALK[k] for k in SKY); lv = sum(SKY_DAWN[k] * LEAVE[k] for k in SKY)
        return (SKY_DAWN["cloud"] * step(b, trans(rain, True, wk, lv, rained, eff))
                + (1 - SKY_DAWN["cloud"]) * step(b, trans(rain, False, wk, lv, rained, eff)))
    return step(b, trans(rain, sky == "cloud", WALK[sky], LEAVE[sky], rained, eff))


# ---------------------------------------------------------------- the routine and the chooser

def policy(place, hour, level):
    """[(probability, action)] from the place, the hour and this line's level word (an int, or None in the shade).
    Actions: ('rest',) ('listen',) ('look_reeds',) ('go', P) ('go_act', P, act) ('act_go', act, P) ('act', act)."""
    acts = []
    for w, a in ROUTINE[place]:
        if a[0] in ("act", "act_go"):
            if a[1] == "dig" and level != 0:
                continue
            if a[1] != "dig" and level == 0:
                continue
        acts.append((w, a))
    tgt = BAND.get(hour)
    acts = [(w * (TILT if ((a[0] in ("go", "go_act") and a[1] == tgt) or a[0] == "act_go"
                          or (place == tgt and a[0] in ("rest", "act", "listen"))) else 1.0), a) for w, a in acts]
    tot = sum(w for w, _ in acts)
    out = []
    for w, a in acts:
        if a[0] == "act_go":
            for p in PLACES:
                if p != place:
                    out.append((w / tot / 3, ("act_go", a[1], p)))
        else:
            out.append((w / tot, a))
    if place == "pan" and hour == FIRST + 1:
        out = [(p * (1 - P_REEDS), a) for p, a in out] + [(P_REEDS, ("look_reeds",))]
    return out


def legal(place):
    """Every command the world allows at a place: the routine's acts there, whatever the level (a drink at a gone pan
    is allowed and fails), each act-and-go to every other place, and at the pan the look in the reeds."""
    out = []
    for _, a in ROUTINE[place]:
        if a[0] == "act_go":
            out += [("act_go", a[1], p) for p in PLACES if p != place]
        else:
            out.append(a)
    if place == "pan":
        out.append(("look_reeds",))
    return out


def pick(u, dist):
    acc = 0.0
    for p, a in dist:
        acc += p
        if u < acc:
            return a
    return dist[-1][1]


def effect(action, place):
    """(effect kind, place) of an act on the world, or (None, None)."""
    if action[0] == "go_act":
        p, a = action[1], action[2]
    elif action[0] in ("act", "act_go"):
        p, a = place, action[1]
    else:
        return None, None
    return {"drink": {"pan": "drink_pan", "bank": "drink_bank"}.get(p), "graze": "graze" if p == "plain" else None,
            "dig": "dig" if p == "bank" else None}[a], p


def next_place(action, place):
    if action[0] in ("go", "go_act"):
        return action[1]
    if action[0] == "act_go":
        return action[2]
    return place


def asks_act(cmd):
    """The command kinds after which 'the act at h+1?' is a candidate question: go-and-act and any dig."""
    return cmd is not None and (cmd[0] == "go_act" or (cmd[0] in ("act", "act_go") and cmd[1] == "dig"))


def candidates(h, place, cmd, last_word, last_seen):
    """The chooser's candidate questions and weights at hour h, from the printed record only."""
    out = []
    for st in STORES:
        w = Q_STORE_W[st]
        if st != "herd" and STORE_PLACE[st] == place:
            w *= 0.5
        if last_word[st] in ("gone", "dry"):
            w *= 0.5
        if h - last_seen[st] > 3:
            w *= 0.5
        for k in ((1,) if h == LAST - 1 else (1, 2)):
            out.append((w * (1.0 if h == LAST - 1 else Q_HORIZON[k]), (st, h + k)))
    if asks_act(cmd):
        out.append((Q_EFF_W, (cmd[2] if cmd[0] == "go_act" else cmd[1], h + 1)))
    return out


# ---------------------------------------------------------------- the day, as a script of chance draws

@dataclass
class Line:
    kind: str                 # morning | hour | command | question
    hour: int
    place: str = ""
    sky: str = ""
    level: int = -1           # the store level at a store place
    heard: str = ""           # a listen's result on an hour line: "plain" | "river"
    lion: str = ""            # a reeds look's result: "lion" | "no lion"
    outcome: str = ""         # drinks | no drink | grazes | no graze | digs | no dig
    action: tuple | None = None
    q: dict | None = None     # store, module (= store), T, asked, options, truth (index into options)
    script: dict | None = None
    truth: dict | None = None


ANSWERS = {"water": LEVELS, "grass": LEVELS, "well": WELL_LEVELS, "herd": ("pan", "plain", "bank"),
           "drink": ("yes", "no"), "graze": ("yes", "no"), "dig": ("yes", "no")}
HERD_ANSWER = (1, 0, 2)        # herd state -> index in the answer order (pan, plain, bank)


def draw_script(rng: random.Random, q_rate=0.9):
    sc = dict(pan=rng.randrange(3), grass=rng.randrange(3), lion=int(rng.random() < P_LION),
              sky={}, herd_u={}, act_u={}, q_u={})
    sky = rng.choices(list(SKY_DAWN), weights=list(SKY_DAWN.values()))[0]
    for h in range(FIRST, LAST + 1):
        if h > FIRST:
            sky = rng.choices(list(SKY_NEXT[sky]), weights=list(SKY_NEXT[sky].values()))[0]
        sc["sky"][h] = sky
        sc["herd_u"][h] = rng.random()
        sc["act_u"][h] = rng.random()
        sc["q_u"][h] = tuple((rng.random() < q_rate, rng.random()) for _ in range(2))
    return sc


def replay(sc, forced_qs=None, forced_cmds=None):
    """One day from a script. forced_qs: {hour: [(store, T)]} asks exactly those questions; forced_cmds: {hour:
    action} fixes the animal's commands (a counterfactual day keeps the animal's behaviour; any command is legal
    at any hour, a drink at a dry well simply fails). Otherwise the routine chooses."""
    g = day(sc, forced_qs)
    try:
        lines, place, h, lev = next(g)
        while True:
            act = forced_cmds[h] if forced_cmds else pick(sc["act_u"][h], policy(place, h, lev))
            lines, place, h, lev = g.send(act)
    except StopIteration as e:
        return e.value


def day(sc, forced_qs=None):
    """The day as a generator: at each command point it yields (lines so far, place, hour, the level printed here
    or None) and is sent the command; it returns the finished lines. Whoever sends the commands is the actor."""
    st = dict(herd=0, lion=sc["lion"], pan=sc["pan"], grass=sc["grass"], well=0, dug=0)
    sky = sc["sky"][FIRST]
    rained = sky == "rain"
    lines = [Line("morning", FIRST, "pan", sky, level=st["pan"]),
             Line("morning", FIRST, "plain", sky, level=st["grass"]),
             Line("morning", FIRST, "bank", sky, level=st["well"])]
    hist = {FIRST: dict(st)}
    last_seen = {s: FIRST for s in STORES}
    last_word = dict(water=LEVELS[st["pan"]], grass=LEVELS[st["grass"]], well="dry", herd="plain")
    place, act = "pan", None
    for h in range(FIRST + 1, LAST + 1):
        eff, ep = effect(act, place) if act else (None, None)
        outcome = ""
        if asks_act(act):   # the act at h-1, judged on the state at h-1
            kind = act[2] if act[0] == "go_act" else act[1]
            lev = {"pan": st["pan"], "plain": st["grass"], "bank": st["well"]}[ep]
            ok = (rained and not st["dug"]) if kind == "dig" else lev > 0
            outcome = {"drink": ("drinks", "no drink"), "graze": ("grazes", "no graze"), "dig": ("digs", "no dig")}[kind][0 if ok else 1]
        # the tick h-1 -> h
        rain = sky == "rain"
        fall = st["herd"] == 1 or (eff == "drink_pan" and st["pan"] > 0)
        pan = st["pan"] if (rain and fall) else st["pan"] + 1 if rain else st["pan"] - 1 if fall else st["pan"]
        fall = (st["herd"] == 0 and sky == "cloud") or (eff == "graze" and st["grass"] > 0)
        grass = st["grass"] if (rain and fall) else st["grass"] + 1 if rain else st["grass"] - 1 if fall else st["grass"]
        if st["dug"]:
            drink = st["herd"] == 2 or (eff == "drink_bank" and st["well"] > 0)
            well, dug = st["well"] - 1 if drink else st["well"] + 1, 1
        elif rained and (eff == "dig" or st["herd"] == 2):
            well, dug = 1, 1
        else:
            well, dug = 0, 0
        p_move = WALK[sky] if st["herd"] == 0 else LEAVE[sky]
        herd = st["herd"]
        if sc["herd_u"][h - 1] < p_move:
            herd = (1 + st["lion"]) if st["herd"] == 0 else 0
        st = dict(herd=herd, lion=st["lion"], pan=min(2, max(0, pan)), grass=min(2, max(0, grass)),
                  well=min(2, max(0, well)), dug=dug)
        if act is not None:
            place = next_place(act, place)
        sky = sc["sky"][h]
        rained = rained or sky == "rain"
        hist[h] = dict(st)
        ln = Line("hour", h, place, sky, outcome=outcome)
        if place != "shade":
            ln.level = {"pan": st["pan"], "plain": st["grass"], "bank": st["well"]}[place]
            last_seen[PLACE_STORE[place]] = h
            last_word[PLACE_STORE[place]] = (WELL_LEVELS if place == "bank" else LEVELS)[ln.level]
        if act == ("listen",):
            ln.heard = "plain" if st["herd"] == 0 else "river"
            last_seen["herd"], last_word["herd"] = h, ln.heard
        if act == ("look_reeds",):
            ln.lion = "lion" if st["lion"] else "no lion"
        lines.append(ln)
        act = None
        if h < LAST:
            act = yield lines, place, h, (ln.level if place != "shade" else None)
            lines.append(Line("command", h, place, action=act))
            cands = candidates(h, place, act, last_word, last_seen)
            if forced_qs is not None:
                for st_, T in forced_qs.get(h, []):
                    lines.append(question(h, place, st_, T))
                continue
            wts = [w for w, _ in cands]
            for ask, u in sc["q_u"][h]:
                if ask and sum(wts) > 0:
                    acc, j = 0.0, len(cands) - 1
                    for i, w in enumerate(wts):
                        acc += w / sum(wts)
                        if u < acc:
                            j = i; break
                    wts[j] = 0.0
                    lines.append(question(h, place, cands[j][1][0], cands[j][1][1]))
    for ln in lines:
        if ln.kind == "question":
            ln.q["truth"] = truth_of(ln.q, hist, lines)
    lines[0].script, lines[0].truth = sc, hist
    return lines


def question(h, place, store, T):
    return Line("question", h, place, q=dict(store=store, module=store, T=T, asked=h, options=ANSWERS[store]))


def truth_of(q, hist, lines):
    s, T = q["store"], q["T"]
    if s in ("water", "grass", "well"):
        return hist[T][{"water": "pan", "grass": "grass", "well": "well"}[s]]
    if s == "herd":
        return HERD_ANSWER[hist[T]["herd"]]
    ln = next(l for l in lines if l.kind == "hour" and l.hour == T)
    return 0 if ln.outcome in ("drinks", "grazes", "digs") else 1


def questions_of(lines):
    out = {}
    for ln in lines:
        if ln.kind == "question":
            out.setdefault(ln.hour, []).append((ln.q["store"], ln.q["T"]))
    return out


def simulate(rng: random.Random, q_rate=0.9):
    return replay(draw_script(rng, q_rate))


# ---------------------------------------------------------------- text and the training interface

def render(ln: Line) -> str:
    if ln.kind in ("morning", "hour"):
        parts = [f"Hour {ln.hour}.", f"{ln.place.capitalize()}.", f"{ln.sky}."]
        if ln.place != "shade":
            parts.append(f"{PLACE_STORE[ln.place]} {(WELL_LEVELS if ln.place == 'bank' else LEVELS)[ln.level]}.")
        if ln.outcome:
            parts.append(f"{ln.outcome}.")
        if ln.heard:
            parts.append("herd on the plain." if ln.heard == "plain" else "herd at the river.")
        if ln.lion:
            parts.append(f"{ln.lion} in the reeds.")
        return " ".join(parts)
    if ln.kind == "command":
        a = ln.action
        if a[0] == "rest": return "Command: rest."
        if a[0] == "listen": return "Command: listen."
        if a[0] == "look_reeds": return "Command: look in the reeds."
        if a[0] == "go": return f"Command: go to the {a[1]}."
        if a[0] == "go_act": return f"Command: go to the {a[1]} and {a[2]}."
        if a[0] == "act_go": return f"Command: {a[1]} and go to the {a[2]}."
        return f"Command: {a[1]}."
    if ln.kind == "question":
        s, T = ln.q["store"], ln.q["T"]
        name = "the water at the pan" if s == "water" else f"the {s}"
        a = ln.q["options"]
        opts = f"{a[0]} or {a[1]}" if len(a) == 2 else f"{a[0]}, {a[1]} or {a[2]}"
        return f"Question: {name} at hour {T}? {opts}. Answer:"
    raise ValueError(ln.kind)


def render_episode(lines):
    return [render(ln) for ln in lines]


def options(ln: Line):
    """The answer words of a question line, in the order the question lists them."""
    return ln.q["options"]


def to_word(module, truth):
    """The true answer word of a question (module = the asked store or act; truth = its index)."""
    return ANSWERS[module][truth]


def tag_question(lines, qi):
    """Tags for one question (tagger.py), with truth_w = the index of the true option word."""
    import tagger
    return tagger.tag_question(lines, qi)


# ---------------------------------------------------------------- the ideal, from the printed record only

def read_record(lines, qi, drop=()):
    """The printed facts up to the question at index qi. drop: line indices whose fact is removed from the record:
    a look command (and its result on the next hour line) or that hour line's look result."""
    R = dict(sky={}, place={}, lev={}, heard={}, lion={}, act={}, outcome={}, dawn={})
    dropped_cmd = set()
    for i in range(qi):
        ln = lines[i]
        if ln.kind == "morning":
            R["sky"][FIRST] = ln.sky
            R["dawn"][ln.place] = ln.level
        elif ln.kind == "hour":
            h = ln.hour
            R["sky"][h], R["place"][h] = ln.sky, ln.place
            if ln.place != "shade":
                R["lev"][h] = ln.level
            if ln.heard and i not in drop and (h - 1) not in dropped_cmd:
                R["heard"][h] = ln.heard == "plain"
            if ln.lion and (h - 1) not in dropped_cmd and "lion" not in BLIND:
                R["lion"][h] = ln.lion == "lion"
            if ln.outcome:
                R["outcome"][h] = ln.outcome
        elif ln.kind == "command":
            if i in drop:
                dropped_cmd.add(ln.hour)
                continue
            R["act"][ln.hour] = (ln.action, ln.place)
    return R


def rained_by(R, h):
    return any(R["sky"].get(u) == "rain" for u in range(FIRST, h + 1))


def outcome_lik(R, h):
    """Likelihood over states at h-1 of the outcome word printed at h (the act at h-1 judged on the state at h-1)."""
    oc = R["outcome"].get(h)
    if not oc:
        return None
    act, pl = R["act"][h - 1]
    eff, ep = effect(act, pl)
    ok = oc in ("drinks", "grazes", "digs")
    if eff == "dig":
        if not rained_by(R, h - 1):
            return None
        return (DUG == 0) == ok
    lev = {"drink_pan": PAN, "graze": GRASS, "drink_bank": WELL}[eff]
    return (lev > 0) == ok


def line_lik(R, h):
    lik = np.ones(N)
    pl = R["place"][h]
    if pl != "shade":
        lik = lik * ({"pan": PAN, "plain": GRASS, "bank": WELL}[pl] == R["lev"][h])
    if h in R["heard"]:
        lik = lik * ((HERD == 0) if R["heard"][h] else (HERD != 0))
    if h in R["lion"]:
        lik = lik * (LION == int(R["lion"][h]))
    return lik


def filter_to(R, t):
    """Beliefs by hour given the record up to hour t: {h: belief}, or None if the record is impossible."""
    b = np.zeros(N)
    for lion in (0, 1):
        b[index(0, lion, R["dawn"]["pan"], R["dawn"]["plain"], 0, 0)] = 0.5
    B = {FIRST: b}
    for h in range(FIRST + 1, t + 1):
        ol = outcome_lik(R, h)
        if ol is not None:
            b = b * ol
        act, pl = R["act"].get(h - 1, (None, None))
        eff = effect(act, pl)[0] if act else None
        b = step_sky(b, R["sky"][h - 1], rained_by(R, h - 1), eff)
        b = b * line_lik(R, h)
        s = b.sum()
        if s <= 0:
            return None
        b = b / s
        B[h] = b
    return B


def marginal(b, store):
    col = {"water": PAN, "grass": GRASS, "well": WELL, "herd": HERD}[store]
    m = np.bincount(col, weights=b, minlength=3)
    return m[[1, 0, 2]] if store == "herd" else m      # herd answers in the order pan, plain, bank


_EFF = {}


def effect_probs(place1, hour1):
    """Per state at t+1: the chance that the routine's act there has each effect on the world."""
    key = (place1, hour1)
    if key not in _EFF:
        P = {e: np.zeros(N) for e in (None, "drink_pan", "graze", "drink_bank", "dig")}
        for s in range(N):
            lev = {"pan": PAN[s], "plain": GRASS[s], "bank": WELL[s]}.get(place1)
            for p, a in policy(place1, hour1, lev):
                P[effect(a, place1)[0]][s] += p
        _EFF[key] = P
    return _EFF[key]


def forecast(R, B, t, store, T):
    """The answer distribution for the question (store, T) asked at t, after hour t's line and command."""
    act, pl = R["act"].get(t, (None, None))
    b_t = B[t]
    if store in ACTS:
        eff, ep = effect(act, pl)
        if eff == "dig":
            p = float((b_t * (DUG == 0)).sum()) if rained_by(R, t) else 0.0
        else:
            p = float((b_t * ({"drink_pan": PAN, "graze": GRASS, "drink_bank": WELL}[eff] > 0)).sum())
        return np.array([p, 1 - p])
    eff = effect(act, pl)[0] if act else None
    rained = rained_by(R, t)
    b1 = step_sky(b_t, R["sky"][t], rained, eff)
    if T == t + 1:
        return marginal(b1, store)
    place1 = next_place(act, pl) if act else pl
    P = effect_probs(place1, t + 1)
    out = np.zeros(3)
    for sky1, ps in SKY_NEXT[R["sky"][t]].items():
        rained1 = rained or sky1 == "rain"
        b2 = sum(step_sky(b1 * P[e], sky1, rained1, e) for e in P)
        out += ps * marginal(b2, store)
    return out


def ideal(lines, qi, drop=()):
    """Exact answer distribution for the question at index qi from the lines before it. None if impossible."""
    q = lines[qi].q
    R = read_record(lines, qi, drop)
    B = filter_to(R, q["asked"])
    if B is None:
        return None
    d = np.clip(forecast(R, B, q["asked"], q["store"], q["T"]), 0, None)
    return d / d.sum(), dict(B=B, R=R)


def argmax_set(d):
    mx = max(d)
    return {i for i, v in enumerate(d) if v >= mx - 1e-9}


def entropy(d):
    d = np.asarray(d); d = d[d > 0]
    return float(-(d * np.log(d)).sum())
