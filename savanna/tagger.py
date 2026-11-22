"""Tags for every question of a day, from the printed record and the rules only (world4's pre-registered
definitions, predictions.md §3, with the listen in the dust look's place).

A deciding line is a printed cause whose removal changes the ideal's top answer. Removals: a rain line becomes
cloud and the day is replayed; a sky colour is substituted (sun <-> cloud) and replayed; an own act line becomes
its bare move (or rest) and the day is replayed; a listen or a reeds look is dropped from the record (the truth
unchanged). The herd's "last printed word": "plain" at dawn (rule 11) or after "herd on the plain."; after "herd at
the river." it is taken as "pan" (the lion is a fair coin, so this equals a half-weight copier in expectation:
such questions enter the changed column on lion days, when the herd is at the bank). Each tag dict also carries
what the measure script needs, and truth_w (the index of the true option) for the training code."""
import copy
import numpy as np
import world_i1 as W

_HOURS = {}      # id(lines) -> {hour: index of its hour line (dawn: the first morning line)}


def hour_lines(lines):
    k = id(lines)
    if k not in _HOURS or _HOURS[k][1] is not lines:
        H = {}
        for i, ln in enumerate(lines):
            if ln.kind == "hour" or (ln.kind == "morning" and ln.hour not in H):
                H[ln.hour] = i
        _HOURS.clear(); _HOURS[k] = (H, lines)
    return _HOURS[k][0]


def kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    m = p > 0
    return float((p[m] * np.log(p[m] / np.maximum(q[m], 1e-12))).sum())


def logp(d, truth):
    return float(np.log(max(d[truth], 1e-12)))


def last_sight(lines, qi, store):
    """(hour, printed word index) of the asked store's last print before the question."""
    s, w = W.FIRST, None
    for i in range(qi):
        ln = lines[i]
        if ln.kind not in ("morning", "hour"):
            continue
        if store == "herd":
            if ln.heard:
                s, w = ln.hour, 1 if ln.heard == "plain" else 0
        elif ln.place == W.STORE_PLACE[store]:
            s, w = ln.hour, ln.level
    if store == "herd" and w is None:
        w = 1
    return s, w


def replay_ideal(sc_edit, lines, qi, cmds):
    """The ideal for the same question on the edited day (same questions, same commands)."""
    q = lines[qi].q
    new = W.replay(sc_edit, forced_qs=W.questions_of(lines), forced_cmds=cmds)
    qn = next(i for i, ln in enumerate(new) if ln.kind == "question" and ln.hour == q["asked"]
              and ln.q["store"] == q["store"] and ln.q["T"] == q["T"])
    r = W.ideal(new, qn)
    return None if r is None else r[0]


def tag_question(lines, qi):
    HOUR_LINE = hour_lines(lines)
    sc, hist = lines[0].script, lines[0].truth
    q = lines[qi].q
    store, T, t = q["store"], q["T"], q["asked"]
    is_act = store in W.ACTS
    cmd_t = next(l for l in lines[:qi][::-1] if l.kind == "command")
    if is_act:
        eff, ep = W.effect(cmd_t.action, cmd_t.place)
        a_store = W.ACT_STORE[eff]
    else:
        a_store = store
    d0, info = W.ideal(lines, qi)
    B, R = info["B"], info["R"]
    base = W.argmax_set(d0)
    s, last_w = last_sight(lines, qi, a_store)
    cur_start = HOUR_LINE[t]
    end = t if is_act else T            # the last hour whose state the question is about
    path = {h: hist[h]["herd"] for h in hist}
    key = {"water": "pan", "grass": "grass", "well": "well", "herd": "herd"}[a_store]
    hidden_hours = max(0, end - s - (0 if is_act else 1))
    hidden_steps = sum(hist[h + 1][key] != hist[h][key] for h in range(s, end))
    # ---- candidates: (kind, form, reveal index, op)
    cands = []
    for h in range(s, t + 1):
        if sc["sky"][h] == "rain":
            cands.append(("rain", "", HOUR_LINE[h], ("sky", h, "cloud")))
        elif h == t or path[h] == 0:
            cands.append(("colour", sc["sky"][h], HOUR_LINE[h], ("sky", h, "cloud" if sc["sky"][h] == "sun" else "sun")))
    listens = [i for i in range(qi) if lines[i].kind == "hour" and lines[i].heard]
    for i in listens:
        cands.append(("herd", lines[i].heard, i, ("drop", i)))
    if a_store == "grass" and not listens:
        h1 = next((h for h in range(W.FIRST, t) if W.WALK[sc["sky"][h]] > 0), None)
        if h1 is not None:
            cands.append(("herd_dawn", "plain", 1, ("herd_u", h1, 0.0)))
    for i in range(qi):
        ln = lines[i]
        if ln.kind == "command" and ln.action == ("look_reeds",) and HOUR_LINE.get(ln.hour + 1, qi) < qi:
            cands.append(("lion", "", HOUR_LINE[ln.hour + 1], ("drop", i)))
        if ln.kind == "command" and ln.action[0] in ("act", "act_go", "go_act") and not (is_act and ln.hour == t):
            e, p = W.effect(ln.action, ln.place)
            if e is None or W.ACT_STORE[e] != a_store:
                continue
            if e != "dig" and ln.hour < s:
                continue
            rep = ("rest",) if ln.action[0] == "act" else ("go", ln.action[1] if ln.action[0] == "go_act" else ln.action[2])
            cands.append(("dig" if e == "dig" else "act", ln.action[0], i, ("cmd", ln.hour, rep)))
    # ---- test each
    decided, alt = [], {}
    cmds = {ln.hour: ln.action for ln in lines if ln.kind == "command"}
    for kind, form, reveal, op in cands:
        if op[0] == "drop":
            r = W.ideal(lines, qi, drop={op[1]})
            d1 = None if r is None else r[0]
            if kind == "lion" and r is not None:
                alt["lion"] = (d1, float((r[1]["B"][t] * W.LION).sum()))
        else:
            e, c = copy.deepcopy(sc), dict(cmds)
            if op[0] == "sky": e["sky"][op[1]] = op[2]
            elif op[0] == "herd_u": e["herd_u"][op[1]] = op[2]
            elif op[0] == "cmd": c[op[1]] = op[2]
            d1 = replay_ideal(e, lines, qi, c)
        flip = d1 is not None and W.argmax_set(d1) != base
        if flip:
            decided.append((kind, form, reveal, qi - reveal, op))
    kinds = sorted({d[0] for d in decided})
    far = max((d[3] for d in decided), default=0)
    carried = [d for d in decided if d[2] < cur_start]
    # R1.2: a deciding line before the store's last print is excluded (the lion and a dig are lasting states: kept)
    access = [d for d in carried if lines[d[2]].hour >= s or d[0] in ("lion", "dig")]
    access_far = max((d[3] for d in access), default=0)
    # ---- gates: g1 = herd-on-plain x cloud on the grass (reported); g2 = sun x lion on the herd, pan, well (counted)
    g1 = g2 = None
    if a_store == "grass":
        eat_hours = [h for h in range(s, min(t, end - 1) + 1) if sc["sky"][h] == "cloud" and path[h] == 0 and hist[h]["grass"] > 0]
        carried_c = [c for c in cands if (c[0] == "herd" and c[1] == "plain") or c[0] == "herd_dawn"]
        if eat_hours and carried_c:
            cf = any(d[0] in ("herd", "herd_dawn") and d[1] == "plain" for d in decided)
            vf = any(d[0] == "colour" and lines[d[2]].hour in eat_hours for d in decided)
            g1 = dict(carried=cf, view=vf)
    if a_store in ("herd", "water", "well") and "lion" in {c[0] for c in cands}:
        walk_hours = [h for h in range(s, t + 1) if sc["sky"][h] == "sun" and path[h] == 0]
        if walk_hours:
            cf = any(d[0] == "lion" for d in decided)
            vf = any(d[0] == "colour" and lines[d[2]].hour in walk_hours for d in decided)
            g2 = dict(carried=cf, view=vf)
    margin = {}
    for g, name in ((g1, "g1"), (g2, "g2")):
        if g is None:
            continue
        lp0 = logp(d0, q["truth"])
        if name == "g2":
            d_c = alt["lion"][0] if "lion" in alt else None
        else:
            drops = {c[2] for c in cands if c[0] == "herd" and c[1] == "plain"}
            r = W.ideal(lines, qi, drop=drops) if drops else None
            d_c = None if r is None else r[0]
        W.BLIND.add("sky_herd")
        r = W.ideal(lines, qi)
        W.BLIND.discard("sky_herd")
        d_s = None if r is None else r[0]
        margin[name] = dict(lp=lp0, lp_c=logp(d_c, q["truth"]) if d_c is not None else None,
                            lp_s=logp(d_s, q["truth"]) if d_s is not None else None,
                            blind_c_ok=(W.argmax_set(d_c) == base) if d_c is not None else None,
                            blind_s_ok=(W.argmax_set(d_s) == base) if d_s is not None else None)
    # ---- cells (truth over the window and at the asking hour)
    win = range(s, min(t, max(s, end - 1)) + 1)
    lion_line = any(lines[i].lion for i in range(qi) if lines[i].kind == "hour")
    cells = dict(g1_win=(any(path[h] == 0 for h in win), any(sc["sky"][h] == "cloud" for h in win)),
                 g1_t=(path[t] == 0, sc["sky"][t] == "cloud"),
                 g2_win=(bool(sc["lion"]), any(sc["sky"][h] == "sun" and path[h] == 0 for h in win)),
                 g2_t=(bool(sc["lion"]), sc["sky"][t] == "sun"), lion_line=lion_line)
    # ---- joins on the asked store over the window (truth)
    cancel = clamp = False
    for h in range(s, end):
        rain = sc["sky"][h] == "rain"
        act, pl = R["act"].get(h, (None, None))
        e = W.effect(act, pl)[0] if act else None
        if a_store == "water":
            fall = path[h] == 1 or (e == "drink_pan" and hist[h]["pan"] > 0)
            cancel |= rain and fall
            clamp |= (rain and not fall and hist[h]["pan"] == 2) or (fall and not rain and hist[h]["pan"] == 0)
        elif a_store == "grass":
            fall = (path[h] == 0 and sc["sky"][h] == "cloud") or (e == "graze" and hist[h]["grass"] > 0)
            cancel |= rain and fall
            clamp |= (rain and not fall and hist[h]["grass"] == 2) or (fall and not rain and hist[h]["grass"] == 0)
        elif a_store == "well" and hist[h]["dug"]:
            drink = path[h] == 2 or (e == "drink_bank" and hist[h]["well"] > 0)
            clamp |= (drink and hist[h]["well"] == 0) or (not drink and hist[h]["well"] == 2)
    # ---- self: own command, origin, efficacy
    own_go = any(d[0] in ("act", "dig") and d[1] == "act_go" for d in decided)
    own_any = any(d[0] in ("act", "dig") for d in decided) and s < t
    world_dec = a_store != "herd" and any(d[0] == "herd" for d in decided)
    origin = "own" if own_go else "world" if (world_dec and not own_any) else ""
    # ---- look-only and stale
    look = None
    if "lion" in alt:
        d1, p_lion = alt["lion"]
        flip = W.argmax_set(d1) != base
        look = dict(flip=flip, p_lion_nolook=p_lion, kl=kl(d0, d1), strict=flip and abs(p_lion - 0.5) <= 0.15,
                    info=flip or kl(d0, d1) >= 0.2)
    marg_t = W.marginal(B[t], a_store)
    herd_t = W.marginal(B[t], "herd")
    # staleness cells: the herd belief at t by age since its last listen, sky over the unseen hours, lion knowledge
    sh, sw = last_sight(lines, qi, "herd")
    skies = {sc["sky"][h] for h in range(sh, t)}
    lion_known = any(lines[i].lion for i in range(qi) if lines[i].kind == "hour" and lines[i].hour <= t)
    stale = dict(age=t - sh, last=("pan", "plain", "bank")[sw], sky="sun" if skies == {"sun"} else "cloud" if skies == {"cloud"} else "mixed",
                 lion="known" if lion_known else "unknown", ent=W.entropy(herd_t), top=int(np.argmax(herd_t)))
    # ---- chance and look-resolvable
    chance = max(d0) < 1 - 1e-9
    resolvable, ent_left = None, None
    if chance:
        b = B[t] * (W.HERD == hist[t]["herd"]) * (W.LION == sc["lion"])
        if b.sum() > 0:
            B2 = dict(B); B2[t] = b / b.sum()
            d2 = W.forecast(R, B2, t, store, T)
            resolvable = max(d2) >= 1 - 1e-9
            ent_left = W.entropy(d2) / max(W.entropy(d0), 1e-9)
    if is_act:   # the copier's act answer: a drink or graze works unless the last word was gone or dry; a dig works on a dry well
        copier = (0 if last_w == 0 else 1) if store == "dig" else (1 if last_w == 0 else 0)
    else:
        copier = last_w
    return dict(store=store, module=store, a_store=a_store, k=T - t, asked=t, T=T, place=cmd_t.place, hidden=hidden_hours, steps=hidden_steps,
                truth=q["truth"], truth_w=q["truth"], ideal=[float(x) for x in d0], base=sorted(base), last_w=last_w, copier=copier,
                changed=copier not in base, chance=chance, resolvable=resolvable, ent_left=ent_left, ent=W.entropy(d0), ent_t=W.entropy(marg_t),
                decided=[(d[0], d[1], d[2], d[3]) for d in decided], kinds=kinds, far=far, cur_line=bool(decided) and not carried,
                carried=bool(carried), access=[(d[0], d[1], d[2], d[3]) for d in access], access_far=access_far,
                g1=g1, g2=g2, margin=margin, cells=cells, cancel=cancel, clamp=clamp, own_go=own_go, own_any=own_any,
                origin=origin, look=look, stale=stale, n_decisive=len(decided), lion_line=lion_line, cur_start=cur_start,
                lion_day=bool(sc["lion"]), outcome=(q["truth"] == 1) if is_act else None)


def look_values(lines, tags_by_qi):
    """Per look command: (kind, hour, value in nats summed over the day's later questions, settled count, n later)."""
    HOUR_LINE = hour_lines(lines)
    out = []
    for i, ln in enumerate(lines):
        if ln.kind == "command" and ln.action in (("listen",), ("look_reeds",)):
            res = HOUR_LINE.get(ln.hour + 1)
            if res is None:
                continue
            later = [j for j in tags_by_qi if j > res]
            val = settled = 0.0; n3 = settled3 = 0
            for j in later:
                r = W.ideal(lines, j, drop={i})
                if r is None:
                    continue
                d0 = np.array(tags_by_qi[j]["ideal"]); truth = tags_by_qi[j]["truth"]
                val += logp(d0, truth) - logp(r[0], truth)
                fl = W.argmax_set(r[0]) != set(tags_by_qi[j]["base"])
                settled += fl
                if tags_by_qi[j]["asked"] <= ln.hour + 3:
                    n3 += 1; settled3 += fl
            out.append(dict(kind=ln.action[0], hour=ln.hour, value=val, settled=settled, n=len(later), n3=n3, settled3=settled3))
    return out


def tag_day(lines):
    tags = {i: tag_question(lines, i) for i, ln in enumerate(lines) if ln.kind == "question"}
    return tags, look_values(lines, tags)
