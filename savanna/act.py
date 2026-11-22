"""The model as the actor: it writes each command line of a day, choosing among the commands the world allows at its
place. Days run in lockstep so one batched, cached forward serves all of them.

The policy is the model's next-token distribution renormalised, token by token, over the tokens that keep the line a
legal command (a trie per place). The same renormalised log-probabilities are what the training losses on command
tokens use (legal_logprobs), so sampling and training share one definition of the policy.
"""
import torch
from tokenizers import Tokenizer
from fast import Cache, hidden_step

import world_i1 as W


class Commands:
    """Token tries of the legal commands at each place."""
    def __init__(self, tok, legal=W.legal):
        self.tok = tok
        self.p0 = tok.encode("Command:").ids
        self.nl = tok.encode("\n").ids[0]
        self.seqs, self.trie = {}, {}
        for pl in W.PLACES:
            seqs = {}
            for a in legal(pl):
                t = tok.encode(W.render(W.Line("command", 0, pl, action=a))).ids
                assert t[:len(self.p0)] == self.p0, a
                seqs[tuple(t[len(self.p0):] + [self.nl])] = a
            trie = {}
            for s in seqs:
                for i in range(len(s)):
                    trie.setdefault(s[:i], set()).add(s[i])
            self.seqs[pl], self.trie[pl] = seqs, {k: sorted(v) for k, v in trie.items()}

    def allowed(self, place, prefix):
        return self.trie[place][tuple(prefix)]

    def tokens(self, place, action):
        return next(list(s) for s, a in self.seqs[place].items() if a == action)


def run_days(m, cmds, scripts, dev, gen=None, temperature=1.0, routine=False, explore=0.0, second=False):
    """Plays one day per script with the model as the actor (or the routine, routine=True, for a like-for-like day
    whose commands are still scored by the model). explore: the chance that a command is drawn evenly from the legal
    set instead (it favours no command; without it a command the model never writes is never tried). Returns, per
    day, the finished lines and a list of decision records: hour, place, action, whether it was an even draw, the
    model's log-probability of it, its probability of each first word, and (second=True) an independent second draw
    from the same policy at the same point (alt, alt_even), which the day does not follow."""
    tok = cmds.tok
    B = len(scripts)
    gens = [W.day(sc) for sc in scripts]
    reqs = [next(g) for g in gens]
    done_lines = [0] * B              # lines already fed
    ntok = [0] * B                    # tokens already fed (each day's own positions)
    cache = Cache(2048)               # written in place; a day with its command lines fits well under this (asserted)
    decisions = [[] for _ in range(B)]
    finished = [None] * B

    def feed(chunks):
        T = max(len(c) for c in chunks)
        ids = torch.zeros(B, T, dtype=torch.long); pos = torch.zeros(B, T, dtype=torch.long)
        val = torch.zeros(B, T, dtype=torch.bool)
        for b, c in enumerate(chunks):
            ids[b, :len(c)] = torch.tensor(c, dtype=torch.long)
            pos[b] = torch.arange(ntok[b], ntok[b] + T); val[b, :len(c)] = True
        h = hidden_step(m, ids.to(dev), pos.to(dev), val.to(dev), cache)
        last = torch.tensor([max(len(c), 1) - 1 for c in chunks], device=dev)
        for b, c in enumerate(chunks):
            ntok[b] += len(c)
        return m.head(h[torch.arange(B, device=dev), last]).float()

    def decode(logits, live, reqs, explore, routine):
        """One command per live day, token by token from the logits after 'Command:'; every token is fed to the
        cache. Returns per day (action, even draw, log-probability under the policy, first-word probabilities)."""
        prefix = [[] for _ in range(B)]
        active = list(live)
        first, logp, forced, even = [None] * B, [0.0] * B, [None] * B, [False] * B
        for b in range(B):
            if active[b]:
                lines, place, h, lev = reqs[b]
                if routine:
                    forced[b] = cmds.tokens(place, W.pick(scripts[b]["act_u"][h], W.policy(place, h, lev)))
                elif explore and float(torch.rand(1, generator=gen)) < explore:
                    opts = list(cmds.seqs[place])
                    forced[b] = list(opts[int(torch.randint(len(opts), (1,), generator=gen))])
                    even[b] = True
        while any(active):
            # the renormalised probabilities of every live day at once (allowed tokens padded, pads masked out), moved
            # to the CPU in one transfer; the draws then happen on the CPU, day by day, from the same generator
            rows = [b for b in range(B) if active[b]]
            allows = [cmds.allowed(reqs[b][1], prefix[b]) for b in rows]
            K = max(len(a) for a in allows)
            idx = torch.zeros(len(rows), K, dtype=torch.long); pad = torch.ones(len(rows), K, dtype=torch.bool)
            for i, a in enumerate(allows):
                idx[i, :len(a)] = torch.tensor(a, dtype=torch.long); pad[i, :len(a)] = False
            lg = logits[torch.tensor(rows, device=dev)[:, None], idx.to(dev)] / temperature
            probs = torch.softmax(lg.masked_fill(pad.to(dev), float("-inf")), -1).cpu()
            nxt = [[] for _ in range(B)]
            for i, b in enumerate(rows):
                allow, p = allows[i], probs[i, :len(allows[i])]
                if first[b] is None:
                    first[b] = {tok.decode([t]).strip(): float(q) for t, q in zip(allow, p)}
                if forced[b] is not None:
                    k = allow.index(forced[b][len(prefix[b])])
                else:
                    k = int(torch.multinomial(p, 1, generator=gen))
                logp[b] += float(torch.log(p[k]))
                prefix[b].append(allow[k])
                nxt[b] = [allow[k]]
                if allow[k] == cmds.nl:
                    active[b] = False
            out = feed(nxt)
            if any(active):
                logits = out
        return [(cmds.seqs[reqs[b][1]][tuple(prefix[b])], even[b], logp[b], first[b]) if live[b] else None
                for b in range(B)]

    with torch.no_grad():
        while any(f is None for f in finished):
            # every unfinished day is at a command point; feed its new lines and "Command:"
            chunks = []
            for b in range(B):
                if finished[b] is not None:
                    chunks.append([]); continue
                lines = reqs[b][0]
                new = []
                for t in W.render_episode(lines[done_lines[b]:]):
                    new += tok.encode(t).ids + [cmds.nl]
                done_lines[b] = len(lines)
                chunks.append(new + cmds.p0)
            logits = feed(chunks)
            live = [finished[b] is None for b in range(B)]
            alt = [None] * B
            if second:   # an independent second draw, decoded on the same prefix and then discarded: the cache's
                snap = (cache.n, list(ntok))   # filled length is wound back and the main draw overwrites its slots
                alt = decode(logits, live, reqs, explore, False)
                cache.n, ntok[:] = snap
            main = decode(logits, live, reqs, explore, routine)
            for b in range(B):
                if finished[b] is not None:
                    continue
                lines, place, h, lev = reqs[b]
                a, even, logp, first = main[b]
                decisions[b].append(dict(hour=h, place=place, action=a, even=even, logp=logp, first=first,
                                         alt=alt[b] and alt[b][0], alt_even=alt[b] and alt[b][1]))
                done_lines[b] += 1   # the command line, already fed token by token
                try:
                    reqs[b] = gens[b].send(a)
                except StopIteration as e:
                    finished[b] = e.value
                    continue
                # the day's generator appended the command line and this hour's questions; the command is fed
                assert reqs[b][0][done_lines[b] - 1].kind == "command"
    return finished, decisions


def legal_logprobs(logits, place, prefix_tokens, cmds):
    """Renormalised log-probability of each token of one command line given its logits [n, V] at the positions that
    predict the tokens after 'Command:' (the policy's own definition)."""
    out = []
    for i, t in enumerate(prefix_tokens):
        allow = cmds.allowed(place, prefix_tokens[:i])
        lg = logits[i, allow]
        out.append(torch.log_softmax(lg, -1)[allow.index(t)])
    return torch.stack(out)
