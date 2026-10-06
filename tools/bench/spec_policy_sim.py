"""Offline verify-length policy simulation from logged rounds (spec campaign).

Input: a spec_lmbench JSON produced with a long fixed verify (e.g. block 15, L = 15) — every round's accepted count
and the block's confidences — plus a measured round-cost table (spec_cost_table JSON: full round ms per L).
For each policy it replays the logged rounds: a policy picks L from the round's confidences; the round then accepts
min(accepted, L) drafts (greedy acceptance of a prefix does not depend on how many drafts follow it) and costs
cost(L). Reports tokens per round, ms per round and the implied decode tok/s per prompt. Rounds after a different
choice would differ in reality (other anchors); this is the usual first-order estimate, confirmed end to end after.
"""
import argparse
import json
import math
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True, help='spec_lmbench JSONs (fixed long verify)')
ap.add_argument('--cost', required=True, help='spec_cost_table JSON (full round ms per L)')
ap.add_argument('--sts', default=None, help='JSON {"temperatures": [...]} applied to the logged (uncalibrated) confidences')
a = ap.parse_args()
cost_doc = json.load(open(a.cost))
cost = {int(L): v['full_ms'] for L, v in cost_doc['summary'].items()}
Ls = sorted(cost)


def interp(L):
    if L in cost:
        return cost[L]
    lo = max(x for x in Ls if x < L)
    hi = min(x for x in Ls if x > L)
    w = (L - lo) / (hi - lo)
    return cost[lo] * (1 - w) + cost[hi] * w


C = {L: interp(L) for L in range(Ls[0], Ls[-1] + 1)}
temps = json.load(open(a.sts))['temperatures'] if a.sts else None


def recal(c, k):
    if temps is None:
        return c
    c = min(max(c, 1e-7), 1 - 1e-7)
    z = math.log(c / (1 - c))
    return 1 / (1 + math.exp(-z / temps[k]))


def cost_rule(conf, Lmax, table):
    a_, expect, best, Lb = 1.0, 1.0, 1.0 / table[min(table)], min(table)
    for l in range(1, Lmax + 1):
        a_ *= conf[l - 1]
        expect += a_
        score = expect / table[l]
        if score > best:
            best, Lb = score, l
    return Lb


policies = {f'fixed{L}': (lambda conf, L=L: L) for L in C}
for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
    def th(conf, thr=thr):
        L = 0
        while L < len(conf) and conf[L] >= thr:
            L += 1
        return max(L, 1)
    policies[f'thr{thr}'] = th
policies['cost'] = lambda conf: max(1, cost_rule(conf, max(C), C))
rows = {}
for path in a.runs:
    doc = json.load(open(path))
    for name, r in doc['prompts'].items():
        acc, confs = r['accepted'], r['confidences']
        confs = [[recal(c, k) for k, c in enumerate(cs)] for cs in confs]
        out = {}
        for pol, f in policies.items():
            toks = ms = 0.0
            for ac, cs in zip(acc, confs):
                L = min(f(cs), max(C))
                toks += 1 + min(ac, L)
                ms += C[max(L, min(C))]
            out[pol] = dict(tok_per_round=toks / len(acc), ms_per_round=ms / len(acc), tok_s=1000 * toks / ms)
        rows[f'{path.split("/")[-1]}:{name}'] = out
        best = max(out, key=lambda k: out[k]['tok_s'])
        print(f"{name:7s} rounds {len(acc):3d}  " + '  '.join(f"{k} {out[k]['tok_s']:.1f}" for k in ('fixed7', 'fixed11', 'fixed15', 'thr0.3', 'cost') if k in out)
              + f"   best {best} {out[best]['tok_s']:.1f} ({out[best]['tok_per_round']:.2f} tok/rnd)")
geo = {pol: math.exp(statistics.mean(math.log(r[pol]['tok_s']) for r in rows.values())) for pol in policies}
print('geomean tok/s:', ' '.join(f'{k} {v:.1f}' for k, v in sorted(geo.items(), key=lambda kv: -kv[1])[:12]))
