"""Fit per-position STS temperatures of a head's confidence chain from logged runs (spec_head_eval / spec_lmbench
JSONs with per-round confidences, accepted counts and verify lengths) — e.g. T=1 runs under the q rule, whose
per-position acceptance (sum min(p, q)) differs from the greedy acceptance the confidence head was trained for.
Position k uses the rounds that verified it (verify_len > k) and accepted every draft before it (accepted >= k);
the event is accepted >= k + 1. CPU only.

    python tools/bench/spec_sts_from_runs.py runs1.jsonl [runs2.jsonl ...] --out sts.json
"""
import argparse
import json
import math

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--min-n', type=int, default=30)
a = ap.parse_args()
conf, acc, vl = [], [], []
for path in a.runs:
    for line in open(path):
        r = json.loads(line)
        for c, x, L in zip(r.get('confidences') or [], r['accepted'], r.get('verify_len') or []):
            conf.append(c); acc.append(x); vl.append(L)
g = max(len(c) for c in conf)
C = np.array([c + [0.5] * (g - len(c)) for c in conf], dtype=np.float64)
A, V = np.array(acc), np.array(vl)
temps, stats = [], []
for k in range(g):
    m = (V > k) & (A >= k)
    y = (A[m] >= k + 1).astype(np.float64)
    c = np.clip(C[m, k], 1e-6, 1 - 1e-6)
    if m.sum() < a.min_n:
        temps.append(temps[-1] if temps else 1.0); stats.append(dict(k=k, n=int(m.sum()))); continue
    z = np.log(c / (1 - c))
    best, bt = None, 1.0
    for t in np.exp(np.linspace(math.log(0.2), math.log(8.0), 161)):
        p = np.clip(1 / (1 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best is None or bce < best:
            best, bt = bce, float(t)
    temps.append(bt)
    stats.append(dict(k=k, n=int(m.sum()), accept_rate=float(y.mean()), mean_conf=float(c.mean())))
    print(f'k={k:2d} n={int(m.sum()):5d} accept {y.mean():.3f} conf {c.mean():.3f} tau {bt:.2f}')
json.dump(dict(temperatures=temps, stats=stats, runs=a.runs, rounds=len(acc)), open(a.out, 'w'), indent=1)
print('WROTE', a.out)
