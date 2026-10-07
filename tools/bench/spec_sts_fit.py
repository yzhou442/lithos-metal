"""Fit per-position STS temperatures for a DSpark head under the exact serve config.

Runs the serve-config session (``serving.setup.prepare`` + recipe options) with the whole block verified (fixed L =
block) over a calibration prompt set that is disjoint from spec_lmbench's, collects every round's confidences and
accepted count, and fits τ_k per position k on the *conditional* events (draft k accepted given drafts < k were):
the confidence head predicts c_k = P(accept k | accepted < k) and the cost rule multiplies them into survival
probabilities. Writes ``{"temperatures": [...], ...stats}`` for ``load_session(sts_path=...)`` /
``drafter_options={"sts": [...]}``. Run nothing else on the GPU meanwhile.

    python tools/bench/spec_sts_fit.py --model path/to/target --draft path/to/dspark-head --block 15 --out sts_b15.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]),
                help='lithos-metal checkout to import (default: the one containing this script)')
ap.add_argument('--model', required=True, help='target checkpoint (path or hub id)')
ap.add_argument('--draft', default=None)
ap.add_argument('--block', type=int, default=15)
ap.add_argument('--max-new', type=int, default=160)
ap.add_argument('--out', required=True)
a = ap.parse_args()
repo = Path(a.repo).expanduser().resolve()
sys.path.insert(0, str(repo))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from monolith.serving.setup import prepare
from monolith.generate import load_session
from transformers import AutoTokenizer

SYS_WEB = 'You are a frontend engineer. Reply with one complete HTML file (inline CSS and JS) and nothing else.'
CALIB = [
    [{'role': 'user', 'content': 'Write a Python class implementing an LRU cache with get/put in O(1), plus pytest tests.'}],
    [{'role': 'user', 'content': 'Implement merge sort in Rust and explain why it is stable.'}],
    [{'role': 'user', 'content': 'Write a SQL query that returns, per customer, the three most recent orders and their totals.'}],
    [{'role': 'user', 'content': 'A rectangle has perimeter 46 cm and area 120 cm^2. Find its sides, step by step.'}],
    [{'role': 'user', 'content': 'Compute the probability of getting at least two sixes in five rolls of a fair die. Show the work.'}],
    [{'role': 'user', 'content': 'What are good strategies to stay focused while working from home? Be practical.'}],
    [{'role': 'user', 'content': 'Describe the water cycle to a ten-year-old, with a short story to make it memorable.'}],
    [{'role': 'system', 'content': SYS_WEB}, {'role': 'user', 'content': 'A landing page for a coffee shop with a hero section, menu cards and a contact form.'}],
    [{'role': 'system', 'content': SYS_WEB}, {'role': 'user', 'content': 'A todo app with add/remove/complete and local storage persistence.'}],
    [{'role': 'user', 'content': 'Here is a failing test:\n$ pytest\nE   AssertionError: assert parse("2024-02-30") is None\nE    +  where datetime(2024, 3, 1) = parse("2024-02-30")\n'
                                 'Explain the bug in a date parser that rolls invalid days over, and give a fixed version of the function.'}],
    [{'role': 'user', 'content': 'Translate into French and then into German: "The meeting is moved to Thursday afternoon because the client is travelling."'}],
    [{'role': 'user', 'content': 'Write a JSON schema for a blog post (title, author, tags, body, published_at) and an example document.'}],
]


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft, ns.no_draft, ns.draft_revision, ns.draft_block_size = a.draft, False, None, a.block
ns.max_context = 32768 - max(0, a.block - 7)
ns.draft_quantization, ns.draft_pack, ns.pack = 'auto', None, None
ns.kernel_config, ns.kernel_config_key = None, None
assets = prepare(ns)
tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
ids = [list(tok.apply_chat_template(m, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)) for m in CALIB]
key, options = assets.options(max(len(x) for x in ids))
s = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0, autotune=False, prefill_chunk_size=128)
s.generate(ids[0][:32], 8)
g = assets.gamma
conf_rows, acc_rows = [], []
t0 = time.time()
for x in ids:
    gen = s.generate(x, a.max_new)
    conf_rows += gen.confidences
    acc_rows += gen.accepted
conf = np.array(conf_rows, dtype=np.float64)          # [rounds, g]
acc = np.array(acc_rows)
temps, stats = [], []
for k in range(g):
    m = acc >= k                                       # reached position k (drafts < k accepted)
    y = (acc[m] >= k + 1).astype(np.float64)
    c = np.clip(conf[m, k], 1e-6, 1 - 1e-6)
    if m.sum() < 8:
        temps.append(temps[-1] if temps else 1.0)
        stats.append(dict(k=k, n=int(m.sum())))
        continue
    z = np.log(c / (1 - c))
    best, bt = None, 1.0
    for t in np.exp(np.linspace(math.log(0.2), math.log(8.0), 161)):
        p = np.clip(1 / (1 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best is None or bce < best:
            best, bt = bce, float(t)
    temps.append(bt)
    stats.append(dict(k=k, n=int(m.sum()), accept_rate=float(y.mean()), mean_conf=float(c.mean()),
                      mean_conf_cal=float((1 / (1 + np.exp(-z / bt))).mean()), tau=bt))
    print(f'k={k:2d} n={int(m.sum()):4d} accept {y.mean():.3f} conf {c.mean():.3f} -> tau {bt:.2f} cal {stats[-1]["mean_conf_cal"]:.3f}', flush=True)
out = dict(temperatures=temps, head=str(a.draft or 'published'), block=g, rounds=int(len(acc)), stats=stats,
           mean_accepted=float(acc.mean()), seconds=time.time() - t0, prompts=len(ids), max_new=a.max_new)
Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
print('WROTE', a.out, 'mean accepted', acc.mean())
