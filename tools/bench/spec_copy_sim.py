"""Offline estimate of copy (prompt-lookup / n-gram) drafting on logged greedy generations.

For a spec_lmbench JSON (tokens + per-round committed counts), replays the rounds and asks, at each round's start,
what an n-gram copy proposer would have offered: the longest suffix (>= --min-match tokens, <= --max-ngram) of
prompt + committed text that occurs earlier, proposing the tokens that followed its most recent earlier occurrence.
Reports per prompt: rounds where a copy proposal exists, its accepted length versus the head's accepted length on
the same round, and the tokens per round of an oracle-free merge rule (copy when its match length >= m, else the
head) — first-order, like spec_policy_sim (the trajectory would change).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--prompt-source', default=str(Path(__file__).resolve().parents[2]),
                help='checkout that supplied the run\'s prompt text (spec_lmbench --prompt-source; default: this checkout)')
ap.add_argument('--model', required=True, help='the run\'s target checkpoint (path or hub id; its tokenizer)')
ap.add_argument('--run', required=True)
ap.add_argument('--min-match', type=int, default=3)
ap.add_argument('--max-ngram', type=int, default=8)
ap.add_argument('--block', type=int, default=15)
a = ap.parse_args()
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_prompts import load_prompts  # noqa: E402

PROMPTS, _, LONG, doc_prompt = load_prompts(a.prompt_source)
tok = AutoTokenizer.from_pretrained(a.model)
doc = json.load(open(a.run))


def proposal(hist, block):
    for n in range(min(a.max_ngram, len(hist) - 1), a.min_match - 1, -1):
        suf = hist[-n:]
        for s in range(len(hist) - n - 1, -1, -1):
            if hist[s:s + n] == suf:
                cont = hist[s + n:s + n + block]
                if cont:
                    return n, cont
    return 0, []


tot = {}
for name, r in doc['prompts'].items():
    if name in LONG:
        msgs, _ = doc_prompt(*LONG[name])
    else:
        msgs, _ = PROMPTS[name]
    ids = list(tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
    out = r['tokens']
    pos = 1                       # the first token comes from prefill
    rows = []
    for acc, com in zip(r['accepted'], r['committed_log']):
        hist = ids + out[:pos]
        n, cont = proposal(hist, a.block)
        truth = out[pos:pos + a.block + 1]
        copy_acc = 0
        while copy_acc < len(cont) and copy_acc < len(truth) and cont[copy_acc] == truth[copy_acc]:
            copy_acc += 1
        rows.append((n, copy_acc, acc))
        pos += com
    head = sum(1 + x for _, _, x in rows) / len(rows)
    res = {}
    for m in (3, 4, 6, 8):
        merged = sum(1 + (max(c, x) if False else (c if n >= m else x)) for n, c, x in rows) / len(rows)
        oracle = sum(1 + max(c, x) for n, c, x in rows) / len(rows)
        res[m] = (merged, oracle)
    has = sum(1 for n, _, _ in rows if n >= a.min_match)
    print(f"{name:6s} rounds {len(rows):3d} with-match {has:3d}  head {head:.2f} tok/rnd  "
          + '  '.join(f"m>={m}: {v[0]:.2f}" for m, v in res.items()) + f"  oracle-max {res[3][1]:.2f}")
