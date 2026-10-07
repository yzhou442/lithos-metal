"""Frequency-ranked draft vocabulary for an FR-Spec style draft head (CPU only).

Counts target-tokenizer tokens over a text corpus (JSONL lines {"role", "t"}; assistant text weighted
``--assistant-weight``), always keeps the special/control tokens, and writes the ranked token ids plus the
coverage of reference generations (spec_lmbench JSONs: the fraction of generated tokens inside the
top-K set, per prompt and overall) for K in --ks.

    python tools/bench/spec_vocab_rank.py --model path/to/target --corpus fr_corpus.jsonl --refs spec_lmbench_full.json --out vocab_rank.json
"""
import argparse
import collections
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--corpus', required=True)
ap.add_argument('--refs', nargs='*', default=[])
ap.add_argument('--model', required=True, help='target checkpoint (path or hub id; its tokenizer)')
ap.add_argument('--assistant-weight', type=float, default=3.0)
ap.add_argument('--ks', default='8192,16384,32768,49152,65536')
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
counts = collections.Counter()
texts = [json.loads(line) for line in open(a.corpus)]
batch = 256
for i in range(0, len(texts), batch):
    chunk = texts[i:i + batch]
    enc = tok([x['t'] for x in chunk], add_special_tokens=False)['input_ids']
    for x, ids in zip(chunk, enc):
        w = a.assistant_weight if x.get('role') == 'assistant' else 1.0
        for t in ids:
            counts[t] += w
special = sorted(set(tok.all_special_ids) | {i for i in range(248000, len(tok))})
ranked = list(dict.fromkeys(special + [t for t, _ in counts.most_common()]))
seen = set(ranked)
ranked += [t for t in range(len(tok)) if t not in seen]          # the tail in id order (never selected at K < seen)
ks = [int(k) for k in a.ks.split(',')]
cov = {}
for ref in a.refs:
    doc = json.load(open(ref))
    for name, r in doc['prompts'].items():
        toks = r['tokens']
        cov[f'{os.path.basename(ref)}:{name}'] = {k: sum(1 for t in toks if t in set(ranked[:k])) / len(toks) for k in ks}
for k in ks:
    vals = [c[k] for c in cov.values()]
    print(f'K={k:6d}: corpus mass {sum(counts[t] for t in ranked[:k]) / sum(counts.values()):.4f}  '
          f'ref coverage mean {sum(vals) / max(1, len(vals)):.4f} min {min(vals) if vals else 0:.4f}')
json.dump(dict(model=a.model, vocab=len(tok), corpus=a.corpus, n_texts=len(texts), n_tokens=sum(counts.values()),
               distinct=len(counts), special=len(special), ranked=ranked, coverage=cov), open(a.out, 'w'))
print('WROTE', a.out, 'distinct', len(counts))
