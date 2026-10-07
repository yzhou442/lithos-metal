"""Paired, interleaved A/B of verify policies inside ONE serve-config session (spec campaign).

One program carries every policy's ops (the drafter's block, the optional context lookup and the cost-aware
select); a policy only rewrites the verify_select / confidence parameter records (mode, fixed L, cost table,
lookup on/off, STS temperatures) before a generation. Each prompt runs reps x policies, the policy order rotating
per rep, so slow drift of the machine (clocks, other load) hits every policy alike. Reports per policy and prompt
the median decode tok/s, tokens/round, GPU ms/round, identity with the baseline, and per-rep ratios to the first
policy; JSON keeps everything raw. GPU: through gpu_run.

    python tools/bench/spec_policy_ab.py --repo . --draft ~/lmopt/heads/e3j --lookup --suite full --reps 3 \
        --policies '{"fixed7": {"mode": "fixed", "L": 7, "ext": false}, "cost_lk": {"mode": "cost", "ext": true}}' --out ab.json
"""
import argparse
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--model', default='nvidia/Qwen3.8-27B-NVFP4')
ap.add_argument('--draft', default=None)
ap.add_argument('--block', type=int, default=None)
ap.add_argument('--lookup', action='store_true', help='compile the context-lookup extension (policies toggle it)')
ap.add_argument('--lookup-base', type=int, default=None, help='drafter rows before the lookup continuation (default: the block)')
ap.add_argument('--suite', default='full')
ap.add_argument('--only', default=None)
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--policies', required=True, help='JSON {name: {mode: fixed|cost, L, ext, sts: [..] | "file", cost: [16 ms] | null}}')
ap.add_argument('--baseline', default=os.path.expanduser('~/lmopt/results/baseline/full.json'))
ap.add_argument('--out', required=True)
a = ap.parse_args()
repo = Path(a.repo).expanduser().resolve()
sys.path.insert(0, str(repo))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from monolith.serving.setup import prepare
from monolith.generate import load_session
from transformers import AutoTokenizer

# the prompt set of spec_lmbench (same text, same token ids)
src = (repo / 'tools' / 'bench' / 'spec_lmbench.py').read_text()
ns_ = {'Path': Path, 'json': json, 'a': argparse.Namespace(repo=str(repo))}
exec(src[src.index('DOC = '):src.index('class NS')], ns_)
PROMPTS, SUITES, LONG, doc_prompt = ns_['PROMPTS'], ns_['SUITES'], ns_['LONG'], ns_['doc_prompt']
policies = json.loads(a.policies)


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft, ns.no_draft, ns.draft_revision, ns.draft_block_size = a.draft, False, None, a.block
ns.max_context = 32768 - max(0, (a.block or 7) - 7)
ns.draft_quantization, ns.draft_pack, ns.pack = 'auto', None, None
ns.kernel_config, ns.kernel_config_key = None, None
ns.verify_rule, ns.draft_lookup = 'cost', a.lookup
assets = prepare(ns)
tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
names = a.only.split(',') if a.only else SUITES[a.suite]
items = []
for name in names:
    msgs, mx = doc_prompt(*LONG[name]) if name in LONG else PROMPTS[name]
    ids = list(tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
    items.append((name, ids, mx))
base = json.loads(Path(a.baseline).read_text()) if a.baseline and Path(a.baseline).exists() else None
by_key = {}
for name, ids, mx in items:
    key, options = assets.options(len(ids))
    by_key.setdefault(key, []).append((name, ids, mx, options))


def resolve_sts(v):
    if v is None:
        return None
    if isinstance(v, str):
        return json.load(open(os.path.expanduser(v)))['temperatures']
    return v


def patch(session, pol, tier_cost):
    """Rewrite every verify_select / confidence record of the session's programs (and live engines)."""
    for progkey, prog in session._programs.items():
        for op in prog.ops:
            if op.name not in ('verify_select', 'confidence'):
                continue
            slot = 3 if op.name == 'verify_select' else 5
            bname, off = next((n, o) for s, n, o in op.bindings if s == slot)
            spec = prog.buffers[bname]
            raw = bytearray(spec.init)
            if op.name == 'verify_select':
                f = list(struct.unpack_from('<IfII16fIIII', raw, 0))
                gamma, t_max = f[0], f[2]
                if pol['mode'] == 'fixed':
                    f[3], f[1] = 2, float(int(pol.get('L', gamma)))
                else:
                    cost = pol.get('cost') or tier_cost
                    n = min(16, t_max) if pol.get('ext') else min(gamma, t_max - 1) + 1
                    rel = [c / cost[0] for c in cost[:n]] + [1.0] * (16 - n)
                    f[3], f[1] = 1, 0.0
                    f[4:20] = rel
                f[23] = (1 if pol.get('ext') else 0) | (2 if pol.get('prev_full') else 0)
                struct.pack_into('<IfII16fIIII', raw, 0, *f)
            else:
                f = list(struct.unpack_from('<IIII16f', raw, 0))
                sts = resolve_sts(pol.get('sts')) or [1.0] * f[0]
                f[4:20] = [float(x) for x in sts] + [1.0] * (16 - len(sts))
                struct.pack_into('<IIII16f', raw, 0, *f)
            spec.init = bytes(raw)
            eng = session.engines.get(progkey)
            if eng is not None:
                eng.buffers[bname].write(spec.init, 0)


results = {p: {} for p in policies}
session = None
t_start = time.time()
for key, its in by_key.items():
    options = dict(its[0][3])
    if a.lookup and a.lookup_base is not None:
        options['drafter_options'] = dict(options['drafter_options'], lookup={'base': a.lookup_base})
    tier_cost = options.get('verify_cost')
    if session is not None:
        session.release_engines()
        del session
    session = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0, autotune=False,
                           prefill_chunk_size=128)
    session.generate(its[0][1][:64], 16)
    for name, ids, mx, _ in its:
        runs = {p: [] for p in policies}
        order = list(policies)
        for rep in range(a.reps):
            rot = order[rep % len(order):] + order[:rep % len(order)]
            for p in rot:
                patch(session, policies[p], tier_cost)
                g = session.generate(ids, mx)
                runs[p].append(dict(rep=rep, decode_wall_ms=g.decode_wall_ms, decode_gpu_ms=g.decode_ms, steps=g.steps,
                                    tokens=g.tokens[:mx], accepted=g.accepted, verify_len=g.verify_len,
                                    committed=g.committed, tokens_per_step=g.tokens_per_step, confidences=g.confidences))
        for p in policies:
            rs = runs[p]
            tps = [(len(r['tokens']) - 1) / (r['decode_wall_ms'] / 1e3) for r in rs]
            med = sorted(rs, key=lambda r: r['decode_wall_ms'])[len(rs) // 2]
            ident = None
            if base and name in base['prompts']:
                bt = base['prompts'][name]['tokens']
                ident = all(r['tokens'] == bt for r in rs)
            results[p][name] = dict(tok_s=statistics.median(tps), tok_s_all=tps, tokens_per_step=med['tokens_per_step'],
                                    round_ms=med['decode_gpu_ms'] / max(1, med['steps']), identical=ident,
                                    ext_rounds=sum(1 for v in med['verify_len'] if v > (assets.gamma or 7)) / max(1, len(med['verify_len'])),
                                    runs=rs)
        ref = list(policies)[0]
        line = f'{name:6s} ' + '  '.join(f"{p} {results[p][name]['tok_s']:6.1f} ({results[p][name]['tokens_per_step']:.2f}/r {results[p][name]['round_ms']:.1f}ms"
                                          f"{'' if results[p][name]['identical'] is not False else ' NOT-ID'})" for p in policies)
        ratios = {p: [x / y for x, y in zip(results[p][name]['tok_s_all'], results[ref][name]['tok_s_all'])] for p in policies}
        print(line, flush=True)
        print('       ratio vs ' + ref + ': ' + '  '.join(f"{p} {statistics.median(r):.3f} [{min(r):.3f},{max(r):.3f}]" for p, r in ratios.items() if p != ref), flush=True)
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
summary = {p: dict(geomean=gm([v['tok_s'] for v in results[p].values()]),
                   all_identical=all(v['identical'] is not False for v in results[p].values())) for p in policies}
print('SUMMARY', json.dumps({p: round(v['geomean'], 2) for p, v in summary.items()}), 'identical', {p: v['all_identical'] for p, v in summary.items()})
Path(a.out).expanduser().write_text(json.dumps(dict(args=vars(a), gamma=assets.gamma, summary=summary, results=results,
                                                    seconds=time.time() - t_start, time=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
