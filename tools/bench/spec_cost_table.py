"""Measured cost(T) of the DSpark round under the exact serving configuration (spec campaign, M5 Max).

Builds the serving assets like ``lithos-metal serve`` (``serving.setup.prepare`` + the recipe-selected
``load_session``), prefills a real prompt, snapshots StepState and the recurrent states, then replays single rounds
with the verify length forced to every requested L (T = 1 + L rows): the StepState fields the previous round's
``verify_select`` would have written (``t_this_step``, ``verify_len``, ``pending_tokens``) are patched on the host
before each replay. Reports, per T, the GPU time of the verification span (target pass + sampling), the complete
round (verification + accept/commit + next draft + select) and, once, the accept/commit and draft spans.
Optionally attributes the verification span to kernel functions with per-dispatch timestamps (re-encoded, so the
absolute numbers carry encoder gaps; use the shares and the T-scaling, not the totals).

    gpu_run --agent spec -- python tools/bench/spec_cost_table.py --repo . --block 15 --ls 1,3,7,11,15 --out x.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--model', default='nvidia/Qwen3.8-27B-NVFP4')
ap.add_argument('--draft', default=None)
ap.add_argument('--block', type=int, default=None, help='serving block (drafts per round); default the serve default')
ap.add_argument('--max-context', type=int, default=32768)
ap.add_argument('--ctx', type=int, default=256, help='prompt tokens (taken from the repo design doc)')
ap.add_argument('--ls', default=None, help='comma list of verify lengths L (default 1..block)')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--warmup', type=int, default=2)
ap.add_argument('--no-target-recipe', action='store_true')
ap.add_argument('--no-draft-recipe', action='store_true')
ap.add_argument('--recipe-key', default=None, help='force a recipe context key (128/4096/...)')
ap.add_argument('--profile-ops', action='store_true', help='per-kernel-function attribution of the verification span')
ap.add_argument('--session-kw', default='{}')
ap.add_argument('--drafter-kw', default='{}', help='drafter_options entries (e.g. {"lookup": {}})')
ap.add_argument('--out', required=True)
a = ap.parse_args()
repo = Path(a.repo).expanduser().resolve()
sys.path.insert(0, str(repo))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import monolith
from monolith.serving.setup import prepare
from monolith.generate import load_session
from monolith.runtime import _native as nt
assert Path(monolith.__file__).resolve().is_relative_to(repo), monolith.__file__


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft, ns.no_draft, ns.draft_revision, ns.draft_block_size = a.draft, False, None, a.block
ns.max_context, ns.draft_quantization, ns.draft_pack, ns.pack = a.max_context, 'auto', None, None
ns.kernel_config, ns.kernel_config_key = None, a.recipe_key
assets = prepare(ns)
gamma = assets.gamma
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
text = ((repo / 'docs' / 'design' / 'design.md').read_text() + (repo / 'docs' / 'research' / 'apple-gpu-probes.md').read_text()) * 8
ids = tok(text, add_special_tokens=False)['input_ids'][:a.ctx]
key, options = assets.options(len(ids))
if a.no_target_recipe:
    options.pop('decoder_kernel_config', None)
if a.no_draft_recipe and 'drafter_options' in options:
    options['drafter_options'].pop('kernel_config', None)
options.update(json.loads(a.session_kw))
options['drafter_options'] = dict(options['drafter_options'], **json.loads(a.drafter_kw))
t0 = time.time()
s = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0, autotune=False,
                 prefill_chunk_size=128)
load_s = time.time() - t0


def prefill(session, ids):
    session.reset()
    pre = session.prefill_engine(len(ids))
    chunks = [ids[i:i + session.prefill_chunk_size] for i in range(0, len(ids), session.prefill_chunk_size)]
    total = 0.0
    for i, chunk in enumerate(chunks):
        st = pre.state()
        st.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks) - 1 - i, stop_at=0)
        pre.buffers[pre.program.step_state].write(pre.program.layout.pack(st), 0)
        r = pre.run(1, steps_per_cb=1, in_flight=1)
        total += r.gpu_ms
    return total


t0 = time.time()
prefill_ms = prefill(s, ids)
pre = next(iter(s.engines.values()))
s.buffers = {n: b for n, b in pre.buffers.items()
             if pre.program.buffers[n].role in ('state', 'step_state', 'ring') or n in ('accept_log', 'conf_log')}
s.engines.clear()
del pre
gc.collect()
e = s.engine(0)
s.buffers = dict(e.buffers)
compile_s = time.time() - t0
p = e.program
accept_i = next(i for i, o in enumerate(p.ops) if o.name == 'accept_scan')
draft_i = next(i for i, o in enumerate(p.ops) if o.name == 'tap_concat')


def runner(start, end):
    ops = e.ops[start:end]
    icb = nt.Icb(e.dev, ops)
    names = {n for o in p.ops[start:end] for _, n, _ in o.bindings}
    r = nt.Runner(e.dev, icb, ops, [e.buffers[n] for n in names], e.buffers[p.step_state],
                  p.layout.offset('done'), p.layout.offset('ring_head'), p.layout.offset('ring_tail'),
                  e.buffers[p.ring], p.ring_capacity)
    return r, icb


spans = {'verification': runner(0, accept_i), 'accept_commit': runner(accept_i, draft_i), 'draft': runner(draft_i, len(p.ops))}
state0 = e.read(p.step_state)
recurrent = {n: e.read(n) for n, b in p.buffers.items()
             if b.role == 'state' and (n.endswith('rec_state') or n.endswith('conv_state'))}
init = e.state()
assert init['position'] == len(ids), init
drafts0 = list(init['draft_tokens'][:gamma])


def restore(L=None):
    e.buffers[p.step_state].write(state0, 0)
    for n, data in recurrent.items():
        e.buffers[n].write(data, 0)
    if L is not None:
        st = e.state()
        pend = [st['anchor']] + (drafts0 * 3)[:L]          # rows past the block (lookup extension): repeated drafts
        st.update(t_this_step=1 + L, verify_len=L, pending_tokens=pend + [0] * (p.layout.t_max - len(pend)))
        e.buffers[p.step_state].write(p.layout.pack(st), 0)


def run_span(name, L):
    restore(L)
    r, _ = spans[name]
    if name != 'verification':     # the later spans need the verification's outputs: run it first, untimed
        v, _ = spans['verification']
        res = v.run(1, 1, 1, False, 0)
        assert not res.error, res.error
        v.drain()
        if name == 'draft':
            ac, _ = spans['accept_commit']
            res = ac.run(1, 1, 1, False, 0)
            assert not res.error, res.error
            ac.drain()
    res = r.run(1, 1, 1, False, 0)
    assert not res.error, res.error
    r.drain()
    return res.gpu_ms


def run_full(L):
    restore(L)
    r = e.run(1, steps_per_cb=1, in_flight=1)
    st = e.state()
    assert not st['error'], st
    return r.gpu_ms, r.wall_ms, st['accepted'], len(r.tokens)


Ls = [int(x) for x in a.ls.split(',')] if a.ls else list(range(1, gamma + 1))
assert all(1 <= L <= s.decode_t_max - 1 for L in Ls) and (s.decode_t_max > gamma + 1 or all(L <= gamma for L in Ls)), (Ls, gamma)
for _ in range(a.warmup):
    for L in Ls:
        run_full(L)
        run_span('verification', L)
rows = {L: dict(verify=[], full=[], full_wall=[], accepted=None, committed=None) for L in Ls}
for rep in range(a.reps):
    order = Ls if rep % 2 == 0 else list(reversed(Ls))
    for L in order:
        rows[L]['verify'].append(run_span('verification', L))
        g, w, acc, n = run_full(L)
        rows[L]['full'].append(g)
        rows[L]['full_wall'].append(w)
        rows[L]['accepted'], rows[L]['committed'] = acc, n
Lnat = Ls[-1]
stages = {name: statistics.median(run_span(name, Lnat) for _ in range(a.reps)) for name in ('accept_commit', 'draft')}
ops_attr = {}
if a.profile_ops:
    import collections
    for L in Ls:
        acc = collections.defaultdict(list)
        for _ in range(3):
            restore(L)
            prof = e.profile(1)[0]
            per = collections.defaultdict(float)
            for i, (t_s, t_e) in enumerate(prof):
                fn = p.kernels[p.ops[i].kernel].function
                phase = 'V' if i < accept_i else ('C' if i < draft_i else 'D')
                nm = p.ops[i].name
                if phase == 'D':
                    fn = ('markov.' if '.markov' in nm or (fn.startswith('argmax') and i > draft_i) else '') + fn
                per[phase + ':' + fn] += (t_e - t_s)
            for k, v in per.items():
                acc[k].append(v)
        ops_attr[L] = {k: statistics.median(v) for k, v in sorted(acc.items(), key=lambda kv: -statistics.median(kv[1]))}
summary = {}
for L in Ls:
    r = rows[L]
    summary[L] = dict(T=1 + L, verify_ms=statistics.median(r['verify']), verify_min=min(r['verify']), verify_max=max(r['verify']),
                      full_ms=statistics.median(r['full']), full_min=min(r['full']), full_max=max(r['full']),
                      full_wall_ms=statistics.median(r['full_wall']), accepted=r['accepted'], committed=r['committed'])
    print(f"L={L:2d} T={1+L:2d} verify {summary[L]['verify_ms']:7.2f} ms [{summary[L]['verify_min']:.2f},{summary[L]['verify_max']:.2f}]  "
          f"full {summary[L]['full_ms']:7.2f} ms [{summary[L]['full_min']:.2f},{summary[L]['full_max']:.2f}] acc {r['accepted']}", flush=True)
print('stages at L=%d' % Lnat, stages)
out = dict(args=vars(a), recipe_key=key, gamma=gamma, ctx=len(ids), dispatches=len(p.ops), accept_index=accept_i, draft_index=draft_i,
           t_max=p.layout.t_max, prefill_ms=prefill_ms, load_s=load_s, compile_s=compile_s, initial_drafts=drafts0,
           target_recipe=bool(options.get('decoder_kernel_config')),
           draft_recipe=bool((options.get('drafter_options') or {}).get('kernel_config')),
           repo_head=os.popen(f'git -C {repo} rev-parse --short HEAD').read().strip(),
           repo_dirty=bool(os.popen(f'git -C {repo} status --porcelain').read().strip()),
           summary=summary, stages=stages, raw=rows, ops=ops_attr, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
print('WROTE', a.out)
