#!/usr/bin/env python3
"""Replay chat rows through lithos-metal's exact serving configuration (serving.setup.prepare + the recipe-selected
load_session) with a chosen DSpark head, sampling parameters, block size and the serve verify flags (--verify-rule,
--draft-lookup, --spec-sampling, --adaptive-block), so the policies run exactly as `lithos-metal serve` would; the
checkpoint's EOS ids end a row unless --ignore-eos.

Resumable: one JSON line per finished row is appended to --out; rows whose id is already there are skipped. Use
--budget-s to stop starting new rows after N seconds and call again. Run nothing else on the GPU meanwhile.

    python tools/bench/spec_head_eval.py --model path/to/target --rows rows.jsonl --out out.jsonl
        [--draft path/to/dspark-head] [--tag NAME] [--temperature 0] [--top-k 20 --top-p 0.95] [--seed 42]
        [--block 7] [--max-new 1024] [--max-context 32768] [--pack DIR] [--budget-s 660]
    python tools/bench/spec_head_eval.py --report out1.jsonl [out2.jsonl ...]   (CPU only: per-suite / per-bucket table)

Row fields used: id, input_ids (pre-rendered; else messages + tmpl chat-template kwargs), suite/domain, bucket,
max_new (optional per-row cap). Per row recorded: prompt/generated tokens, output_ids, per-step accepted/committed,
steps, accept length = generated / steps, sum committed / steps, prefill and decode wall, decode tok/s, finish
(stop|length).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]),
                help='lithos-metal checkout to import (default: the one containing this script)')
ap.add_argument('--rows')
ap.add_argument('--out')
ap.add_argument('--tag', default='')
ap.add_argument('--model', default=None, help='target checkpoint (path or hub id); required unless --report')
ap.add_argument('--draft', default=None)
ap.add_argument('--no-draft', action='store_true')
ap.add_argument('--block', type=int, default=None, help='DSpark block (proposals per round); default = serving (7)')
ap.add_argument('--max-context', type=int, default=32768)
ap.add_argument('--pack', default=None, help='explicit target pack dir (e.g. a larger-capacity pack)')
ap.add_argument('--draft-pack', default=None)
ap.add_argument('--prefill-chunk-size', type=int, default=128)
ap.add_argument('--max-new', type=int, default=1024)
ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--top-k', type=int, default=0)
ap.add_argument('--top-p', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--ignore-eos', action='store_true')
ap.add_argument('--eos-ids', default=None, help='comma list of stop token ids (default: the checkpoint generation_config eos_token_id)')
ap.add_argument('--budget-s', type=float, default=0, help='do not start a new row after this many seconds')
ap.add_argument('--max-rows', type=int, default=0)
ap.add_argument('--only', default=None, help='comma list of row ids')
ap.add_argument('--session-kw', default='{}')
ap.add_argument('--verify-rule', default='fixed', choices=['fixed', 'cost'])
ap.add_argument('--draft-lookup', action='store_true')
ap.add_argument('--spec-sampling', default='match', choices=['match', 'q'])
ap.add_argument('--adaptive-block', action='store_true', help='serve option: adaptive block (block 7 rounds when L>7 is rare)')
ap.add_argument('--report', nargs='*', default=None)
ap.add_argument('--by', default='suite,bucket')
a = ap.parse_args()

def ctx_bucket(n):
    for lim, name in ((4096, '<4K'), (16384, '4-16K'), (32768, '16-32K'), (65536, '32-64K')):
        if n < lim:
            return name
    return '64K+'


def report(paths, by):
    rows = []
    for p in paths:
        for line in open(Path(p).expanduser()):
            r = json.loads(line)
            if r.get('error'):
                continue
            rows.append(r)
    groups = {}
    for r in rows:
        for key in by.split(','):
            v = r.get(key) if key != 'ctx' else ctx_bucket(r['prompt_tokens'])
            groups.setdefault((r.get('tag', ''), key, v), []).append(r)
        groups.setdefault((r.get('tag', ''), 'ALL', 'ALL'), []).append(r)
    print(f"{'tag':14s} {'by':6s} {'value':12s} {'n':>4s} {'pooled':>7s} {'rowmean':>7s} {'tok/s':>7s} {'gen':>7s}")
    for (tag, key, v), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        steps = sum(r['steps'] for r in rs)
        gen = sum(r['gen_tokens'] for r in rs)
        pooled = gen / steps if steps else float('nan')
        rowmean = sum(r['accept_len'] for r in rs if r['steps']) / max(1, sum(1 for r in rs if r['steps']))
        dec = sum(r['decode_wall_ms'] for r in rs) / 1e3
        tps = (gen - len(rs)) / dec if dec else float('nan')
        print(f"{tag:14s} {key:6s} {str(v):12s} {len(rs):4d} {pooled:7.3f} {rowmean:7.3f} {tps:7.1f} {gen:7d}")


if a.report is not None:
    report(a.report, a.by)
    sys.exit(0)

if not a.model or not a.rows or not a.out:
    ap.error('--model, --rows and --out are required unless --report')
sys.path.insert(0, str(Path(a.repo).expanduser().resolve()))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import monolith
from monolith.serving.setup import prepare
from monolith.generate import load_session
assert Path(monolith.__file__).resolve().is_relative_to(Path(a.repo).expanduser().resolve()), monolith.__file__

t_start = time.time()
out_path = Path(a.out).expanduser()
out_path.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out_path.exists():
    for line in open(out_path):
        try:
            done.add(json.loads(line)['id'])
        except Exception:
            pass
rows = [json.loads(l) for l in open(Path(a.rows).expanduser()) if l.strip()]
if a.only:
    keep = set(a.only.split(','))
    rows = [r for r in rows if r['id'] in keep]
todo = [r for r in rows if r['id'] not in done]
print(f'[spec_head_eval] {len(rows)} rows, {len(done)} done, {len(todo)} to do; tag={a.tag}', flush=True)
if not todo:
    print('[spec_head_eval] ALL_DONE', flush=True)
    sys.exit(0)


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft = None if a.no_draft else a.draft
ns.no_draft, ns.draft_revision, ns.draft_block_size = a.no_draft, None, a.block
ns.max_context, ns.draft_quantization, ns.draft_pack, ns.pack = a.max_context, 'auto', a.draft_pack, a.pack
ns.kernel_config, ns.kernel_config_key = None, None
ns.verify_rule, ns.draft_lookup, ns.spec_sampling = a.verify_rule, a.draft_lookup, a.spec_sampling
ns.draft_adaptive_block = a.adaptive_block
t = time.time()
assets = prepare(ns)
setup_s = time.time() - t
tok = None


def ids_of(r):
    global tok
    if r.get('input_ids'):
        return list(r['input_ids'])
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
    kw = dict(r.get('tmpl') or {})          # e.g. {"reasoning_effort": "low"} / {"enable_thinking": false}; {} = template default
    return list(tok.apply_chat_template(r['messages'], tokenize=True, return_dict=False, add_generation_prompt=True,
                                        **({'tools': r['tools']} if r.get('tools') else {}), **kw))


def stop_ids(model_dir):
    """--eos-ids, else the checkpoint's generation_config.json eos_token_id (an id or a list), else none."""
    if a.eos_ids:
        return {int(x) for x in a.eos_ids.split(',')}
    path = Path(model_dir) / 'generation_config.json'
    eos_id = json.loads(path.read_text()).get('eos_token_id') if path.exists() else None
    return set() if eos_id is None else {int(eos_id)} if isinstance(eos_id, int) else {int(x) for x in eos_id}


eos = -1 if a.ignore_eos else None
eos_set = set() if a.ignore_eos else stop_ids(assets.model_dir)
extra = json.loads(a.session_kw)
plan = []
for r in todo:
    ids = ids_of(r)
    mx = int(r.get('max_new') or a.max_new)
    if len(ids) + mx - 1 > a.max_context:
        rec = dict(id=r['id'], tag=a.tag, suite=r.get('suite'), bucket=r.get('bucket'), prompt_tokens=len(ids),
                   error=f'does not fit max_context {a.max_context}', skipped=True)
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        continue
    key, options = assets.options(len(ids))
    plan.append((key, r, ids, mx, options))
plan.sort(key=lambda x: int(x[0]) if x[0] is not None else 0)
session, cur_key, n_done = None, object(), 0
sampling = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=a.seed)
for key, r, ids, mx, options in plan:
    if a.budget_s and time.time() - t_start > a.budget_s:
        print(f'[spec_head_eval] budget reached after {n_done} rows', flush=True)
        break
    if a.max_rows and n_done >= a.max_rows:
        break
    if key != cur_key:
        if session is not None:
            session.release_engines(); del session
        t = time.time()
        opts = dict(options); opts.update(extra)            # --session-kw overrides the recipe's options
        session = load_session(str(assets.model_dir), str(assets.pack_dir), **opts, eos=eos, autotune=False,
                               prefill_chunk_size=a.prefill_chunk_size, **sampling)
        load_s = time.time() - t
        t = time.time(); session.generate(ids[:64], 8); warm_s = time.time() - t
        cur_key = key
        print(f'[spec_head_eval] session recipe={key} load {load_s:.1f}s warm {warm_s:.1f}s', flush=True)
    t0 = time.time()
    try:
        g = session.generate(ids, mx)
    except Exception as exc:  # record and continue (e.g. capacity)
        rec = dict(id=r['id'], tag=a.tag, suite=r.get('suite'), bucket=r.get('bucket'), prompt_tokens=len(ids),
                   error=f'{type(exc).__name__}: {exc}')
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"[spec_head_eval] {r['id']} ERROR {rec['error']}", flush=True)
        continue
    wall = time.time() - t0
    toks = list(g.tokens[:mx])
    stop = next((i for i, x in enumerate(toks) if x in eos_set), None)
    if stop is not None:
        toks = toks[:stop + 1]
    gen = len(toks)
    acc = list(g.accepted or [])
    com = list(g.committed or [])
    steps = len(acc)
    rec = dict(id=r['id'], tag=a.tag, suite=r.get('suite') or r.get('domain'), bucket=r.get('bucket') or ctx_bucket(len(ids)),
               mode=r.get('mode'), ctx=ctx_bucket(len(ids)), recipe=key, prompt_tokens=len(ids), gen_tokens=gen,
               finish='stop' if stop is not None else 'length', steps=steps,
               accept_len=(gen / steps) if steps else None,
               committed_per_step=(sum(com) / steps) if steps else None,
               mean_accepted=(sum(acc) / steps) if steps else None,
               prefill_wall_ms=g.prefill_wall_ms, prefill_gpu_ms=g.prefill_ms, decode_wall_ms=g.decode_wall_ms,
               decode_gpu_ms=g.decode_ms, decode_tok_s=((gen - 1) / (g.decode_wall_ms / 1e3)) if g.decode_wall_ms else None,
               prefill_tok_s=(len(ids) / (g.prefill_wall_ms / 1e3)) if g.prefill_wall_ms else None,
               wall_s=wall, accepted=acc, committed=com, output_ids=toks, verify_len=list(g.verify_len or []), alt_steps=int(getattr(g, 'alt_steps', 0)),
               confidences=[[round(x, 4) for x in c] for c in (g.confidences or [])], sampling=sampling, block=assets.gamma,
               max_new=mx, max_context=a.max_context, draft=str(assets.draft_dir) if assets.draft_dir else None,
               time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(out_path, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    n_done += 1
    print(f"[spec_head_eval] {r['id'][:34]:34s} P={len(ids):6d} gen={gen:5d} acc={rec['accept_len'] or 0:5.2f} "
          f"dec={rec['decode_tok_s'] or 0:6.1f} tok/s pre={rec['prefill_tok_s'] or 0:6.1f} tok/s wall={wall:5.1f}s", flush=True)
left = len(plan) - n_done
print(f'[spec_head_eval] wrote {n_done} rows this call; {left} left' + ('' if left else '; ALL_DONE'), flush=True)
