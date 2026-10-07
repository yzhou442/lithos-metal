"""[spec campaign copy of ~/lmopt/bin/lmbench.py: identical prompts/metrics; --session-kw entries override the serving
options instead of being added beside them (verify rule, STS, drafter options), and every repetition records the
per-round accepted / committed / verify-length / confidence logs.]
End-to-end lithos-metal benchmark with the exact `lithos-metal serve` configuration (serving.setup.prepare + the
recipe-selected load_session), greedy, fixed-length (EOS ignored), plus a token-identity check against a baseline.

usage (always through gpu_run):
  gpu_run --agent NAME -- ~/lmopt/venv/bin/python ~/lmopt/bin/lmbench.py --repo ~/lmopt/wt/NAME \
      --suite quick|full|long --out ~/lmopt/results/NAME/run1.json [--baseline ~/lmopt/results/baseline/full.json]
      [--draft PATH_OR_HUB_ID | --no-draft] [--draft-block-size N] [--repeat 2]

Metrics per prompt: prefill tok/s (prompt tokens / prefill wall), decode tok/s (generated tokens after the first /
decode wall), tokens per round, mean accepted drafts, GPU ms per token. Summary = geometric means over the suite.
`identical` = every generated token equals the baseline's for that prompt (greedy). A change that alters numerics
must say so; the integration gate is identity on the quick+full suites unless a numerics change is agreed.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True, help='lithos-metal checkout to import (worktree)')
ap.add_argument('--suite', default='quick', choices=['quick', 'full', 'long', 'prefill', 'agent'])
ap.add_argument('--out', required=True)
ap.add_argument('--baseline')
ap.add_argument('--model', default='nvidia/Qwen3.8-27B-NVFP4')
ap.add_argument('--draft', default=None)
ap.add_argument('--no-draft', action='store_true')
ap.add_argument('--draft-block-size', type=int, default=None)
ap.add_argument('--max-context', type=int, default=32768)
ap.add_argument('--prefill-chunk-size', type=int, default=128)
ap.add_argument('--max-new', type=int, default=None, help='override generated tokens per prompt')
ap.add_argument('--repeat', type=int, default=1, help='timed repetitions per prompt (median reported)')
ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--only', default=None, help='comma list of prompt ids')
ap.add_argument('--session-kw', default='{}', help='load_session kwargs as JSON; override the serving options (experiments)')
ap.add_argument('--drafter-kw', default='{}', help='drafter_options entries as JSON, merged into the serving ones')
ap.add_argument('--no-target-recipe', action='store_true')
ap.add_argument('--no-draft-recipe', action='store_true')
ap.add_argument('--recipe-key', default=None)
ap.add_argument('--verify-rule', default='fixed', choices=['fixed', 'cost'], help='serve option: per-round verify length rule')
ap.add_argument('--draft-lookup', action='store_true', help='serve option: context-lookup extension into rows 9-16')
ap.add_argument('--spec-sampling', default='match', choices=['match', 'q'], help='serve option: T > 0 accept rule')
ap.add_argument('--top-k', type=int, default=0)
ap.add_argument('--top-p', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo).expanduser().resolve()))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import monolith
from monolith.serving.setup import prepare
from monolith.generate import load_session
from transformers import AutoTokenizer

assert Path(monolith.__file__).resolve().is_relative_to(Path(a.repo).expanduser().resolve()), monolith.__file__

DOC = Path(a.repo).expanduser() / 'docs' / 'design' / 'design.md'
DOC2 = Path(a.repo).expanduser() / 'docs' / 'research' / 'apple-gpu-probes.md'
SYS_WEB = ('You are a coding expert. Produce a complete, self-contained single HTML file with inline CSS and '
           'JavaScript. Output only the code.')
PROMPTS = {
    'code': ([{'role': 'user', 'content': 'Write a Python function that parses an ISO-8601 timestamp without '
              'using the datetime module. Include a docstring, input validation and unit tests.'}], 256),
    'chat': ([{'role': 'user', 'content': 'Explain to a curious high-school student how attention works in a '
              'transformer language model, with an everyday analogy.'}], 256),
    'math': ([{'role': 'user', 'content': 'A train leaves at 9:40 travelling 84 km/h; a second train leaves the '
              'same station at 10:05 at 102 km/h on a parallel track. When and where does the second catch up? '
              'Solve step by step.'}], 256),
    'web': ([{'role': 'system', 'content': SYS_WEB},
             {'role': 'user', 'content': 'Build a pricing page with three tiers, a monthly/yearly toggle and a '
              'feature comparison table.'}], 384),
    'tool': ([{'role': 'user', 'content': 'Here is a shell session:\n$ ls src\nmain.rs lib.rs parser.rs\n$ cargo test\n'
              'error[E0308]: mismatched types\n --> src/parser.rs:42:17\n   |\n42 |     let n: u32 = tok.len();\n'
              '   |            ---   ^^^^^^^^^ expected `u32`, found `usize`\n\nExplain the error and give the fixed '
              'line plus a short justification.'}], 192),
}
def doc_prompt(n_chars, ask, max_new):
    text = (DOC.read_text() + '\n\n' + DOC2.read_text()) * 4
    return ([{'role': 'user', 'content': text[:n_chars] + '\n\n' + ask}], max_new)
# agentic, copy-heavy prompts (spec campaign): rewrite a file the prompt contains
_BASE = Path('~/lmopt/base').expanduser()      # fixed source text (the base commit), whatever repo is measured
_SETUP = (_BASE / 'monolith' / 'serving' / 'setup.py').read_text()
_RECIPE = json.dumps(json.loads((_BASE / 'monolith' / 'backends' / 'metal' / 'm5_max_40c' / 'recipes' / 'dspark'
                                / 'selected-nvfp4-endpoints.json').read_text())['128']['target'], indent=1)
PROMPTS['edit'] = ([{'role': 'user', 'content': 'Here is a Python module:\n```python\n' + _SETUP + '```\nAdd a one-line docstring to '
                     'every function and method that lacks one, change nothing else, and output the complete updated file.'}], 640)
PROMPTS['json'] = ([{'role': 'user', 'content': 'Here is a JSON config:\n```json\n' + _RECIPE + '\n```\nChange every "workers" value '
                     'of 160 to 192 and every "sgs" value of 4 to 8. Output the complete updated JSON only.'}], 512)
SUITES = {
    'agent': ['edit', 'json'],
    'quick': ['code', 'chat', 'web'],
    'full': ['code', 'chat', 'math', 'web', 'tool', 'doc4k'],
    'long': ['doc4k', 'doc16k'],
    'prefill': ['doc4k', 'doc16k', 'doc28k'],
}
LONG = {'doc4k': (14000, 'Summarize the key design decisions above in 8 bullet points.', 192),
        'doc16k': (58000, 'List the five most important measured hardware facts above and why each matters.', 160),
        'doc28k': (100000, 'Give a one-paragraph summary.', 32)}

class NS:  # argparse-like namespace for serving.setup.prepare
    pass
ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft = None if a.no_draft else a.draft
ns.no_draft, ns.draft_revision, ns.draft_block_size = a.no_draft, None, a.draft_block_size
ns.max_context, ns.draft_quantization, ns.draft_pack, ns.pack = a.max_context, 'auto', None, None
ns.kernel_config, ns.kernel_config_key = None, a.recipe_key
ns.verify_rule, ns.draft_lookup = a.verify_rule, a.draft_lookup
ns.spec_sampling = a.spec_sampling
t = time.time()
assets = prepare(ns)
setup_s = time.time() - t
tok = AutoTokenizer.from_pretrained(str(assets.model_dir))
ids_of = {}
names = SUITES[a.suite] if not a.only else a.only.split(',')
for name in names:
    if name in LONG:
        msgs, mx = doc_prompt(*LONG[name])
    else:
        msgs, mx = PROMPTS[name]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True,
                                  enable_thinking=False)
    ids_of[name] = (list(ids), a.max_new or mx)
extra = json.loads(a.session_kw)
base = json.loads(Path(a.baseline).read_text()) if a.baseline else None
results, sessions_built = {}, 0
by_key = {}
for name, (ids, mx) in ids_of.items():
    key, options = assets.options(len(ids))
    by_key.setdefault(key, []).append((name, ids, mx, options))
session = None
for key, items in by_key.items():
    options = items[0][3]
    if session is not None:
        session.release_engines(); del session
    t = time.time()
    options = dict(options)
    if a.no_target_recipe:
        options.pop('decoder_kernel_config', None)
    if 'drafter_options' in options:
        options['drafter_options'] = dict(options['drafter_options'], **json.loads(a.drafter_kw))
        if a.no_draft_recipe:
            options['drafter_options'].pop('kernel_config', None)
    options.update(extra)
    options.setdefault('temperature', a.temperature)
    if a.temperature > 0:
        options.setdefault('top_k', a.top_k); options.setdefault('top_p', a.top_p); options.setdefault('seed', a.seed)
    session = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, autotune=False,
                           prefill_chunk_size=a.prefill_chunk_size)
    load_s = time.time() - t; sessions_built += 1
    t = time.time(); session.generate(items[0][1][:64], 16); warm_s = time.time() - t   # compile + warm
    for name, ids, mx, _ in items:
        reps = []
        for r in range(a.repeat):
            g = session.generate(ids, mx)
            n = len(g.tokens)
            reps.append(dict(prefill_wall_ms=g.prefill_wall_ms, prefill_gpu_ms=g.prefill_ms,
                             decode_wall_ms=g.decode_wall_ms, decode_gpu_ms=g.decode_ms, steps=g.steps,
                             tokens=g.tokens[:mx], committed=(sum(g.committed) if g.committed else None),
                             mean_accepted=g.mean_accepted, tokens_per_step=g.tokens_per_step,
                             accepted=g.accepted, committed_log=g.committed, verify_len=g.verify_len,
                             confidences=g.confidences))
        reps.sort(key=lambda x: x['decode_wall_ms'])
        m = reps[len(reps) // 2]
        P = len(ids); gen = len(m['tokens'])
        rec = dict(recipe=key, prompt_tokens=P, gen_tokens=gen,
                   prefill_tok_s=P / (m['prefill_wall_ms'] / 1e3) if m['prefill_wall_ms'] else None,
                   decode_tok_s=(gen - 1) / (m['decode_wall_ms'] / 1e3) if m['decode_wall_ms'] else None,
                   gpu_ms_per_token=m['decode_gpu_ms'] / max(1, (m['committed'] or gen - 1)),
                   tokens_per_step=m['tokens_per_step'], mean_accepted=m['mean_accepted'],
                   ttft_ms=m['prefill_wall_ms'], **{k: m[k] for k in ('prefill_wall_ms', 'decode_wall_ms',
                   'decode_gpu_ms', 'steps', 'tokens', 'accepted', 'committed_log', 'verify_len', 'confidences')},
                   load_s=load_s, warm_s=warm_s, round_ms=m['decode_gpu_ms'] / max(1, m['steps']),
                   decode_wall_all=[x['decode_wall_ms'] for x in reps])
        if base and name in base['prompts']:
            bt = base['prompts'][name]['tokens']
            k = min(len(bt), gen)
            first = next((i for i in range(k) if bt[i] != m['tokens'][i]), None)
            rec['identical'] = first is None and len(bt) == gen
            rec['first_diff'] = first
            rec['decode_speedup'] = rec['decode_tok_s'] / base['prompts'][name]['decode_tok_s']
            rec['prefill_speedup'] = (rec['prefill_tok_s'] / base['prompts'][name]['prefill_tok_s']
                                      if base['prompts'][name].get('prefill_tok_s') else None)
        results[name] = rec
        print(f"{name:8s} P={P:6d} prefill {rec['prefill_tok_s'] or 0:8.1f} tok/s  decode {rec['decode_tok_s']:7.1f} "
              f"tok/s  tok/step {rec['tokens_per_step']:.2f}  acc {rec['mean_accepted']:.2f}"
              + (f"  ident={rec['identical']} dspd={rec['decode_speedup']:.3f}" if 'identical' in rec else ''),
              flush=True)
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None
summary = dict(decode_tok_s_geomean=gm([r['decode_tok_s'] for r in results.values() if r['decode_tok_s']]),
               prefill_tok_s_geomean=gm([r['prefill_tok_s'] for r in results.values() if r['prefill_tok_s']]),
               all_identical=(all(r.get('identical') for r in results.values()) if base else None),
               decode_speedup_geomean=(gm([r['decode_speedup'] for r in results.values()]) if base else None),
               prefill_speedup_geomean=(gm([r['prefill_speedup'] for r in results.values() if r.get('prefill_speedup')])
                                        if base else None),
               setup_s=setup_s, sessions=sessions_built)
out = dict(args=vars(a), repo_head=os.popen(f'git -C {a.repo} rev-parse --short HEAD 2>/dev/null').read().strip(),
           repo_dirty=bool(os.popen(f'git -C {a.repo} status --porcelain 2>/dev/null').read().strip()),
           summary=summary, prompts=results, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
print('SUMMARY', json.dumps(summary))
