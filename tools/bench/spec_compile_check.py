"""CPU-only compile of the serving decode program (no Metal device, no dispatch): build the exact serve-time
Program for a head / block / recipe choice and report its dispatches per kernel function, or the compile error.
A fake device stands in for the M5 Max so iterating on compiler/recipe changes never takes the GPU lock.

    python tools/bench/spec_compile_check.py --repo . --block 15 [--target-recipe-key 128] [--dump ops.txt]
"""
import argparse
import collections
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--model', default='nvidia/Qwen3.8-27B-NVFP4')
ap.add_argument('--draft', default=None)
ap.add_argument('--block', type=int, default=None)
ap.add_argument('--max-context', type=int, default=None)
ap.add_argument('--prompt-tokens', type=int, default=128)
ap.add_argument('--target-recipe-key', default=None, help='force this context key\'s 8-row target recipe onto the program')
ap.add_argument('--draft-recipe-key', default=None, help='force this context key\'s draft recipe onto the program')
ap.add_argument('--no-target-recipe', action='store_true')
ap.add_argument('--no-draft-recipe', action='store_true')
ap.add_argument('--session-kw', default='{}')
ap.add_argument('--drafter-kw', default='{}')
ap.add_argument('--dump', default=None, help='write one line per dispatch')
a = ap.parse_args()
repo = Path(a.repo).expanduser().resolve()
sys.path.insert(0, str(repo))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


class FakeInfo:
    gpu_cores, apple_family, name = 40, 10, 'Apple M5 Max'
    recommended_working_set = int(37.44e9)
    max_buffer_length = int(28.08e9)


class FakeDevice:
    def info(self):
        return FakeInfo()


from monolith.runtime import _native as nt
nt.Device = FakeDevice
from monolith.serving.setup import prepare
from monolith.generate import load_session


class NS:
    pass


ns = NS()
ns.model, ns.revision, ns.download_dir, ns.local_files_only = a.model, None, None, True
ns.draft, ns.no_draft, ns.draft_revision, ns.draft_block_size = a.draft, False, None, a.block
ns.max_context = a.max_context or (32768 - max(0, (a.block or 7) - 7))
ns.draft_quantization, ns.draft_pack, ns.pack = 'auto', None, None
ns.kernel_config, ns.kernel_config_key = None, None
assets = prepare(ns, device_info=FakeInfo())
key, options = assets.options(a.prompt_tokens)
if a.target_recipe_key or a.draft_recipe_key:
    from monolith.backends.metal.m5_max_40c import serving
    root = Path(serving.__file__).parent / 'recipes' / 'dspark'
    sel = json.loads((root / 'selected-nvfp4-endpoints.json').read_text())
    ctxs = json.loads((root / 'selected-contexts.json').read_text())
    for k in ('4096', '8192', '16384'):
        sel[k] = copy.deepcopy(sel['32768'])
        sel[k]['target'] = ctxs[k]['target']
    if a.target_recipe_key:
        options['decoder_kernel_config'] = copy.deepcopy(sel[a.target_recipe_key]['target'])
        options['prefill_attention'] = 'auto'
    if a.draft_recipe_key:
        options['drafter_options'] = dict(options['drafter_options'], kernel_config=copy.deepcopy(sel[a.draft_recipe_key]['draft']))
if a.no_target_recipe:
    options.pop('decoder_kernel_config', None)
if a.no_draft_recipe:
    options['drafter_options'].pop('kernel_config', None)
options.update(json.loads(a.session_kw))
options['drafter_options'] = dict(options['drafter_options'], **json.loads(a.drafter_kw))
s = load_session(str(assets.model_dir), str(assets.pack_dir), **options, eos=-1, temperature=0.0, autotune=False,
                 prefill_chunk_size=128)
t0 = time.time()
try:
    prog = s._compile(s.decode_t_max, dynamic=True, prefill=False)
except Exception:
    traceback.print_exc()
    print('COMPILE FAILED', flush=True)
    sys.exit(1)
fn = collections.Counter(prog.kernels[o.kernel].function for o in prog.ops)
acc = next(i for i, o in enumerate(prog.ops) if o.name == 'accept_scan')
dr = next(i for i, o in enumerate(prog.ops) if o.name == 'tap_concat')
print(f'OK gamma={assets.gamma} t_max={s.decode_t_max} recipe_key={key} target_recipe={bool(options.get("decoder_kernel_config"))} '
      f'draft_recipe={bool(options["drafter_options"].get("kernel_config"))} dispatches={len(prog.ops)} verify={acc} commit={dr-acc} '
      f'draft={len(prog.ops)-dr} compile_s={time.time()-t0:.1f}')
print(dict(fn.most_common()))
if a.dump:
    with open(a.dump, 'w') as f:
        for i, o in enumerate(prog.ops):
            k = prog.kernels[o.kernel]
            m = {x: k.macros.get(x) for x in ('TM', 'TN', 'TK', 'KSPLIT', 'T_LO', 'T_HI', 'TP', 'SL', 'SINGLE_PASS', 'LOCAL_PREPARE', 'QM') if x in k.macros}
            f.write(f'{i}\t{k.function}\t{o.name}\t{o.grid}\t{o.threadgroup}\t{int(o.barrier_before)}\t{o.meta.get("fusion_region","")}\t{m}\n')
