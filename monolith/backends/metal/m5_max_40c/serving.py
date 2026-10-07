"""Select measured decoder/draft recipes by shapes and formats, independent of checkpoint names."""
import copy
import json
import os
from pathlib import Path

# Opt-in numerics-changing recipe overlays (recipes/dspark/numerics-overlays.json), named in
# LITHOS_NUMERICS_OVERLAYS (comma separated). Off by default: the selected recipes are the measured,
# baseline-identical ones; an overlay changes rounding and is gated separately (see the overlay's _gate).
OVERLAY_ENV = 'LITHOS_NUMERICS_OVERLAYS'


def _merge(dst, src):
    for k, v in src.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply_overlays(selected, names=None, root=None):
    """Deep-merge the named overlays' per-context patches into selected (in place) and return it."""
    names = os.environ.get(OVERLAY_ENV, '') if names is None else names
    names = [n.strip() for n in (names.split(',') if isinstance(names, str) else names) if n.strip()]
    if not names:
        return selected
    root = root or Path(__file__).parent / 'recipes' / 'dspark'
    overlays = json.loads((root / 'numerics-overlays.json').read_text())
    for name in names:
        if name not in overlays:
            raise ValueError(f'unknown numerics overlay {name!r} (known: {sorted(k for k in overlays if not k.startswith("_"))})')
        for key, patch in overlays[name].items():
            if not key.startswith('_') and key in selected:
                _merge(selected[key], patch)
                if 'target' in patch:
                    # a named overlay decides the whole target recipe: drop the row-bound default (target_rows8)
                    selected[key].pop('target_rows8', None)
    return selected


def recipes(model, drafter, quantization):
    if drafter is None or quantization not in (None, 'nvfp4'):
        return {}
    target, draft = model.config, drafter.cfg
    if (tuple(getattr(target, k, None) for k in ('hidden_size','num_hidden_layers','num_experts',
                  'num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size')) == (2048,40,256,8,512,512)
            and tuple(getattr(draft,k,None) for k in ('hidden_size','intermediate_size','num_hidden_layers',
                  'num_attention_heads','num_key_value_heads','head_dim','vocab_size')) == (2048,6144,6,32,8,128,248320)
            and draft.block_size == 7 and draft.target_layer_ids == [1,6,11,16,22,27,32,37]
            and draft.markov_rank == 256 and draft.target_hidden == 2048):
        return {'0':dict(bf16_min_t=2,draft_attention='mma')}
    fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size')
    if tuple(getattr(target, k, None) for k in fields) != (5120, 17408, 64, 24, 4, 256, 248320):
        return {}
    if tuple(getattr(draft, k, None) for k in fields) != (5120, 17408, 5, 32, 8, 128, 248320):
        return {}
    if (not 1 <= draft.block_size <= 15 or draft.target_layer_ids != [5, 19, 33, 47, 61]
            or draft.markov_rank != 256 or draft.target_hidden != 5120):
        return {}
    shape = tuple(getattr(target, k, None) for k in ('linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim'))
    if shape != (16, 48, 128, 128, 4):
        return {}
    if list(getattr(target, 'layer_types', ())) != [
            'full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)]:
        return {}
    formats = {s.format for _, mod in model.named_modules() for s in mod.weight_map().values() if not s.aux}
    if not {'nvfp4', 'fp8_e4m3'} <= formats:
        return {}
    return measured_recipes(quantization)


def measured_recipes(quantization):
    """The measured 27B DSpark recipe map for a draft quantization (None or 'nvfp4'), without the shape checks."""
    root = Path(__file__).parent / 'recipes' / 'dspark'
    contexts = json.loads((root / 'selected-contexts.json').read_text())
    if quantization is None:
        return apply_overlays(contexts)
    selected = json.loads((root / 'selected-nvfp4-endpoints.json').read_text())
    # Target recipes do not depend on draft precision. Reuse the long-context
    # NVFP4 draft with each measured target recipe instead of duplicating JSON.
    for key in ('4096', '8192', '16384'):
        selected[key] = copy.deepcopy(selected['32768'])
        selected[key]['target'] = contexts[key]['target']
        if 'target_rows8' in contexts[key]:
            # a patch for the eight-row verify program only (serving.setup applies it when the block is <= 7 without
            # the lookup extension): measured faster there and slower in the sixteen-row program
            selected[key]['target_rows8'] = contexts[key]['target_rows8']
    # The recipes were measured at block 7 (verify 8 rows); blocks 8-15 compile the same recipes at the sixteen-row
    # bound (TM16 tiles, T_HI 16, single-pass GDN at T <= 16: static_fusion / emit), with every per-row result equal.
    return apply_overlays(selected)


def verify_costs(recipes=None):
    """Measured full-round GPU ms per verify length l = 0 … 15 (l = 0 as l = 1) for the sixteen-row programs, per
    context tier: ``b15`` = block 15 drafts; ``b7lk`` = block 7 drafts + the context-lookup extension in rows 9-16.
    The cost-aware verify rule (``verify='cost'``) uses them relative to l = 0 (tools/bench/spec_cost_table.py).
    They were measured on one workload, the NVFP4-draft DSpark serving recipes selected by ``recipes()``, so they
    are returned only when the session's recipe map is exactly that one (empty otherwise: other models, the MoE
    pairing, a source-precision draft or a custom ``--kernel-config`` keep fixed verification)."""
    if not recipes or recipes != measured_recipes('nvfp4'):
        return {}
    path = Path(__file__).parent / 'recipes' / 'dspark' / 'verify-cost.json'
    return json.loads(path.read_text()) if path.exists() else {}
