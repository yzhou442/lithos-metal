"""Build serving assets once; select chip-owned recipes for each request."""
import copy
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from ..backends.metal import config_for_device, get_backend
from ..backends.metal.config import COST_FORMAT
from ..formats import PackLayout
from ..models import resolve_model
from ..models.catalog import default_draft
from ..nn.pack_plan import bind_formats
from ..formats.safetensors_reader import SafetensorsDir
from ..spec import DRAFTERS
from .cache import default_pack_cache, ensure_pack, resolve_checkpoint

LOG = logging.getLogger(__name__)


def _merge_recipe(dst, src):
    """Deep-merge a recipe patch (None deletes a key)."""
    for key, value in src.items():
        if value is None:
            dst.pop(key, None)
        elif isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_recipe(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


@dataclass
class ServingAssets:
    model_dir: Path
    pack_dir: Path
    max_context: int
    capacity: int
    profile: object
    draft_dir: Path | None = None
    draft_pack: Path | None = None
    gamma: int = 0
    recipes: dict | None = None
    recipe_key: str | None = None
    prefill_chunk_size: int | None = None   # the backend's measured prefill pass size for this model, if any
    verify_rule: str = 'fixed'
    lookup: bool = False
    spec_sampling: str = 'match'
    verify_costs: dict | None = None

    def options(self, prompt_tokens):
        profile = copy.deepcopy(self.profile)
        options = dict(profile=profile, max_context=self.capacity)
        key = None
        if self.draft_dir:
            options.update(drafter_dir=str(self.draft_dir), drafter_pack=str(self.draft_pack),
                           drafter_kind='dspark', verify='fixed', verify_length=self.gamma,
                           drafter_options=dict(block_size=self.gamma))
        if self.recipes:
            keys = sorted(int(k) for k in self.recipes)
            key = self.recipe_key or str(max((k for k in keys if k <= prompt_tokens), default=keys[0]))
            recipe = self.recipes[key]
            profile.accelerator_min_t.update({COST_FORMAT.get(k, k): v
                for k, v in recipe.get('accelerator_min_t', {}).items()})
            if 'bf16_min_t' in recipe:
                profile.accelerator_min_t['bf16'] = recipe['bf16_min_t']
            target = copy.deepcopy(recipe.get('target'))
            if target and recipe.get('target_rows8') and self.gamma <= 7 and not self.lookup:
                # Recipe entries measured on the eight-row verify program only (blocks above 7 and the lookup
                # extension compile a sixteen-row program, where they were measured slower).
                _merge_recipe(target, recipe['target_rows8'])
            options.update(drafter_options=dict(block_size=self.gamma, attention=recipe.get('draft_attention', 'mma'),
                                               kernel_config=copy.deepcopy(recipe.get('draft'))),
                           decoder_kernel_config=target,
                           prefill_attention=recipe.get('prefill_attention', 'auto' if profile.backend == 'm5_max_40c'
                                                        and recipe.get('target') else 'v3'), accelerator='on')
        if self.draft_dir and self.spec_sampling != 'match':
            options['spec_sampling'] = self.spec_sampling          # temperature > 0 only; greedy requests are unaffected
        if self.draft_dir and self.lookup:
            options['drafter_options'] = dict(options['drafter_options'], lookup={})
        if self.draft_dir and self.verify_rule == 'cost':
            # The chip's measured whole-round cost per verify length, for this program shape and context tier.
            kind = 'b7lk' if self.lookup and self.gamma <= 7 else 'b15' if self.gamma > 7 else None
            tables = (self.verify_costs or {}).get(kind) or {}
            if tables:
                tier = str(max((int(k) for k in tables if int(k) <= prompt_tokens), default=min(int(k) for k in tables)))
                options.update(verify='cost', verify_cost=list(tables[tier]))
                options.pop('verify_length', None)
        return key, options


def prepare(args, *, device_info=None):
    if device_info is None:
        from ..runtime import _native
        if _native is None:
            raise RuntimeError('The Metal runtime is unavailable. Install lithos-metal on Apple silicon with macOS 26+; see docs/installation.md')
        device_info = _native.Device().info()
    profile = config_for_device(device_info.gpu_cores, device_info.apple_family, device_info.name)
    if profile is None:
        raise ValueError(f'No registered backend for {device_info.name} ({device_info.gpu_cores} cores)')
    backend = get_backend(profile.backend)
    resolve = dict(download_dir=args.download_dir, local_files_only=args.local_files_only)
    model_dir = resolve_checkpoint(args.model, revision=args.revision, **resolve)
    if not args.draft and not getattr(args, 'no_draft', False):
        args.draft = default_draft(args.model, model_dir)
        if args.draft:
            LOG.info('Automatically selected DSpark head: %s (seven proposals plus anchor)', args.draft)
    draft_dir = resolve_checkpoint(args.draft, revision=args.draft_revision, **resolve) if args.draft else None
    architecture = json.loads((model_dir/'config.json').read_text())['architectures'][0]
    cls = resolve_model(architecture)
    if cls is None:
        raise ValueError(f'No model package registered for {architecture!r}')
    # Reserve the extra block positions used by draft attention beyond the target context.
    draft = DRAFTERS.get('dspark').from_checkpoint(str(draft_dir), target_lm_head=None,
                max_context=args.max_context) if draft_dir else None
    gamma = (min(7, draft.gamma) if args.draft_block_size is None else args.draft_block_size) if draft else 0
    if draft and not 1 <= gamma <= draft.cfg.max_block_size:
        raise ValueError(f'--draft-block-size must be between 1 and {draft.cfg.max_block_size}')
    capacity = args.max_context + max(0, gamma - 1)
    model = cls.from_checkpoint(str(model_dir), max_context=capacity)
    if draft:
        draft = DRAFTERS.get('dspark').from_checkpoint(str(draft_dir), target_lm_head=model.lm_head,
                                                       max_context=capacity, block_size=gamma)
        draft.bind_target(model)
        if (draft.cfg.target_hidden != model.config.hidden_size
                or draft.cfg.hidden_size != model.config.hidden_size
                or draft.cfg.vocab_size != model.config.vocab_size or any(
                i not in model.feature_taps() for i in draft.tap_layers())):
            raise ValueError('DSpark feature taps/hidden size/vocabulary do not match the target')
    quantization = None if args.draft_quantization == 'none' else args.draft_quantization
    if quantization == 'auto':
        if args.draft_pack and (Path(args.draft_pack)/'manifest.json').exists():
            quantization = json.loads((Path(args.draft_pack)/'manifest.json').read_text()).get('quantize', {}).get('format')
        else:
            quantization = 'nvfp4' if backend.serving_recipes(model, draft, 'nvfp4') else None
    recipes = backend.serving_recipes(model, draft, quantization)
    if args.kernel_config:
        recipes = json.loads(Path(args.kernel_config).read_text())
        if 'draft' in recipes:
            recipes = {'0': recipes}
        if not recipes or any(not str(k).isdigit() or not isinstance(v, dict) for k, v in recipes.items()):
            raise ValueError('--kernel-config must contain a recipe or numeric context-to-recipe map')
    if args.kernel_config_key and args.kernel_config_key not in recipes:
        raise ValueError('--kernel-config-key is not present in the selected recipe map')
    if draft and quantization:
        if quantization != 'nvfp4':
            raise ValueError('Only native or NVFP4 draft packing is supported')
        checkpoint = SafetensorsDir(draft_dir)
        try:
            bind_formats(draft, checkpoint, requantize=quantization, keep=('markov_w1',))
        finally:
            checkpoint.close()
        LOG.info('DSpark draft format: %s (Markov embedding kept in source precision)', quantization)
    root = Path(args.pack).expanduser() if args.pack else default_pack_cache()
    layout = PackLayout(lane_order=profile.lane_order, scale_placement=profile.scale_placement)
    pack = ensure_pack(model_dir, model, root, capacity=capacity, layout=layout,
                       backend=profile.backend)
    draft_pack = None
    if draft:
        draft_root = Path(args.draft_pack).expanduser() if args.draft_pack else (
            root/'draft-cache' if (root/'manifest.json').exists() else root)
        draft_pack = ensure_pack(draft_dir, draft, draft_root, capacity=capacity, layout=layout,
                                  backend=profile.backend, role='draft', quantization=quantization)
    verify_rule, lookup = getattr(args, 'verify_rule', 'fixed') or 'fixed', bool(getattr(args, 'draft_lookup', False))
    verify_costs = backend.serving_verify_costs() if draft and hasattr(backend, 'serving_verify_costs') else None
    LOG.info('Serving backend=%s, target=%s, draft=%s, verify=%s (%s%s), recipe contexts=%s',
             profile.backend, model_dir, draft_dir, gamma + 1 if gamma else 1, verify_rule, ', lookup' if lookup else '', sorted(recipes))
    chunk = backend.serving_prefill_chunk(model, draft, recipes)
    return ServingAssets(model_dir, pack, args.max_context, capacity, profile,
                         draft_dir, draft_pack, gamma, recipes, args.kernel_config_key, prefill_chunk_size=chunk,
                         verify_rule=verify_rule, lookup=lookup, verify_costs=verify_costs,
                         spec_sampling=getattr(args, 'spec_sampling', 'match') or 'match')
