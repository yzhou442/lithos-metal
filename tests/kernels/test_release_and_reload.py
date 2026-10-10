"""Released-and-reloaded engines and prefix checkpoints restored from disk reproduce a cold generation (hybrid
GDN + attention target with a DSpark drafter, exact chunked prefill, resident decoder)."""
import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contract'))
from test_nn_lowering import _checkpoint
from dspark_synth import write_checkpoint as write_dspark, build
from monolith.generate import Session
from monolith.formats import PackLayout
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.nn.pack_plan import pack_model
from monolith.runtime.prefix_store import PrefixStore


@pytest.fixture(scope='module')
def factory(tmp_path_factory):
    root = tmp_path_factory.mktemp('release')
    target, draft = root / 'target', root / 'draft'
    target.mkdir(); draft.mkdir()
    _checkpoint(target)
    pack_model(Qwen3_5Model.from_checkpoint(str(target), max_context=512), str(target), str(target / 'pack'),
               PackLayout(rows=16))
    write_dspark(draft, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1])

    def session(**options):
        m = Qwen3_5Model.from_checkpoint(str(target), max_context=512)
        d, _, _, _ = build(draft, target_lm_head=m.lm_head, max_context=512)
        if not (draft / 'pack').exists():
            pack_model(d, str(draft), str(draft / 'pack'), PackLayout(rows=16))
        return Session(m, str(target / 'pack'), eos=-1, autotune=False, attention='v1', accelerator='on',
                       drafter=d, drafter_pack=str(draft / 'pack'), verify='fixed', verify_length=3,
                       decoder_kernel_config={}, prefill_chunk_size=512, prefill_exact=True, **options)
    return session


def ids(seed, n):
    return np.random.default_rng(seed).integers(0, 50, n).tolist()


def test_release_and_load_cycles_reproduce_tokens_without_growth(factory):
    session = factory()
    prompts = [ids(1, 300), ids(2, 40)]
    expected = [session.generate(p, 12).tokens for p in prompts]
    sizes = None
    for _ in range(4):
        engines = [weakref.ref(e) for e in session.engines.values()]
        session.release_engines()
        gc.collect()
        assert not session.engines and session.buffers is None
        assert all(ref() is None for ref in engines)              # nothing else holds the released engines
        session.load()
        assert 0 in session.engines
        allocated = sorted((n, b.nbytes) for n, b in session.buffers.items())
        assert sizes is None or allocated == sizes                 # the same allocations every cycle
        sizes = allocated
        assert [session.generate(p, 12).tokens for p in prompts] == expected


def test_disk_checkpoints_restore_the_state_a_cold_prefill_computes(factory, tmp_path):
    reference = factory()
    system = ids(3, 200)
    turns = [system + ids(10 + k, 30) for k in range(3)]
    expected = [reference.generate(t, 12).tokens for t in turns]
    follow = turns[0] + expected[0] + ids(20, 25)                  # a second turn: prompt, answer, new message
    expected_follow = reference.generate(follow, 12).tokens
    del reference

    session = factory(prefix_cache=True)
    cache = session.prefix_cache
    cache.store = PrefixStore(tmp_path / 'store', 1 << 30, session.prefix_identity(test=True), persist=True)
    first = session.generate(turns[0], 12, cache_prefix_tokens=[200, len(turns[0]) - 1])
    assert first.tokens == expected[0] and first.cached_prompt_tokens == 0
    second = session.generate(turns[1], 12, cache_prefix_tokens=[200])
    assert second.tokens == expected[1] and second.cached_prompt_tokens == 200 and cache.last['source'] == 'memory'

    assert cache.flush(timeout=60) and len(list(cache.store.dir.glob('*.lpc'))) == 2
    cache.clear()                                                   # an idle release
    session.release_engines()
    gc.collect()
    session.load()
    third = session.generate(turns[2], 12, cache_prefix_tokens=[200])
    assert third.tokens == expected[2] and third.cached_prompt_tokens == 200 and cache.last['source'] == 'disk'
    turn2 = session.generate(follow, 12, cache_prefix_tokens=[len(turns[0]) - 1])
    assert turn2.tokens == expected_follow and turn2.cached_prompt_tokens == len(turns[0]) - 1

    # A later process with persistence enabled and the same identity starts from the stored checkpoints.
    cache.flush(timeout=60)
    identity = cache.store.identity
    restarted = factory(prefix_cache=True)
    assert restarted.prefix_identity(test=True) == identity
    restarted.prefix_cache.store = PrefixStore(tmp_path / 'store', 1 << 30, identity, persist=True)
    again = restarted.generate(turns[1], 12, cache_prefix_tokens=[200])
    assert again.tokens == expected[1] and again.cached_prompt_tokens == 200
    assert restarted.prefix_cache.last['source'] == 'disk'
