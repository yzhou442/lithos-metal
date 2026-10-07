"""Exact prefill (large chunks, the 128-row chunking's results): one-token chunks take the reference path.

The 128-row prompt graph computes a chunk of ONE token with one-row projection kernels; the large graph only has
matrix tiles (last-bit different). A request whose 128-row chunking hands the prompt graph such a chunk must run on
the reference graph with the reference chunks; every other request keeps the large chunks."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from monolith.core.step_state import StepStateLayout
from monolith.generate import Session


def fake_session(*, chunk, exact, recipe):
    session = Session.__new__(Session)
    session.decoder_kernel_config = {} if recipe else None
    session.layout = StepStateLayout(max(chunk, 8), 7)
    session.seed, session.prefill_chunk_size, session.prefill_exact = 0, chunk, exact
    session.resident_prefill_tokens, session.decode_t_max = min(chunk, 128), 8
    session.verify, session.verify_length = 'fixed', 7
    session.drafter = SimpleNamespace(gamma=7)
    session.spec_steps_per_cb, session.spec_in_flight = 1, 2
    session.reset, session.release_engines = Mock(), Mock()
    session._keep_both = Mock(return_value=True)
    session.engines = {}
    writes = []
    state = Mock(read=Mock(return_value=session.layout.pack({})),
                 write=Mock(side_effect=lambda data, offset: writes.append(session.layout.unpack(data))))

    def engine(name):
        return SimpleNamespace(name=name, program=SimpleNamespace(context_capacity=1 << 20, step_state='state'),
                               buffers={'state': state}, state=lambda: {'error': 0},
                               run=Mock(return_value=SimpleNamespace(tokens=[], gpu_ms=1.0, wall_ms=1.0, done=True)))
    large, reference, decoder = engine('large'), engine('reference'), engine('decoder')
    decoder.run.return_value = SimpleNamespace(tokens=[10], gpu_ms=1.0, wall_ms=1.0, done=True, host_busy_ms=0.0, steps=1)
    session.prefill_engine = Mock(side_effect=lambda n=None, reference=False: globals_[1] if reference else globals_[0])
    globals_ = (large, reference)
    session.engine = Mock(return_value=decoder)
    session._accept_stats = Mock(return_value=([], [], [], []))
    return session, writes, large, reference


def passes(writes):
    return [w['t_this_step'] for w in writes]


@pytest.mark.parametrize('prompt_graph_tokens', [128, 129, 130, 255, 256, 257, 512, 513, 514, 639, 641, 4224, 4225,
                                                 4226, 4351, 8320, 8321, 8322, 8447, 16512, 16513, 16514, 16639])
def test_recipe_session_takes_the_reference_path_exactly_for_one_token_chunks(prompt_graph_tokens):
    p = prompt_graph_tokens + 8                        # the verification graph ingests the last eight tokens
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=True)
    session.generate(list(range(p)), 1)
    edge = prompt_graph_tokens % 128 == 1
    assert session.last_prefill_reference is edge
    rows = 128 if edge else 512
    expected = [rows] * (prompt_graph_tokens // rows) + ([prompt_graph_tokens % rows] if prompt_graph_tokens % rows else [])
    assert passes(writes) == expected + [8]
    assert (reference.run.call_count, large.run.call_count) == ((len(expected), 0) if edge else (0, len(expected)))
    if edge:
        # the reference graph's weights cannot stay next to the other engines: released before and after it
        session._keep_both.assert_not_called()
        assert [c.kwargs.get('keep_state_from') for c in session.release_engines.call_args_list] == [None, reference]
    else:
        session.release_engines.assert_not_called()


def test_one_token_prompt_graph_chunks_between_boundaries_count_too():
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=True)
    session.prefix_cache = SimpleNamespace(match=lambda ids: None, min_tokens=128, save=Mock(), restore=Mock())
    session.generate(list(range(1000)), 1, cache_prefix_tokens=[385])          # 385 = 3 * 128 + 1
    assert session.last_prefill_reference and passes(writes) == [128, 128, 128, 1, 128, 128, 128, 128, 95, 8]
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=True)
    session.prefix_cache = SimpleNamespace(match=lambda ids: None, min_tokens=128, save=Mock(), restore=Mock())
    session.generate(list(range(1000)), 1, cache_prefix_tokens=[386])
    assert not session.last_prefill_reference and passes(writes) == [386, 512, 94, 8]


@pytest.mark.parametrize('chunk,exact', [(512, False), (128, True), (128, False)])
def test_other_modes_never_take_the_reference_path(chunk, exact):
    session, writes, large, reference = fake_session(chunk=chunk, exact=exact, recipe=True)
    session.generate(list(range(137)), 1)                                      # 129 prompt-graph tokens
    assert not session.last_prefill_reference and reference.run.call_count == 0
    assert passes(writes) == ([129, 8] if chunk == 512 else [128, 1, 8])


def test_short_prompts_stay_on_the_verification_graph():
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=True)
    session.generate(list(range(9)), 1)
    assert not session.last_prefill_reference and passes(writes) == [8, 1]
    assert large.run.call_count == reference.run.call_count == 0


def test_without_a_decoder_recipe_the_last_chunk_is_a_prompt_graph_chunk():
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=False)
    session.generate(list(range(129)), 1)
    assert session.last_prefill_reference and passes(writes) == [128, 1] and reference.run.call_count == 2
    session, writes, large, reference = fake_session(chunk=512, exact=True, recipe=False)
    session.generate(list(range(130)), 1)
    assert not session.last_prefill_reference and passes(writes) == [130] and large.run.call_count == 1
