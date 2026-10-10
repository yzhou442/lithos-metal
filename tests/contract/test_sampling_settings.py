"""One compiled sampling program serves every sampling setting: the samplers' parameter records are rewritten in
place (Session.set_sampling); no GPU needed."""
import itertools
from types import SimpleNamespace

import pytest

from monolith import kernels
from monolith.generate import Session
from monolith.nn.sampler import GreedySampler, StochasticSampler
from monolith.runtime.program import BufferSpec


def test_resampled_params_equal_freshly_compiled_ones():
    for (vocab, rows, n_sg), min_p, in_topk in itertools.product(((248320, 8, 12), (151936, 1, 40)), (0.0, 0.05), (False, True)):
        compiled = kernels.sample_params(vocab=vocab, t_active=rows, n_sg=n_sg, top_k=0, temperature=0.7, top_p=0.9,
                                         min_p=min_p, seed=42, topp_in_topk=in_topk)
        for temperature, top_k, top_p, seed in ((0.2, 0, 0.9, 0), (1.0, 20, 1.0, 7), (1.5, 1, 0.5, 2**40 + 3)):
            assert kernels.resample_params(compiled, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed) == \
                kernels.sample_params(vocab=vocab, t_active=rows, n_sg=n_sg, top_k=top_k, temperature=temperature,
                                      top_p=top_p, min_p=min_p, seed=seed, topp_in_topk=in_topk)


class Buffer:
    def __init__(self, data):
        self.data = data

    def write(self, data, offset):
        assert offset == 0
        self.data = data


def sampling_session(sampler):
    sample = kernels.sample_params(vocab=1000, t_active=8, n_sg=12, temperature=0.7, top_p=0.9, seed=42)
    draft_q = kernels.sample_params(vocab=1000, t_active=1, n_sg=12, temperature=0.7)
    gemv = bytes(range(16))
    prog = SimpleNamespace(buffers={'params.D8.sample.7': BufferSpec(len(sample), sample, 'params'),
                                    'params.D8.draft_q.3': BufferSpec(len(draft_q), draft_q, 'params'),
                                    'params.D8.gemv.0': BufferSpec(len(gemv), gemv, 'params'),
                                    'k_cache': BufferSpec(64, role='state')})
    other = SimpleNamespace(buffers={'params.D512.sample.9': BufferSpec(len(sample), sample, 'params')})
    s = Session.__new__(Session)
    s.model, s.drafter, s.seed = SimpleNamespace(sampler=sampler), SimpleNamespace(sampling=0.7), 42
    s._programs = {0: prog, 'prefill.512': other}
    s.engines = {0: SimpleNamespace(program=prog, buffers={n: Buffer(spec.init) for n, spec in prog.buffers.items()})}
    s._sampling = (0.7, 0, 0.9, 42)
    return s, prog, other


def test_set_sampling_rewrites_the_samplers_records_and_nothing_else():
    s, prog, other = sampling_session(StochasticSampler(0.7, 0, 0.9, 0.0, 42, prefix='sampler.'))
    gemv = prog.buffers['params.D8.gemv.0'].init
    s.set_sampling(0.2, 20, 0.95, 7)
    want = kernels.sample_params(vocab=1000, t_active=8, n_sg=12, top_k=20, temperature=0.2, top_p=0.95, seed=7)
    assert prog.buffers['params.D8.sample.7'].init == want                       # the program, for later engines
    assert s.engines[0].buffers['params.D8.sample.7'].data == want               # and the live engine
    assert other.buffers['params.D512.sample.9'].init == kernels.sample_params(
        vocab=1000, t_active=8, n_sg=12, top_k=20, temperature=0.2, top_p=0.95, seed=7)
    assert s.engines[0].buffers['params.D8.draft_q.3'].data == kernels.sample_params(
        vocab=1000, t_active=1, n_sg=12, temperature=0.2, seed=7)
    assert prog.buffers['params.D8.gemv.0'].init == gemv and s.engines[0].buffers['params.D8.gemv.0'].data == gemv
    assert (s.seed, s.drafter.sampling, s.model.sampler.temperature, s.model.sampler.top_k) == (7, 0.2, 0.2, 20)


def test_a_greedy_session_does_not_take_sampling_settings():
    s, _, _ = sampling_session(GreedySampler(prefix='sampler.'))
    with pytest.raises(ValueError):
        s.set_sampling(0.2, 0, 0.9, 0)


def test_set_sampling_waits_for_a_decoder_still_compiling():
    import threading
    s, prog, _ = sampling_session(StochasticSampler(0.7, 0, 0.9, 0.0, 42, prefix='sampler.'))
    late = SimpleNamespace(buffers={'params.D8.sample.11': BufferSpec(len(prog.buffers['params.D8.sample.7'].init),
                                                                      prog.buffers['params.D8.sample.7'].init, 'params')})
    started, finish = threading.Event(), threading.Event()

    def map_decoder():                          # a cancelled request's decoder, compiled with the old settings
        started.set()
        finish.wait(5)
        s._programs['late'] = late
    s._prefetch = (threading.Thread(target=map_decoder), [])
    s._prefetch[0].start()
    started.wait(5)
    threading.Timer(0.05, finish.set).start()
    s.set_sampling(0.2, 20, 0.95, 7)            # rewrites that program too
    assert late.buffers['params.D8.sample.11'].init == kernels.sample_params(
        vocab=1000, t_active=8, n_sg=12, top_k=20, temperature=0.2, top_p=0.95, seed=7)

