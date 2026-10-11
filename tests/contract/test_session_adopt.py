"""Sessions of one model hand their state allocations to the next selected session (no GPU needed)."""
import threading
from types import SimpleNamespace

from monolith.generate import Session
from monolith.runtime.program import BufferSpec, Program


class Buffer:
    def __init__(self, nbytes):
        self.nbytes, self.cleared = nbytes, False

    def fill(self, value):
        self.cleared = True


def session(buffers, specs, kv=()):
    s = Session.__new__(Session)
    s.buffers, s._programs, s._kv_buffers, s.engines = buffers, {0: Program({}, specs, [])}, set(kv), {}
    return s


def test_a_session_adopts_same_states_scratch_and_weight_windows(tmp_path):
    (tmp_path / 'w').write_bytes(bytes(16384))
    window = dict(role='weights', file=str(tmp_path / 'w'))
    specs = {'k_cache': BufferSpec(64, role='state'), 'gdn': BufferSpec(32, role='state'),
             'ring': BufferSpec(16, role='ring'), 'ws': BufferSpec(16), 'w': BufferSpec(16384, **window),
             'moved': BufferSpec(8192, **window, file_offset=8192), 'resized': BufferSpec(64, role='state'),
             'params': BufferSpec(8, bytes(8), 'params'), 'mine': BufferSpec(8, role='state')}
    previous = {name: spec for name, spec in specs.items() if name != 'mine'} | dict(
        resized=BufferSpec(32, role='state'), ws=BufferSpec(32), moved=BufferSpec(8192, **window),
        theirs=BufferSpec(8, role='state'))
    buffers = {name: Buffer(spec.nbytes) for name, spec in previous.items()}
    old, new = session(buffers, previous), session(None, specs, kv={'k_cache'})
    new.adopt_buffers(old)
    assert set(new.buffers) == {'k_cache', 'gdn', 'ring', 'ws', 'w'}
    assert all(new.buffers[name] is buffers[name] for name in new.buffers)
    # like reset(): every adopted state is cleared except the KV caches, whose rows are written before they are read
    assert (buffers['k_cache'].cleared, buffers['gdn'].cleared, buffers['ring'].cleared) == (False, True, True)
    assert not buffers['ws'].cleared and not buffers['w'].cleared


def test_a_session_with_allocations_keeps_its_own():
    specs = {'gdn': BufferSpec(32, role='state')}
    old = session({'gdn': Buffer(32)}, specs)
    held = session({'own': Buffer(8)}, specs)
    held.adopt_buffers(old)
    assert set(held.buffers) == {'own'}


def test_a_new_session_compiles_its_decoder_program_and_adopts():
    specs = {'gdn': BufferSpec(32, role='state'), 'ws': BufferSpec(16)}
    old = session({'gdn': Buffer(32), 'ws': Buffer(16)}, specs)
    for drafter, key, bound, dynamic in ((object(), 0, 9, True), (None, 1, 1, False)):
        fresh, compiled = session(None, specs), []
        fresh._programs, fresh.drafter, fresh.decode_t_max = {}, drafter, 9
        fresh._compile = lambda bound, **kw: compiled.append((bound, kw)) or Program({}, specs, [])
        fresh.adopt_buffers(old)                # the first switch to a variant: no compiled programs yet
        assert compiled == [(bound, dict(dynamic=dynamic, prefill=False))] and list(fresh._programs) == [key]
        assert set(fresh.buffers) == {'gdn', 'ws'}


def test_adoption_waits_for_a_decoder_still_mapping_in_the_background():
    specs = {'gdn': BufferSpec(32, role='state'), 'w': BufferSpec(16)}
    old, new = session({'gdn': Buffer(32)}, specs), session(None, specs)
    started, finish = threading.Event(), threading.Event()

    def map_decoder():                          # a cancelled request's prefetch, still adding the decoder's buffers
        started.set()
        finish.wait(5)
        old.buffers['w'] = Buffer(16)
    old._prefetch = (threading.Thread(target=map_decoder), [RuntimeError('unused')])
    old._prefetch[0].start()
    started.wait(5)
    threading.Timer(0.05, finish.set).start()
    new.adopt_buffers(old)                      # joins the prefetch first, without raising its error
    assert old._prefetch is None and set(new.buffers) == {'gdn', 'w'}


def test_switching_sessions_adopts_before_releasing(monkeypatch):
    from pathlib import Path
    from monolith import generate
    from monolith.backends.metal import load_configs
    from monolith.serve import Backend
    from monolith.serving.setup import ServingAssets
    events = []

    class Fake:
        def __init__(self, name):
            self.name, self.dev, self._pipelines = name, 'device', {}

        def adopt_buffers(self, other):
            events.append(('adopt', self.name, other.name))

        def release_engines(self):
            events.append(('release', self.name))
    created = iter(Fake(name) for name in ('greedy', 'sampled'))
    monkeypatch.setattr(generate, 'load_session', lambda *args, **kwargs: next(created))
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 4096
    backend.prefill_chunk_size, backend.prefill_exact = 128, False
    backend.session, backend.sampling = None, None
    backend.assets = ServingAssets(Path('m'), Path('p'), 4096, 4102, load_configs()['apple-m5-max-40c'],
                                   Path('draft'), Path('draft-pack'), 7)
    request = lambda temperature: SimpleNamespace(temperature=temperature, top_p=1.0, top_k=0, seed=None)
    backend.select_session(request(0.0), 100)
    backend.select_session(request(0.7), 100)
    backend.select_session(request(0.0), 100)
    assert events == [('adopt', 'sampled', 'greedy'), ('release', 'greedy'),
                      ('adopt', 'greedy', 'sampled'), ('release', 'sampled')]
