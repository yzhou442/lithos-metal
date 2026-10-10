"""Idle release: the model's lifecycle, its races with requests and what an unload frees; no GPU needed."""

import gc
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from monolith.serving.lifecycle import IdleRelease, ModelLoadError, ModelState


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def lifecycle(seconds=30, **kwargs):
    unloads = []
    clock = Clock()
    life = IdleRelease(threading.Lock(), lambda: unloads.append(clock.now), seconds, clock=clock, **kwargs)
    return life, clock, unloads


def test_idle_period_must_be_positive_seconds():
    for value in (0, -1, float('nan'), True, '5'):
        with pytest.raises(ValueError, match='positive number of seconds'):
            IdleRelease(threading.Lock(), lambda: None, value)


def test_releases_only_after_the_period_since_the_last_completion():
    life, clock, unloads = lifecycle(30)
    with life.request():
        clock.now += 100                           # a long generation: the period starts when it finishes
    clock.now += 29.9
    assert not life.release_if_idle() and life.state is ModelState.READY
    clock.now += 0.1
    assert life.release_if_idle() and unloads == [clock.now]
    assert life.state is ModelState.UNLOADED and life.unloads == 1
    assert not life.release_if_idle()              # nothing left to release
    assert unloads == [clock.now]


def test_a_new_request_restarts_the_period():
    life, clock, unloads = lifecycle(30)
    clock.now += 20
    with life.request():
        pass
    clock.now += 20
    assert not life.release_if_idle()
    clock.now += 10
    assert life.release_if_idle() and len(unloads) == 1


def test_pending_requests_keep_the_model_even_past_the_period():
    life, clock, unloads = lifecycle(30)
    context = life.request()
    context.__enter__()                            # arrived, waiting for the GPU lock or generating/streaming
    clock.now += 1000
    assert not life.release_if_idle() and unloads == []
    context.__exit__(None, None, None)
    assert life.pending == 0 and not life.release_if_idle()     # completion restarted the period
    clock.now += 30
    assert life.release_if_idle()


def test_failed_and_cancelled_requests_do_not_leak_pending_counts():
    life, clock, unloads = lifecycle(30)
    for exc in (RuntimeError('generation failed'), KeyboardInterrupt(), GeneratorExit()):
        with pytest.raises(type(exc)):
            with life.request():
                raise exc
    assert life.pending == 0
    clock.now += 30
    assert life.release_if_idle()


def test_release_waits_for_the_gpu_lock_and_rechecks():
    life, clock, unloads = lifecycle(30)
    clock.now += 31
    life.gpu_lock.acquire()                        # a generation owns the GPU
    result = []
    worker = threading.Thread(target=lambda: result.append(life.release_if_idle()))
    worker.start()
    time.sleep(0.05)
    assert worker.is_alive() and unloads == []
    with life.request():                           # a request arrived and finished while the unload waited
        pass
    life.gpu_lock.release()
    worker.join(5)
    assert result == [False] and unloads == [] and life.state is ModelState.READY


def test_load_is_single_flight_and_waiters_share_its_result():
    life, clock, unloads = lifecycle(30, state=ModelState.UNLOADED)
    loads, started, finish = [], threading.Event(), threading.Event()

    def load():
        loads.append(1)
        started.set()
        assert finish.wait(5)

    def request(results):
        with life.request() as arrived:
            with life.gpu_lock:
                life.ensure_loaded(arrived, load)
                results.append(life.state)

    results = []
    first = threading.Thread(target=request, args=(results,))
    first.start()
    assert started.wait(5)
    others = [threading.Thread(target=request, args=(results,)) for _ in range(4)]
    for thread in others:
        thread.start()
    time.sleep(0.05)
    assert life.state is ModelState.LOADING and life.pending == 5
    finish.set()
    for thread in (first, *others):
        thread.join(5)
    assert loads == [1] and results == [ModelState.READY] * 5 and life.loads == 1 and life.pending == 0


def test_a_failed_load_fails_its_waiters_and_a_later_request_retries():
    life, clock, unloads = lifecycle(30, state=ModelState.UNLOADED)
    started, finish = threading.Event(), threading.Event()
    attempts = []

    def failing():
        attempts.append('fail')
        started.set()
        assert finish.wait(5)
        raise FileNotFoundError('pack shard missing')

    outcomes = []

    def request(load):
        try:
            with life.request() as arrived:
                with life.gpu_lock:
                    life.ensure_loaded(arrived, load)
            outcomes.append('ok')
        except ModelLoadError as exc:
            outcomes.append(str(exc))

    first = threading.Thread(target=request, args=(failing,))
    first.start()
    assert started.wait(5)
    waiter = threading.Thread(target=request, args=(lambda: attempts.append('waiter'),))
    waiter.start()
    time.sleep(0.05)
    finish.set()
    first.join(5)
    waiter.join(5)
    assert attempts == ['fail'] and len(outcomes) == 2
    assert all('pack shard missing' in o for o in outcomes)
    assert life.state is ModelState.FAILED and isinstance(life.error, FileNotFoundError)
    assert unloads, 'a failed load releases what it allocated'
    request(lambda: attempts.append('retry'))      # arrived after the failure: tries again
    assert outcomes[-1] == 'ok' and attempts[-1] == 'retry' and life.state is ModelState.READY
    assert life.error is None and life.pending == 0


def test_a_request_during_an_unload_waits_and_loads_again():
    unloading, finish = threading.Event(), threading.Event()
    clock = Clock()

    def unload():
        unloading.set()
        assert finish.wait(5)

    life = IdleRelease(threading.Lock(), unload, 30, clock=clock)
    clock.now += 30
    releaser = threading.Thread(target=life.release_if_idle)
    releaser.start()
    assert unloading.wait(5) and life.state is ModelState.UNLOADING
    loads, seen = [], []

    def request():
        with life.request() as arrived:
            with life.gpu_lock:
                seen.append(life.state)
                life.ensure_loaded(arrived, lambda: loads.append(1))

    worker = threading.Thread(target=request)
    worker.start()
    time.sleep(0.05)
    assert worker.is_alive() and loads == []       # it waits for the unload instead of using freed buffers
    finish.set()
    releaser.join(5)
    worker.join(5)
    assert seen == [ModelState.UNLOADED] and loads == [1] and life.state is ModelState.READY


def test_watcher_thread_releases_on_the_real_clock_and_stops():
    unloaded = threading.Event()
    life = IdleRelease(threading.Lock(), unloaded.set, 0.05).start()
    try:
        assert unloaded.wait(5) and life.state is ModelState.UNLOADED
    finally:
        life.stop()
    life._thread.join(5)
    assert not life._thread.is_alive()


def test_a_period_longer_than_a_timer_can_wait_keeps_the_watcher_alive():
    life = IdleRelease(threading.Lock(), lambda: None, 1e12).start()
    time.sleep(0.05)
    try:
        assert life._thread.is_alive() and life.state is ModelState.READY
    finally:
        life.stop()
    life._thread.join(5)


# Backend integration: what an unload frees and how requests see the lifecycle.

class Engine:
    def __init__(self, size):
        self.buffer = bytearray(size)


class Session:
    def __init__(self, log):
        self.log, self.engines, self.buffers = log, {}, None
        self.prefix_cache = SimpleNamespace(items=[b'snapshot'], clear=lambda: self.prefix_cache.items.clear())
        self.eos = 0

    def load(self):
        self.log.append('load')
        self.engines.update({0: Engine(1 << 20), 'prefill.512': Engine(1 << 20)})
        self.buffers = {'weights': SimpleNamespace(nbytes=3 << 20), 'state': SimpleNamespace(nbytes=1 << 20)}

    def release_engines(self):
        self.log.append('release')
        self.engines.clear()
        self.buffers = None

    def generate(self, ids, limit, **options):
        self.log.append('generate' if self.engines else 'generate-unloaded')
        return SimpleNamespace(tokens=[1, 0])


def backend(log):
    from monolith.serve import Backend
    b = Backend.__new__(Backend)
    b.session, b._sessions, b.assets, b.max_context = Session(log), {}, None, 4096
    b.select_session = lambda request, n: log.append('select')
    b.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [5, 6, 7],
                                  decode=lambda *a, **kw: 'Hi')
    b.session.load()
    log.clear()
    return b


def test_unload_drops_every_engine_buffer_and_snapshot_but_keeps_programs():
    log = []
    b = backend(log)
    cached = Session(log)
    cached.load()
    b._sessions = {'other': cached}
    refs = [weakref.ref(e) for s in (b.session, cached) for e in s.engines.values()]
    log.clear()
    assert b.loaded and b.resident_bytes == 4 << 20
    b.unload()
    gc.collect()
    assert log == ['release', 'release'] and not b.loaded and b.resident_bytes == 0
    assert all(ref() is None for ref in refs)
    assert b.session.prefix_cache.items == []


def test_repeated_cycles_do_not_accumulate_engines():
    log = []
    b = backend(log)
    refs = []
    for _ in range(50):
        b.session.load()
        refs += [weakref.ref(e) for e in b.session.engines.values()]
        b.unload()
    gc.collect()
    assert sum(ref() is not None for ref in refs) == 0


def test_requests_load_after_an_idle_release_and_keep_the_api(monkeypatch):
    from fastapi.testclient import TestClient
    from monolith.serve import create_app
    log = []
    b = backend(log)
    life = b.enable_idle_release(3600)
    life.stop()
    client = TestClient(create_app(b, 'test-model'))
    assert client.get('/health').json() == {'status': 'ok', 'model': 'ready'}
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 4}
    assert client.post('/v1/chat/completions', json=body).status_code == 200
    assert log == ['select', 'generate']
    life.last_active -= 3600
    assert life.release_if_idle() and not b.loaded
    assert client.get('/health').json()['model'] == 'unloaded'
    log.clear()
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200 and response.json()['choices'][0]['message']['content'] == 'Hi'
    assert log == ['select', 'load', 'generate'] and life.state is ModelState.READY and life.pending == 0


def test_a_failed_load_is_a_clear_503_and_the_next_request_retries():
    from fastapi.testclient import TestClient
    from monolith.serve import create_app
    log = []
    b = backend(log)
    life = b.enable_idle_release(3600)
    life.stop()
    life.last_active -= 3600
    assert life.release_if_idle()
    load = b.session.load
    b.session.load = lambda: (_ for _ in ()).throw(OSError('weights missing'))
    client = TestClient(create_app(b, 'test-model'))
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 4}
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'model_load_failed' and 'weights missing' in response.json()['error']['message']
    events = client.post('/v1/chat/completions', json={**body, 'stream': True}).text
    assert 'model_load_failed' in events                         # a stream reports it as an error event
    assert life.pending == 0 and life.state is ModelState.FAILED
    b.session.load = load
    assert client.post('/v1/chat/completions', json=body).status_code == 200
    assert life.state is ModelState.READY


def test_without_the_option_requests_and_health_are_unchanged():
    from fastapi.testclient import TestClient
    from monolith.serve import create_app, parse_args
    assert parse_args(['--model', 'org/target', '--no-draft']).model_ttl is None
    assert parse_args(['--model', 'org/target', '--no-draft', '--model-ttl', '300']).model_ttl == 300
    for value in ('0', '-1', 'inf', 'nan'):
        with pytest.raises(SystemExit):
            parse_args(['--model', 'org/target', '--no-draft', '--model-ttl', value])
    log = []
    b = backend(log)
    assert b.lifecycle is None
    client = TestClient(create_app(b, 'test-model'))
    assert client.get('/health').json() == {'status': 'ok'}
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 4}
    assert client.post('/v1/chat/completions', json=body).status_code == 200
    assert log == ['select', 'generate']
