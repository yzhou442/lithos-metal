"""Ollama chat adapter and idle unloading, without GPU weights."""
import json
import math
import threading
import time
from types import SimpleNamespace

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from monolith.serve import Backend, create_app, parse_args
from monolith.serving.protocol import APIError, keep_alive_seconds, ollama_request

TOOL = {'type': 'function', 'function': {'name': 'read_file', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'}}, 'required': ['path']}}}
XML = '<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>'


def chat(**fields):
    return {'model': 'local', 'messages': [{'role': 'system', 'content': 'Fix the transcript.'},
                                           {'role': 'user', 'content': 'helo world'}], **fields}


def make_client(text='Hello world.', served='local'):
    seen = []
    def complete(request, **_):
        seen.append(request)
        return text, 'stop', 12, 3
    backend = SimpleNamespace(complete=complete, last_metrics={'prefill_wall_ms': 20.0, 'wall_ms': 60.0, 'setup_ms': 5.0,
                                                               'decode_wall_ms': 30.0})
    return TestClient(create_app(backend, served)), seen


def lines(response):
    assert response.status_code == 200, response.text
    assert response.headers['content-type'].startswith('application/x-ndjson')
    return [json.loads(line) for line in response.text.splitlines()]


class Model:
    """A backend whose residency the keep-alive manager drives."""
    resident_bytes = 1 << 30

    def __init__(self):
        self.loaded, self.events, self.last_metrics = True, [], {}

    def complete(self, request, **_):
        self.loaded = True
        self.events.append('generate')
        return 'ok', 'stop', 3, 1

    def unload(self):
        self.loaded = False
        self.events.append('unload')

    def load(self):
        self.loaded = True
        self.events.append('load')


def wait_for(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_voxt_request():
    client, seen = make_client()
    assert client.head('/api/chat').status_code == 200
    body = chat(stream=False, think=False, keep_alive='5m', options={'temperature': 0, 'top_p': 1, 'num_predict': 256})
    data = client.post('/api/chat', json=body).json()
    assert data['message'] == {'role': 'assistant', 'content': 'Hello world.'}
    assert data['model'] == 'local' and data['done'] is True and data['done_reason'] == 'stop'
    assert data['created_at'].endswith('Z')
    assert (data['prompt_eval_count'], data['eval_count']) == (12, 3)
    assert (data['load_duration'], data['prompt_eval_duration'], data['eval_duration']) == (5_000_000, 20_000_000, 30_000_000)
    assert data['total_duration'] > 0
    request = seen[-1]
    assert (request.temperature, request.top_p, request.token_limit, request.stream) == (0, 1, 256, False)
    assert request.template_inputs()[0] == [{'role': 'system', 'content': 'Fix the transcript.'},
                                            {'role': 'user', 'content': 'helo world'}]


def test_body_is_json_without_a_content_type():
    client, seen = make_client()
    response = client.post('/api/chat', content=json.dumps(chat(stream=False)), headers={'Content-Type': ''})
    assert response.status_code == 200 and response.json()['message']['content'] == 'Hello world.'
    for body in ('{"model": ', '[1]'):
        response = client.post('/api/chat', content=body)
        assert response.status_code == 400 and isinstance(response.json()['error'], str)


def test_streams_ndjson_by_default():
    client, _ = make_client()
    chunks = lines(client.post('/api/chat', json=chat()))
    assert ''.join(chunk['message']['content'] for chunk in chunks) == 'Hello world.'
    assert [chunk['done'] for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert chunks[-1]['done_reason'] == 'stop' and chunks[-1]['eval_count'] == 3


def test_options():
    request = ollama_request(chat(model='local:latest', options={
        'temperature': 0.7, 'top_p': 0.9, 'top_k': 20, 'seed': 7, 'stop': ['\n\n'], 'num_ctx': 8192, 'num_thread': 8,
        'repeat_penalty': 1, 'presence_penalty': 0}), 'local')
    assert (request.model, request.temperature, request.top_p, request.top_k, request.seed, request.stop) == (
        'local', 0.7, 0.9, 20, 7, ['\n\n'])
    assert request.stream and request.token_limit is None                     # Ollama's defaults: stream, no limit
    assert ollama_request(chat(options={'num_predict': -2, 'seed': -1})).token_limit is None
    assert ollama_request(chat(options={'seed': -1})).seed == 0


@pytest.mark.parametrize('served', ['local', 'local:latest'])
def test_latest_is_the_same_model(served):
    client, seen = make_client(served=served)
    for name in ('local', 'local:latest'):
        assert client.post('/api/chat', json=chat(model=name, stream=False)).status_code == 200
        assert seen[-1].model == served
    assert client.get('/api/tags').json()['models'][0]['name'] == served


@pytest.mark.parametrize('fields', [
    {'format': 'json'}, {'format': {'type': 'object'}}, {'think': True}, {'think': 'high'}, {'logprobs': True},
    {'options': {'repeat_penalty': 1.1}}, {'options': {'min_p': 0.05}}, {'options': {'num_predict': 0}}, {'options': 'fast'},
    {'messages': [{'role': 'user', 'content': 'Describe this', 'images': ['aGk=']}]}, {'keep_alive': 'soon'},
])
def test_rejects_what_it_cannot_honor(fields):
    def unexpected(*_args, **_options):
        pytest.fail('Invalid request reached the GPU backend')
    client = TestClient(create_app(SimpleNamespace(complete=unexpected), 'local'))
    response = client.post('/api/chat', json=chat(stream=False, **fields))
    assert response.status_code == 400 and isinstance(response.json()['error'], str)
    assert client.post('/api/chat', json=chat(model='other')).status_code == 404


def test_tool_calls_and_results():
    client, seen = make_client(XML)
    body = chat(stream=False, tools=[TOOL])
    message = client.post('/api/chat', json=body).json()['message']
    assert message['tool_calls'] == [{'function': {'name': 'read_file', 'arguments': {'path': 'a.py', 'limit': 10}}}]
    body['messages'] += [message, {'role': 'tool', 'tool_name': 'read_file', 'content': 'contents'}]
    chunks = lines(client.post('/api/chat', json={**body, 'stream': True}))
    assert chunks[-2]['message']['tool_calls'] == message['tool_calls'] and chunks[-1]['done']
    messages, _ = seen[-1].template_inputs()
    call = messages[-2]['tool_calls'][0]
    assert call['function']['arguments'] == {'path': 'a.py', 'limit': 10}
    assert messages[-1] == {'role': 'tool', 'content': 'contents', 'tool_call_id': call['id'], 'name': 'read_file'}


def test_requests_wait_for_the_model():
    entered, release = threading.Event(), threading.Event()
    def complete(request, **_):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return 'ok', 'stop', 1, 1
    client = TestClient(create_app(SimpleNamespace(complete=complete), 'local'))
    first = threading.Thread(target=lambda: client.post('/api/chat', json=chat(stream=False)))
    first.start()
    try:
        assert entered.wait(5)
        busy = {'model': 'local', 'messages': [{'role': 'user', 'content': 'hi'}]}
        assert client.post('/v1/chat/completions', json=busy).status_code == 429
        queued = []
        second = threading.Thread(target=lambda: queued.append(client.post('/api/chat', json=chat(stream=False))))
        second.start()
        time.sleep(.2)
        assert not queued                                                     # Ollama clients queue
    finally:
        release.set()
        first.join(5)
    second.join(5)
    assert queued[0].status_code == 200


def test_keep_alive_unloads_when_idle_and_a_probe_reloads():
    model = Model()
    client = TestClient(create_app(model, 'local'))
    client.post('/v1/chat/completions', json={'model': 'local', 'messages': [{'role': 'user', 'content': 'hi'}]})
    loaded = client.get('/api/ps').json()['models']
    assert loaded[0]['name'] == 'local' and loaded[0]['size_vram'] == 1 << 30 and loaded[0]['expires_at'] is None
    client.post('/api/chat', json=chat(stream=False, keep_alive=0))
    wait_for(lambda: not model.loaded)
    assert client.get('/api/ps').json() == {'models': []}
    assert client.get('/api/tags').json()['models'][0]['name'] == 'local'
    client.head('/api/chat')
    wait_for(lambda: model.loaded)
    time.sleep(.2)
    assert model.loaded and model.events[-2:] == ['unload', 'load']          # held for the request a probe announces
    client.post('/api/chat', json=chat(stream=False, keep_alive='300ms'))
    assert client.get('/api/ps').json()['models'][0]['expires_at'].endswith('Z')
    wait_for(lambda: not model.loaded)
    assert model.events[-2:] == ['generate', 'unload']


def test_requests_wait_for_a_load_in_progress():
    model, started = Model(), threading.Event()
    model.loaded = False
    def load():
        started.set()
        time.sleep(.3)
        model.loaded = True
        model.events.append('load')
    model.load = load
    client = TestClient(create_app(model, 'local'))
    client.head('/api/chat')
    client.head('/api/chat')                                                  # a second probe while loading
    assert started.wait(5)
    response = client.post('/v1/chat/completions', json={'model': 'local', 'messages': [{'role': 'user', 'content': 'hi'}]})
    assert response.status_code == 200 and model.events == ['load', 'generate']


def test_server_keep_alive_and_load_requests():
    model = Model()
    client = TestClient(create_app(model, 'local', keep_alive=0.2))
    wait_for(lambda: not model.loaded)                                        # idle since startup
    assert client.post('/api/chat', json={'model': 'local:latest'}).json()['done_reason'] == 'load'
    assert model.loaded
    assert client.post('/api/chat', json={'model': 'local', 'messages': [], 'keep_alive': 0}).json()['done_reason'] == 'unload'
    assert not model.loaded
    assert client.post('/api/chat', json={'model': 'other'}).status_code == 404


def test_keep_alive_durations():
    assert keep_alive_seconds('5m') == 300 and keep_alive_seconds('1h30m') == 5400 and keep_alive_seconds('1.5s') == 1.5
    assert keep_alive_seconds('300ms') == pytest.approx(.3) and keep_alive_seconds(600) == 600
    assert keep_alive_seconds('0') == keep_alive_seconds(0) == 0
    assert keep_alive_seconds(-1) == keep_alive_seconds('-1m') == math.inf
    for value in ('soon', '5 m', 'm', '', 'nan', True, None, [1]):
        with pytest.raises(APIError):
            keep_alive_seconds(value)
    parse = lambda *flags: parse_args(['--model', 'org/target', *flags]).keep_alive
    assert parse() == math.inf and parse('--keep-alive', '5m') == 300 and parse('--keep-alive', '0') == 0
    with pytest.raises(SystemExit):
        parse('--keep-alive', 'soon')


class Session:
    def __init__(self):
        self.engines, self.buffers, self.generated = {0: object()}, {'w': SimpleNamespace(nbytes=4)}, []

    def release_engines(self):
        self.engines.clear()

    def generate(self, ids, n, on_tokens=None, cancelled=None, **_):
        self.generated.append(n)
        self.engines[0] = object()
        tokens = []
        while len(tokens) < n and not (cancelled and cancelled()):
            tokens.append(10 + len(tokens))                                   # one token per round
            if on_tokens:
                on_tokens(list(tokens))
        return SimpleNamespace(tokens=tokens)


def test_backend_unload_keeps_programs_and_load_maps_the_current_session():
    backend = Backend.__new__(Backend)
    backend.session, backend._sessions = Session(), {'other': Session()}
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])
    assert backend.loaded and backend.resident_bytes == 4
    backend.unload()
    assert not backend.loaded and not backend._sessions['other'].engines and backend.resident_bytes == 0
    backend.load()
    assert backend.loaded and backend.session.generated == [1]


def test_unlimited_output_stops_at_the_context_capacity():
    session = Session()
    session.eos = 99
    backend = Backend.__new__(Backend)
    backend.max_context, backend.session, backend.sampling = 8, session, (0.0, 1.0, 0, 0, None)
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3], decode=lambda *a, **kw: 'x')
    backend.complete(ollama_request(chat(stream=False)))
    assert session.generated == [6]                                           # 3 prompt + 6 new - 1 = capacity 8


def test_stop_strings_end_a_non_streaming_generation():
    session = Session()
    session.eos = 99
    backend = Backend.__new__(Backend)
    backend.max_context, backend.session, backend.sampling = 64, session, (0.0, 1.0, 0, 0, None)
    decode = lambda ids, **kw: ''.join({10: 'Hello', 11: ' world', 12: '\n\n', 13: 'More'}.get(i, '!') for i in ids)
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3], decode=decode)
    content, finish, _, completion = backend.complete(ollama_request(chat(stream=False, options={'stop': ['\n\n']})))
    assert (content, finish, completion) == ('Hello world', 'stop', 3)                # not 62 tokens of context


def test_an_expiry_that_loses_the_race_to_a_request_keeps_the_model():
    from monolith.serve import KeepAlive
    model, lock = Model(), threading.Lock()
    keeper = KeepAlive(model, lock)
    epoch = keeper.epoch                                                      # the expiry thread read this deadline
    keeper.touch(60)                                                          # a request finished and restarted it
    keeper.run(lambda: keeper._unload(epoch))
    assert model.loaded
    keeper.run(lambda: keeper._unload(keeper.epoch))
    assert not model.loaded


def test_load_without_warmup_starts_the_default_session(monkeypatch):
    from monolith import generate
    calls = []
    def load_session(*args, **kwargs):
        calls.append(kwargs)
        return Session()
    monkeypatch.setattr(generate, 'load_session', load_session)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 64
    backend.prefill_chunk_size, backend.prefill_exact = 128, False
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])
    assert not backend.loaded
    backend.load()
    assert backend.loaded and backend.session.generated == [1] and calls[0]['temperature'] == 0


def test_an_expiry_during_a_request_unloads_after_it():
    from monolith.serve import KeepAlive
    model, lock = Model(), threading.Lock()
    keeper = KeepAlive(model, lock)
    with lock:                                                                # a keep_alive 0 request finishing
        keeper.touch(0)
        time.sleep(.1)                                                        # the expiry thread is waiting for the model
        assert model.loaded
    wait_for(lambda: not model.loaded)


def test_a_request_after_a_probe_sets_the_period():
    model = Model()
    model.loaded = False
    def load():
        time.sleep(.2)
        model.loaded = True
        model.events.append('load')
    model.load = load
    client = TestClient(create_app(model, 'local'))
    client.head('/api/chat')
    time.sleep(.05)                                                           # loading; the request queues behind it
    assert client.post('/api/chat', json=chat(stream=False, keep_alive=0)).status_code == 200
    wait_for(lambda: not model.loaded)                                        # not held for the probe's minute
    assert model.events == ['load', 'generate', 'unload']


def test_a_load_restarts_the_period_before_an_expiry_waiting_for_the_model():
    from monolith.serve import KeepAlive
    model, lock = Model(), threading.Lock()
    model.loaded = False
    keeper = KeepAlive(model, lock)
    with lock:                                                                # the load request owns the model
        keeper.touch(0)                                                       # an older period expires meanwhile
        time.sleep(.1)                                                        # its expiry waits for the model
        keeper.load(60)
    time.sleep(.2)
    assert model.loaded and model.events == ['load']
