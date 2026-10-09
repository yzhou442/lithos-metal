"""lithos-metal text/tool server: Chat Completions, Responses, Anthropic Messages and Ollama chat."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import queue
import logging
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__

from .serving.protocol import (APIError, ChatRequest, Message, TextPart, anthropic_request, keep_alive_seconds,
                               ollama_model, ollama_request, responses_request, parse_completion, streaming_text)
from .serving.events import WireResponse, ollama_time
from .serving.tool_stream import tool_prefixes


class Backend:
    """One cached session; sampling changes rebuild its compiled programs, never duplicate model residency."""

    prefill_exact = False

    def __init__(self, model_dir, pack_dir, max_context=4096, prefill_chunk_size=128, *, assets=None, prefill_exact=False):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("The checkpoint must provide a chat template")
        self.model_dir, self.pack_dir, self.max_context = model_dir, pack_dir, max_context
        self.prefill_chunk_size, self.prefill_exact = prefill_chunk_size, prefill_exact
        self.session, self.sampling = None, None
        self._sessions = {}
        self.assets = assets
        self.last_metrics = {}

    def warmup(self, model_name):
        """Compile and page in the default generation path before accepting traffic."""
        logging.getLogger(__name__).info('Warming generation kernels and model weights…')
        request = ChatRequest(model=model_name, messages=[Message(role='user', content='Hello')], max_tokens=2)
        assets = getattr(self, 'assets', None)
        contexts = sorted(int(k) for k in assets.recipes) if assets and assets.recipes else [1]
        if assets and assets.recipe_key:
            contexts = [int(assets.recipe_key)]
        else:
            contexts = [k for k in contexts if k <= self.max_context] or contexts[:1]
        for context in contexts:
            logging.getLogger(__name__).info('Compiling serving context recipe %s', context)
            self.select_session(request, context)
            self.session.prepare()
        # Exercise both initial allocation and reuse/reset, including the
        # incremental text path that the first real streaming request uses.
        for _ in range(2):
            self.complete(request, on_text=lambda text: None)
        self.last_metrics = {}

    def select_session(self, request, prompt_tokens):
        from .generate import load_session

        assets = getattr(self, 'assets', None)
        recipe_key, options = assets.options(prompt_tokens) if assets else (None, {'max_context': self.max_context})
        sampling = (request.temperature, request.top_p, request.top_k, request.seed, recipe_key)
        if self.session is not None and sampling == self.sampling:
            return
        # Keep CPU programs for a bounded number of recipe/sampling variants.
        # Only the selected session retains GPU buffers; all share executable
        # pipelines on the same device and exact-token sequence checkpoints.
        sessions = getattr(self, '_sessions', {})
        self._sessions = sessions
        prefix_cache = getattr(self.session, 'prefix_cache', None)
        if self.session is not None:
            if hasattr(self.session, '_pipelines'):
                options.update(device=self.session.dev, pipeline_cache=self.session._pipelines)
                self.session.release_engines()
            sessions[self.sampling] = self.session
        self.session = sessions.pop(sampling, None)
        if self.session is None:
            self.session = load_session(self.model_dir, self.pack_dir, **options,
                temperature=request.temperature, top_p=request.top_p, top_k=request.top_k, seed=request.seed,
                autotune=False, prefill_chunk_size=self.prefill_chunk_size, prefix_cache=True,
                # exact chunks follow the 128-row chunking, its reusable prefixes included
                prefix_cache_min_tokens=min(self.prefill_chunk_size, 128) if self.prefill_exact else self.prefill_chunk_size,
                prefill_exact=self.prefill_exact)
        if prefix_cache is not None:
            self.session.prefix_cache = prefix_cache
        while len(sessions) > 8:
            sessions.pop(next(iter(sessions)))
        self.sampling = sampling

    def complete(self, request, *, on_text=None, on_content=None, on_start=None, cancelled=None, on_progress=None):
        """on_progress(content, completion_tokens) pairs each content snapshot with the committed output count."""
        from jinja2 import TemplateError

        request_started = time.perf_counter()
        first_text_ms = None
        messages, tools = request.template_inputs()
        try:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                                     enable_thinking=False, **({'tools': tools} if tools else {}))
        except (ValueError, TemplateError) as exc:
            raise APIError(str(exc), param="messages") from exc
        # No limit (Ollama's default): until a stop or the context capacity.
        room = max(1, self.max_context - len(ids) + 1)
        limit = request.token_limit or room
        if getattr(request, 'within_context', False):
            limit = min(limit, room)
        if not ids or len(ids) + limit - 1 > self.max_context:
            raise APIError(f"Prompt ({len(ids)} tokens) plus output budget ({limit}) exceeds context capacity "
                           f"({self.max_context}); reduce messages or max_completion_tokens.",
                           code="context_length_exceeded")
        if on_start:
            on_start(len(ids))
        assets = getattr(self, 'assets', None)
        self.select_session(request, len(ids))
        try:
            started = time.perf_counter()
            stopped = False
            stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
            def publish(tokens):
                nonlocal first_text_ms, stopped
                if on_text or on_content or on_progress:
                    content, _ = self.visible_text(tokens, request)
                    text = streaming_text(content, request)
                    if text and first_text_ms is None:
                        first_text_ms = (time.perf_counter() - request_started) * 1000
                    if on_text:
                        on_text(text)
                    if on_content:
                        on_content(content)
                    if on_progress:
                        on_progress(content, min(len(tokens), limit))
                if stops:
                    raw = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    stopped = any(stop in raw for stop in stops)
            # A stop string ends decoding on every path, not only while streaming.
            options = (dict(on_tokens=publish, cancelled=lambda: stopped or bool(cancelled and cancelled()))
                       if on_text or on_content or on_progress or stops else {})
            if getattr(self.session, 'prefix_cache', None) is not None:
                stable = set()
                if messages:
                    # Keep earlier text blocks of the final message (agent
                    # project context), omitting its last block (the question).
                    # Exact token comparison handles template/BPE boundaries.
                    probes = []
                    for mode in ('message', True):
                        prefix_messages, _ = request.template_inputs(cache_prefix=mode)
                        if prefix_messages in probes:
                            continue
                        probes.append(prefix_messages)
                        try:
                            prefix = self.tokenizer.apply_chat_template(prefix_messages,
                                tokenize=True, return_dict=False, add_generation_prompt=False,
                                enable_thinking=False, **({'tools': tools} if tools else {}))
                        except (ValueError, TemplateError):
                            continue  # A cache probe must not reject a valid request.
                        count = 0
                        for a, b in zip(ids, prefix):
                            if a != b:
                                break
                            count += 1
                        if count:
                            stable.add(count)
                # Keep the system/tool checkpoint as a fallback if the project
                # text changes after a tool edits a file or git status changes.
                options['cache_prefix_tokens'] = sorted(stable) if len(stable) > 1 else next(iter(stable), 0)
            generation = self.session.generate(ids, limit, **options)
            tokens = generation.tokens
            self.last_metrics = dict(steps=getattr(generation, 'steps', 0),
                decode_gpu_ms=getattr(generation, 'decode_ms', 0.0), wall_ms=(time.perf_counter()-started)*1000,
                prefill_gpu_ms=getattr(generation, 'prefill_ms', 0.0),
                prefill_wall_ms=getattr(generation, 'prefill_wall_ms', 0.0),
                state_reset_ms=getattr(generation, 'state_reset_ms', 0.0),
                checkpoint_ms=getattr(generation, 'checkpoint_ms', 0.0),
                setup_ms=getattr(generation, 'setup_ms', 0.0), first_text_ms=first_text_ms,
                decode_wall_ms=getattr(generation, 'decode_wall_ms', 0.0),
                cached_prompt_tokens=getattr(generation, 'cached_prompt_tokens', 0),
                prefill_encode_ms=sum(t['encode_ms'] for t in getattr(generation, 'prefill_timings', [])),
                prefill_commit_ms=sum(t['commit_ms'] for t in getattr(generation, 'prefill_timings', [])),
                prefill_wait_ms=sum(t['wait_ms'] for t in getattr(generation, 'prefill_timings', [])),
                verify_tokens=assets.gamma+1 if assets and assets.gamma else 1)
            self.last_metrics['decode_step_ms'] = self.last_metrics['decode_gpu_ms'] / max(1, self.last_metrics['steps'])
            logging.getLogger(__name__).info('Generation: %s', self.last_metrics)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        content, finish = self.visible_text(tokens, request)
        if on_content:
            on_content(content)
        if on_progress:
            on_progress(content, len(tokens))
        return content, finish, len(ids), len(tokens)

    @property
    def loaded(self):
        return bool(getattr(self.session, 'engines', None))

    @property
    def resident_bytes(self):
        buffers = tuple((getattr(self.session, 'buffers', None) or {}).values())     # a generation may add buffers
        return sum({id(b): b.nbytes for b in buffers}.values()) if self.loaded else 0

    def unload(self):
        """Release the GPU allocations: weights, states and scratch. Compiled programs, pipelines and prefix
        checkpoints stay in host memory, so the next request maps the weights again without compiling."""
        for session in (self.session, *getattr(self, '_sessions', {}).values()):
            if session is not None:
                session.release_engines()
        gc.collect()
        logging.getLogger(__name__).info('Unloaded the model')

    def load(self):
        """Map the most recent session again, or the default one, by generating one token for a short prompt."""
        if self.loaded:
            return
        started = time.perf_counter()
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hello'}], tokenize=True,
                                                 return_dict=False, add_generation_prompt=True, enable_thinking=False)
        if self.session is None:                         # --no-warmup
            self.select_session(ChatRequest(model='', messages=[Message(role='user', content='Hello')]), len(ids))
        self.session.generate(ids, 1)
        logging.getLogger(__name__).info('Loaded the model in %.2f s', time.perf_counter() - started)

    def visible_text(self, tokens, request):
        eos = self.session.eos
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        finish = "stop" if end is not None else "length"
        visible = tokens[:end] if end is not None else tokens
        content = self.tokenizer.decode(visible, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        positions = [content.index(s) for s in stops if s in content]
        if positions:
            content, finish = content[:min(positions)], "stop"
        return content, finish


class KeepAlive:
    """Unload the model after an idle period; the next request, or a client's HEAD probe, loads it again."""

    PROBE_HOLD = 60.0          # seconds a probe-loaded model waits for its request, at least

    def __init__(self, backend, lock, seconds=math.inf):
        self.backend, self.lock, self.default = backend, lock, seconds
        self.deadline, self.recent, self.busy, self.epoch = None, seconds, False, 0
        self.condition = threading.Condition()
        if hasattr(backend, 'unload'):
            threading.Thread(target=self._expire, name='keep-alive', daemon=True).start()
        self.touch()

    def touch(self, seconds=None):
        """Restart the idle period: the finished request's keep_alive, else the server's."""
        seconds = self.default if seconds is None else seconds
        with self.condition:
            self.recent, self.epoch = seconds, self.epoch + 1
            self.deadline = None if math.isinf(seconds) else time.monotonic() + seconds
            self.condition.notify()

    def expires_at(self):
        deadline = self.deadline
        return None if deadline is None else ollama_time(time.time() + deadline - time.monotonic())

    def run(self, action, *, wait=True):
        """Load or unload between generations; requests wait for it instead of reporting a busy model."""
        if not self.lock.acquire(blocking=wait):
            return
        self.busy = True
        try:
            action()
        finally:
            self.busy = False
            self.lock.release()

    def preload(self):
        # Queue behind whatever owns the model (a request, a load or an unload in progress); residency is
        # decided under the model lock.
        epoch = self.epoch
        threading.Thread(target=self._background, args=(lambda: self._probe_load(epoch),), kwargs={'wait': True},
                         daemon=True).start()

    def unload(self):
        """Run while owning the model: unload it. A probe accepted before stands down; one accepted while the
        unload runs loads the model again after it."""
        with self.condition:
            self.epoch, self.deadline, self.recent = self.epoch + 1, None, 0
        self.backend.unload()

    def _probe_load(self, epoch):
        # A request that ran after the probe has loaded the model and set its own period. A resident model keeps
        # a running period; one whose period already ran out has an expiry waiting for the model, which the
        # restart retires. Either way, a probe that no request follows still ends in an unload.
        if self.epoch != epoch:
            return
        if not getattr(self.backend, 'loaded', True):
            self.load(max(self.recent, self.PROBE_HOLD))
        elif self.deadline is None and not math.isinf(self.recent):
            self.touch(max(self.recent, self.PROBE_HOLD))

    def load(self, seconds=None):
        """Run while owning the model: load it and restart the period, so an expiry waiting for the model sees a
        new epoch and a request that runs next sets the period after it."""
        try:
            self.backend.load()
        finally:
            self.touch(seconds)

    def _background(self, action, wait=False):
        try:
            self.run(action, wait=wait)
        except Exception:
            logging.getLogger(__name__).exception('Background model load or unload failed')

    def _expire(self):
        while True:
            with self.condition:
                while self.deadline is None or self.deadline > time.monotonic():
                    self.condition.wait(None if self.deadline is None else self.deadline - time.monotonic())
                self.deadline, epoch = None, self.epoch
            # Wait out a request in progress: it restarts the period before releasing the model, and a period
            # that ends at once (keep_alive 0) has no later deadline to retry from.
            if self.backend.loaded:
                self._background(lambda: self._unload(epoch), wait=True)

    def _unload(self, epoch):
        # The period may have restarted between the deadline and this thread taking the model lock.
        if self.epoch == epoch:
            self.backend.unload()


def create_app(backend, model_name, api_key=None, *, keep_alive=math.inf):
    app = FastAPI(title="lithos-metal", version=__version__)
    lock = threading.Lock()
    keeper = KeepAlive(backend, lock, keep_alive)
    created = int(time.time())

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        kind = {401: "authentication_error", 429: "rate_limit_error", 500: "server_error"}.get(
            exc.status, "invalid_request_error")
        if request.url.path.startswith('/v1/messages'):
            return JSONResponse(status_code=exc.status, content={'type': 'error', 'error': {'type': kind, 'message': exc.message}})
        if request.url.path.startswith('/api/'):
            return JSONResponse(status_code=exc.status, content={'error': exc.message})
        return JSONResponse(status_code=exc.status, content={"error": {
            "message": exc.message, "type": kind, "param": exc.param, "code": exc.code}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        param = ".".join(str(p) for p in error["loc"] if p != "body")
        return await api_error(request, APIError(f"{param}: {error['msg']}", param=param))

    async def authorize(request: Request):
        bearer = request.headers.get('authorization', '').removeprefix('Bearer ')
        key = request.headers.get('x-api-key', '') if request.url.path.startswith('/v1/messages') else ''
        if api_key and not any(secrets.compare_digest(value.encode(), api_key.encode()) for value in (bearer, key)):
            raise APIError("Invalid API key", 401, "invalid_api_key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": created,
            "owned_by": "lithos-metal", "context_window": getattr(backend, 'max_context', 32768)}]}

    def validate_model(request):
        if request.model != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")

    def execute(request, wire, **options):
        content, finish, prompt_tokens, completion_tokens = backend.complete(request, **options)
        wire.metrics = dict(getattr(backend, 'last_metrics', {}))
        message, finish = parse_completion(content, request, finish)
        return wire.body(message, finish, prompt_tokens, completion_tokens)

    def acquire(wait):
        # Ollama clients queue requests; the other adapters report a busy model, except during a load or unload.
        return lock.acquire(blocking=False) or ((wait or keeper.busy) and lock.acquire(timeout=-1 if wait else 120))

    def release(keep_alive):
        keeper.touch(keep_alive)                         # while the request still owns the model
        lock.release()

    def dispatch(request, protocol, custom=(), *, keep_alive=None, wait=False):
        validate_model(request)
        arrived = time.perf_counter()                   # Ollama's total_duration includes the queue
        if not acquire(wait):
            raise APIError("The model is busy; retry after the current request finishes", 429, "model_busy")
        wire = WireResponse(protocol, model_name, custom, request.stream_usage if request.stream else None)
        wire.started = arrived
        if request.stream:
            return stream(request, wire, keep_alive)
        try:
            body = execute(request, wire)
            metrics = dict(getattr(backend, 'last_metrics', {}))
        except APIError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation failed")
            raise APIError("Generation failed; see server logs", 500, "generation_failed") from exc
        finally:
            release(keep_alive)
        headers = {f'X-{brand}-{name}': str(metrics[key]) for brand in ('Lithos-Metal', 'LMK', 'Monolith') for name, key in (
            ('Decode-Steps', 'steps'), ('Decode-GPU-Ms', 'decode_gpu_ms'),
            ('Decode-Step-Ms', 'decode_step_ms'), ('Verify-Tokens', 'verify_tokens')) if key in metrics}
        return JSONResponse(headers=headers, content=body)

    def stream(request, wire, keep_alive=None):
        events = queue.Queue()
        cancelled = threading.Event()
        def worker():
            try:
                options = dict(on_text=lambda text: events.put(('text', text)),
                               on_start=lambda count: events.put(('start', count)),
                               cancelled=cancelled.is_set) if isinstance(backend, Backend) else {}
                if options and wire.protocol == 'chat':
                    options.pop('on_text')
                    if wire.usage == 'continuous':
                        # One queue item: the count must arrive with the content it produced.
                        options['on_progress'] = lambda content, count: events.put(('progress', (content, count)))
                    else:
                        options['on_content'] = lambda content: events.put(('content', content))
                result = execute(request, wire, **options)
                if not options:
                    usage = result.get('usage') or {'input_tokens': result.get('prompt_eval_count', 0)}
                    events.put(('start', usage.get('input_tokens', usage.get('prompt_tokens', 0))))
                events.put(('result', result))
            except APIError as exc:
                events.put(('error', (exc.message, exc.code)))
            except Exception:
                logging.getLogger(__name__).exception('Streaming generation failed')
                events.put(('error', ('Generation failed; see server logs', 'generation_failed')))
            finally:
                release(keep_alive)
        # A worker owns the lock from here, even if the response is never consumed.
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        async def generate():
            try:
                last_ping = time.monotonic()
                while True:
                    if events.empty():
                        if not thread.is_alive() and events.empty():
                            yield wire.error('Generation worker terminated', 'generation_failed')
                            break
                        await asyncio.sleep(.01)
                        if time.monotonic() - last_ping > 5 and wire.protocol != 'ollama':
                            yield ': keep-alive\n\n'
                            last_ping = time.monotonic()
                        continue
                    kind, value = events.get_nowait()
                    if kind == 'start':
                        wire.input_tokens = value
                        for event in wire.start():
                            yield event
                    elif kind in ('text', 'content', 'progress'):
                        if kind == 'progress':
                            value, wire.output_tokens = value
                        text = streaming_text(value, request) if kind != 'text' else value
                        if not text.startswith(wire.text):
                            raise RuntimeError('Decoded text changed after streaming')
                        for event in wire.delta(text[len(wire.text):]):
                            yield event
                        if kind != 'text':
                            for snapshot in tool_prefixes(value, request):
                                for event in wire.tool_delta(*snapshot):
                                    yield event
                        for event in wire.progress():
                            yield event
                    elif kind == 'error':
                        yield wire.error(*value)
                        break
                    else:
                        if wire.protocol == 'chat':
                            text = value['choices'][0]['message'].get('content') or ''
                            wire.output_tokens = value['usage']['completion_tokens']
                        elif wire.protocol == 'ollama':
                            text = value['message']['content']
                        elif wire.protocol == 'messages':
                            text = ''.join(b['text'] for b in value['content'] if b['type'] == 'text')
                        else:
                            text = ''.join(p['text'] for b in value['output'] if b['type'] == 'message' for p in b['content'])
                        for event in wire.delta(text[len(wire.text):]):
                            yield event
                        for event in wire.finish(value, wire.usage is not None):
                            yield event
                        break
            except APIError as exc:
                yield wire.error(exc.message, exc.code)
            finally:
                cancelled.set()
        return StreamingResponse(generate(), media_type='application/x-ndjson' if wire.protocol == 'ollama' else 'text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat(request: ChatRequest):
        return dispatch(request, 'chat')

    def convert(body, converter):
        try:
            return converter(body)
        except (ValidationError, KeyError, TypeError, AttributeError) as exc:
            raise APIError(f'Invalid request: {exc}') from exc

    @app.post('/v1/messages', dependencies=[Depends(authorize)])
    def messages(body: dict):
        return dispatch(convert(body, anthropic_request), 'messages')

    @app.post('/v1/messages/count_tokens', dependencies=[Depends(authorize)])
    def count_tokens(body: dict):
        request = convert({**body, 'max_tokens': 1}, anthropic_request)
        validate_model(request)
        messages, tools = request.template_inputs()
        ids = backend.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False, **({'tools': tools} if tools else {}))
        return {'input_tokens': len(ids)}

    @app.post('/v1/responses', dependencies=[Depends(authorize)])
    def responses(body: dict):
        request, custom = convert(body, responses_request)
        return dispatch(request, 'responses', custom)

    async def json_body(request: Request):
        # Ollama reads any request body as JSON; its documented curl calls send no Content-Type.
        try:
            body = json.loads(await request.body() or b'{}')
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise APIError(f'Invalid JSON body: {exc}') from exc
        if not isinstance(body, dict):
            raise APIError('The request body must be a JSON object')
        return body

    def ollama_entry(**fields):
        return {'name': model_name, 'model': model_name, 'modified_at': ollama_time(created), 'size': 0,
                'digest': '', 'details': {}, **fields}

    @app.head('/api/chat', dependencies=[Depends(authorize)])
    def ollama_probe():
        # Clients probe before a request: an unloaded model starts loading now.
        keeper.preload()
        return Response()

    @app.post('/api/chat', dependencies=[Depends(authorize)])
    def ollama_chat(body: dict = Depends(json_body)):
        keep_alive = None if body.get('keep_alive') in (None, '') else keep_alive_seconds(body['keep_alive'])
        if not isinstance(body.get('messages', []), (list, type(None))):
            raise APIError('messages must be an array', param='messages')
        if body.get('messages'):
            return dispatch(convert(body, lambda body: ollama_request(body, model_name)), 'ollama',
                            keep_alive=keep_alive, wait=True)
        # No messages: load the model, or unload it with keep_alive 0.
        if ollama_model(str(body.get('model', '')), model_name) != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")
        unload = keep_alive == 0
        if hasattr(backend, 'unload'):
            try:
                keeper.run(keeper.unload if unload else lambda: keeper.load(keep_alive))
            except Exception as exc:
                logging.getLogger(__name__).exception('Model load failed')
                raise APIError('Model load failed; see server logs', 500, 'load_failed') from exc
        elif not unload:
            keeper.touch(keep_alive)
        return {'model': model_name, 'created_at': ollama_time(), 'message': {'role': 'assistant', 'content': ''},
                'done_reason': 'unload' if unload else 'load', 'done': True}

    @app.get('/api/tags', dependencies=[Depends(authorize)])
    def ollama_tags():
        return {'models': [ollama_entry()]}

    @app.get('/api/ps', dependencies=[Depends(authorize)])
    def ollama_ps():
        if not getattr(backend, 'loaded', True):
            return {'models': []}
        return {'models': [ollama_entry(size_vram=getattr(backend, 'resident_bytes', 0), expires_at=keeper.expires_at())]}

    return app


def _prefill_chunk(value):
    """``N``, ``N-exact`` or ``auto-exact`` -> (rows, or None for the chip's measured size; exact)."""
    rows, exact = (value[:-len('-exact')], True) if value.endswith('-exact') else (value, False)
    rows = None if (rows, exact) == ('auto', True) else int(rows)
    if rows is not None and rows < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return rows, exact


def _keep_alive(value):
    try:
        return keep_alive_seconds(value)
    except APIError as exc:
        raise argparse.ArgumentTypeError(exc.message) from None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="lithos-metal serve", description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local checkpoint path")
    draft = parser.add_mutually_exclusive_group()
    draft.add_argument("--draft", "--drafter", dest="draft", help="Override the automatically selected DSpark head (Hub ID or local path)")
    draft.add_argument("--no-draft", action='store_true', help="Disable automatic DSpark speculative decoding")
    parser.add_argument("--draft-kind", choices=['dspark'], default='dspark')
    parser.add_argument("--draft-block-size", type=int, help="Draft proposals per round (default: up to seven, plus one target anchor)")
    parser.add_argument("--draft-sampling", choices=['argmax', 'sample'], default='argmax',
                        help="Drafts at temperature > 0: argmax = the drafter's argmax, kept while the target samples it; "
                             "sample = drawn from the drafter's distribution, accepted with min(1, p/q). Both preserve the target's distribution")
    parser.add_argument("--pack", help="Local pack-cache directory (default: $XDG_CACHE_HOME/lithos-metal/packs; reuses legacy cache); existing packs also accepted")
    parser.add_argument("--draft-pack", help="Optional separate draft cache or existing draft pack")
    parser.add_argument("--draft-quantization", choices=['auto', 'none', 'nvfp4'], default='auto',
                        help="auto uses validated chip-specific NVFP4 draft recipes when available; otherwise source precision")
    parser.add_argument("--revision", help="Target Hugging Face revision")
    parser.add_argument("--draft-revision", help="Draft Hugging Face revision")
    parser.add_argument("--download-dir", help="Hugging Face download cache directory")
    parser.add_argument("--local-files-only", action='store_true', help="Resolve Hub IDs from the local Hub cache only")
    parser.add_argument("--kernel-config", help="Optional explicit target/draft recipe JSON; defaults to matching chip recipes")
    parser.add_argument("--kernel-config-key", help="Pin a context key in the selected recipe map")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--prefill-chunk-size", type=_prefill_chunk, default='auto-exact', metavar='N|N-exact|auto-exact',
                        help="Prompt tokens per prefill pass. N-exact reproduces the results of 128-token passes bit for bit; "
                             "auto-exact (default) uses the chip's measured size for the model that way, else 128")
    parser.add_argument("--no-warmup", action='store_true', help="Skip startup compilation/warmup; the first request pays this cost")
    parser.add_argument("--keep-alive", type=_keep_alive, default=math.inf, metavar='DURATION',
                        help="Unload the model after this long without requests (5m, 1h30m, or seconds; 0 = after every "
                             "request; default: never). The next request loads it again; Ollama requests can set their own keep_alive")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from .models.catalog import default_draft
    if not args.draft and not args.no_draft:
        args.draft = default_draft(args.model)
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.draft_block_size is not None and args.draft_block_size < 1:
        parser.error('--draft-block-size must be positive')
    if not args.draft and (args.draft_pack or args.draft_revision or args.draft_block_size is not None or args.kernel_config or args.kernel_config_key
                          or args.draft_quantization != 'auto' or args.draft_sampling != 'argmax'):
        parser.error('draft options require --draft or a target with an automatic DSpark head')
    return args


def main(argv=None):
    import uvicorn
    from .serving.setup import prepare

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    assets = prepare(args)
    api_key = os.environ.get("LITHOS_METAL_API_KEY") or os.environ.get("LMK_API_KEY") or os.environ.get("MONOLITH_API_KEY")
    rows, exact = args.prefill_chunk_size
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), args.max_context, rows or assets.prefill_chunk_size or 128,
                      assets=assets, prefill_exact=exact)
    model_name = args.served_model_name or (assets.model_dir.name if Path(args.model).expanduser().exists() else args.model)
    if not args.no_warmup:
        backend.warmup(model_name)
    app = create_app(backend, model_name, api_key, keep_alive=args.keep_alive)
    logging.getLogger(__name__).info('lithos-metal ready: http://%s:%s — model=%s; DSpark=%s; verification rows=%s',
                                   args.host, args.port, model_name, args.draft or 'disabled', assets.gamma + 1)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
