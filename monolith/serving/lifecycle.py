"""Release a served model's GPU allocations after an idle period and load them again for the next request."""
from __future__ import annotations

import contextlib
import enum
import logging
import threading
import time

LOG = logging.getLogger(__name__)


class ModelState(str, enum.Enum):
    UNLOADED = 'unloaded'
    LOADING = 'loading'
    READY = 'ready'
    UNLOADING = 'unloading'
    FAILED = 'failed'


class ModelLoadError(RuntimeError):
    """Loading the model failed. The model stays unloaded and a later request tries again."""


class IdleRelease:
    """The model's lifecycle: ``unloaded -> loading -> ready -> unloading -> unloaded``, ``loading -> failed``.

    Loads and unloads run while the caller holds ``gpu_lock``, the lock every GPU submission holds, so a load is
    single-flight and an unload never overlaps a generation or the command buffers it waits for. A request counts as
    pending from its arrival, before it waits for that lock: the idle period starts when the last request finishes
    with none pending, and a request that arrives during an unload waits for it and then loads the model again.

    ``seconds`` is the idle period on the monotonic clock; ``unload`` is called with the lock held.
    """

    def __init__(self, gpu_lock, unload, seconds, *, state=ModelState.READY, clock=time.monotonic):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0:
            raise ValueError('the idle period must be a positive number of seconds')
        self.gpu_lock, self._unload, self.seconds, self.clock = gpu_lock, unload, float(seconds), clock
        self.state = ModelState(state)
        self.error = None                     # the last failed load's exception
        self.attempts = 0                     # loads finished, failed or not; a request remembers the count it arrived at
        self.pending = 0                      # requests between arrival and completion
        self.loads = self.unloads = 0
        self.last_active = clock()
        self._condition = threading.Condition()
        self._stopped = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._watch, name='idle-release', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the watcher and wait for it (it finishes an unload in progress first); it unloads nothing after."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        thread = getattr(self, '_thread', None)
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    @contextlib.contextmanager
    def request(self):
        """Count a request from its arrival (before it waits for the GPU lock) to its completion, failure or
        cancellation. Yields the count of finished loads it arrived at, for ``ensure_loaded``."""
        with self._condition:
            self.pending += 1
            arrived = self.attempts
        try:
            yield arrived
        finally:
            with self._condition:
                self.pending -= 1
                self.last_active = self.clock()
                self._condition.notify_all()

    def ensure_loaded(self, arrived, load):
        """With the GPU lock held: run ``load`` unless the model is ready. Requests that waited for a load share its
        result: a load that finished failing after a request arrived (it ran or was running meanwhile) fails that
        request without another attempt; a failure from before its arrival is retried."""
        if self.state is ModelState.READY:
            return
        if self.state is ModelState.FAILED and self.attempts != arrived:
            raise ModelLoadError(f'Loading the model failed: {self.error}') from self.error
        self._set(ModelState.LOADING)
        started = self.clock()
        try:
            load()
        except Exception as exc:
            self.error = exc
            LOG.exception('Loading the model failed')
            try:
                self._unload()                # drop whatever the failed load allocated
            except Exception:
                LOG.exception('Releasing a partial model load failed')
            with self._condition:
                self.attempts += 1
            self._set(ModelState.FAILED)
            raise ModelLoadError(f'Loading the model failed: {exc}') from exc
        self.error = None
        with self._condition:
            self.attempts += 1
            self.loads += 1
        self._set(ModelState.READY)
        LOG.info('Loaded the model in %.2f s', self.clock() - started)

    def release_if_idle(self):
        """Unload the model if it is ready and no request arrived or finished within the idle period. Waits for the
        GPU lock, then checks again: a request that arrived in the meantime keeps the model."""
        with self.gpu_lock:
            with self._condition:
                if (self.state is not ModelState.READY or self.pending
                        or self.clock() - self.last_active < self.seconds):
                    return False
                self.state = ModelState.UNLOADING
            LOG.info('Model state: ready -> unloading after %.0f s idle', self.clock() - self.last_active)
            started = self.clock()
            try:
                self._unload()
            except Exception:
                # Whatever remains allocated is mapped again or reused by the next load.
                LOG.exception('Releasing the idle model failed')
            self.unloads += 1
            self._set(ModelState.UNLOADED)
            LOG.info('Released the idle model in %.2f s', self.clock() - started)
        return True

    def _set(self, state):
        with self._condition:
            previous, self.state = self.state, state
            self._condition.notify_all()
        if previous is not state:
            LOG.info('Model state: %s -> %s', previous.value, state.value)

    def _watch(self):
        while True:
            with self._condition:
                while not self._stopped:
                    if self.state is ModelState.READY and not self.pending:
                        remaining = self.last_active + self.seconds - self.clock()
                        if remaining <= 0:
                            break
                        self._condition.wait(min(remaining, threading.TIMEOUT_MAX))   # waits past it raise
                    else:
                        self._condition.wait()
                if self._stopped:
                    return
            try:
                self.release_if_idle()
            except Exception:
                LOG.exception('Idle release failed')
                time.sleep(1.0)
