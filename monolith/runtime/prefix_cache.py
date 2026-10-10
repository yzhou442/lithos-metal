"""Bounded, exact-token checkpoints for attention and recurrent model state."""
from dataclasses import dataclass, field
import math
import time


@dataclass
class Prefix:
    tokens: tuple
    state: bytes
    buffers: dict
    hits: int = field(default=0, compare=False)            # restores served from this copy
    on_disk: bool = field(default=False, compare=False)    # a PrefixStore holds the same bytes


class PrefixCache:
    """Host copies of the newest checkpoints, optionally over a ``store`` (PrefixStore) on local disk.

    Without a store the cache is process memory only. With one, a checkpoint that was restored at least once is
    written to the store when the host tier evicts it, ``flush()`` writes every host copy (before an idle release
    drops them, and at exit when the store persists), and a longer prefix found only on disk is read, verified and
    promoted to the host tier. Both tiers hold the same bytes, so a restore from either sets the same state."""

    def __init__(self, entries, max_bytes=4 * 1024**3, min_tokens=0, store=None):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError('prefix cache byte budget must be a nonnegative integer')
        self.entries = {entry.name: entry for entry in entries}
        self.max_bytes = max_bytes
        self.min_tokens = min_tokens
        self.items = []
        self.store = store
        self.state_size = self.saved_names = None     # StepState bytes and saved buffer names, from the first save
        self.last = {}                                 # the latest match: its tier and the time reading it took

    def match(self, tokens):
        # Leave at least one input token for the final prefill/sampling pass.
        matches = [item for item in self.items if len(item.tokens) < len(tokens)
                   and item.tokens == tuple(tokens[:len(item.tokens)])]
        item = max(matches, key=lambda item: len(item.tokens), default=None)
        self.last = dict(source='memory' if item is not None else None, load_ms=0.0)
        if self.store is not None:
            record = self.store.match(tokens)
            if record is not None and len(record.tokens) > (len(item.tokens) if item is not None else 0):
                started = time.perf_counter()
                loaded = self.store.load(record, self._check)
                if loaded is not None:
                    item = Prefix(*loaded, on_disk=True)
                    self._insert(item)
                    self.last = dict(source='disk', load_ms=(time.perf_counter() - started) * 1000)
        if item is not None:
            item.hits += 1
        return item

    def save(self, tokens, engine):
        tokens = tuple(tokens)
        if not tokens or len(tokens) < self.min_tokens or any(item.tokens == tokens for item in self.items):
            return
        sizes = {}
        for name, spec in engine.program.buffers.items():
            entry = self.entries.get(name)
            # Worker/barrier counters belong to the compiled graph, not the
            # sequence. Only checkpoint model and drafter state.
            if spec.role != 'state' or entry is None:
                continue
            size = spec.nbytes
            if self._position_major(entry):
                size = len(tokens) * math.prod(entry.shape[1:]) * entry.dtype.itemsize
            sizes[name] = size
        if sum(sizes.values()) > self.max_bytes:
            return
        self.state_size, self.saved_names = engine.program.layout.size, frozenset(sizes)
        # Evict before copying so peak memory stays bounded, including the two
        # recurrent slots whose parity is carried by the saved StepState.
        self._evict(tokens, sum(sizes.values()))
        self.items.append(Prefix(tokens,
            engine.buffers[engine.program.step_state].read(0, engine.program.layout.size),
            {name: engine.buffers[name].read(0, size) for name, size in sizes.items()}))

    def flush(self, timeout=None):
        """Queue every host copy the store does not hold yet and wait for the writes."""
        if self.store is None:
            return True
        for item in self.items:
            self._spill(item)
        return self.store.flush(timeout)

    def clear(self):
        """Drop every snapshot (their host copies of the attention and recurrent state)."""
        self.items.clear()

    def restore(self, item, engine):
        # Check first: a checkpoint that does not fit is never partly written.
        if item.on_disk and frozenset(item.buffers) != frozenset(
                n for n, spec in engine.program.buffers.items() if spec.role == 'state' and n in self.entries):
            raise ValueError("a stored prefix checkpoint does not hold this program's state buffers")
        for name, data in item.buffers.items():
            if len(data) > engine.program.buffers[name].nbytes:
                raise ValueError(f'prefix checkpoint buffer {name!r} exceeds its allocation')
        if len(item.state) != engine.program.layout.size:
            raise ValueError('prefix checkpoint StepState does not match the program layout')
        for name, data in item.buffers.items():
            engine.buffers[name].write(data, 0)
        engine.buffers[engine.program.step_state].write(item.state, 0)

    @staticmethod
    def _position_major(entry):
        return entry.checkpoints == 1 and entry.name.endswith(('k_cache', 'v_cache', 'k_ctx', 'v_ctx'))

    def _evict(self, tokens, incoming):
        while self.items and (len(self.items) >= 2 or
                sum(len(v) for item in self.items for v in item.buffers.values()) + incoming > self.max_bytes):
            # Retain the shared system/tool prefix while refreshing the latest
            # conversation checkpoint; replace unrelated prefixes normally.
            index = -1 if len(self.items) > 1 and tokens[:len(self.items[0].tokens)] == self.items[0].tokens else 0
            evicted = self.items.pop(index)
            if evicted.hits:                          # reused once: worth a disk copy
                self._spill(evicted)

    def _insert(self, item):
        nbytes = sum(len(v) for v in item.buffers.values())
        if nbytes <= self.max_bytes:
            self._evict(item.tokens, nbytes)
            self.items.append(item)

    def _spill(self, item):
        if self.store is not None and not item.on_disk:
            self.store.submit(item.tokens, item.state, item.buffers, on_written=lambda: setattr(item, 'on_disk', True))

    def _check(self, sizes, n_tokens, state_size):
        """A stored checkpoint must hold exactly the buffers this cache saves, at the sizes it saves them."""
        from .prefix_store import CorruptEntry
        if self.saved_names is not None and frozenset(sizes) != self.saved_names:
            raise CorruptEntry('different state buffers')
        if self.state_size is not None and state_size != self.state_size:
            raise CorruptEntry('different StepState size')
        for name, size in sizes.items():
            entry = self.entries.get(name)
            if entry is None:
                raise CorruptEntry(f'unknown state buffer {name}')
            if self._position_major(entry):
                want = n_tokens * math.prod(entry.shape[1:]) * entry.dtype.itemsize
            else:
                want = max(1, entry.checkpoints) * math.prod(entry.shape) * entry.dtype.itemsize
            if size != want:
                raise CorruptEntry(f'{name} holds {size} bytes, not {want}')
