"""Exact-token prefix checkpoints on local disk, below PrefixCache's host copies.

An entry is one file: ``MAGIC``, the header length, a JSON header and the blobs it describes (the prompt's token
IDs as little-endian int32, the StepState and each state buffer), each at a 4 KiB-aligned offset with its SHA-256.
Files are written under ``tmp/`` and renamed into place, so a partial write is never an entry. The header names the
identity the bytes depend on (format, code, device, weights, layout; ``Session.prefix_identity``); an entry with
another identity is never matched, and one that fails any check is deleted and treated as a miss.

Entries live in ``entries/`` and outlive the process: a later server with the same identity reuses them. One server
uses a directory at a time: an open store holds an exclusive lock on ``.owner``, and opening another store on the
same root raises ``StoreBusy`` (that server keeps its checkpoints in memory). All entries share one byte quota with
least-recently-used eviction; an entry being read or written is never evicted. Partial writes a stopped server left
are removed when the next one opens the directory.
"""
from __future__ import annotations

import concurrent.futures
import errno
import fcntl
import functools
import hashlib
import json
import logging
import os
import queue
import stat
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

FORMAT = 1
MAGIC = b'LMPREFX1'
SUFFIX = '.lpc'
ALIGN = 4096
MAX_HEADER = 1 << 20


@functools.lru_cache(maxsize=None)
def code_digest():
    """SHA-256 over the engine's Python package and kernel sources: the code a snapshot's bytes came from."""
    from ..resources import kernel_root
    package = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for root in (package, kernel_root()):
        for path in sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                           and p.suffix not in ('.pyc', '.pyo')):
            digest.update(str(path.relative_to(root)).encode() + b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass
class Record:
    path: Path
    identity: str
    tokens: tuple
    nbytes: int


class CorruptEntry(ValueError):
    pass


class StoreBusy(OSError):
    """Another server holds the directory."""


def _pad(n):
    return -n % ALIGN


def _private_dir(path):
    """Create ``path`` (0700) or accept an existing directory of this user that others cannot write."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise PermissionError(f'{path} must be a directory owned by this user and not writable by others')


class PrefixStore:
    """``identity``: the hex digest snapshots must match; ``max_bytes``: the quota of every entry under ``root``."""

    def __init__(self, root, max_bytes, identity, *, workers=8, pending=2):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError('the prefix store quota must be a positive number of bytes')
        self.root = Path(root).expanduser().absolute()
        self.max_bytes, self.identity = max_bytes, identity
        self.workers = workers
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        _private_dir(self.root)
        self._owner = os.open(self.root / '.owner', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._owner)
            raise StoreBusy(errno.EBUSY, f'another server uses the prefix cache at {self.root}') from None
        self.tmp = self.root / 'tmp'
        _private_dir(self.tmp)
        self.dir = self.root / 'entries'
        _private_dir(self.dir)
        self._mutex = threading.Lock()
        self._pinned = set()          # paths being read or written
        self._index = {}              # path -> Record
        self._counter = 0
        self._closed = False          # closing: no new writes, reads or deletions (the next owner may hold the files)
        self.stats = dict(hits=0, misses=0, writes=0, write_bytes=0, write_ms=0.0, read_ms=0.0,
                          evictions=0, errors=0, dropped=0)
        for path in self.tmp.glob('*.partial'):                # no writer holds them: this store owns the root
            path.unlink(missing_ok=True)
        self._scan()
        self._queue = queue.Queue(maxsize=pending)
        self._writer = threading.Thread(target=self._drain, name='prefix-store', daemon=True)
        self._writer.start()
        LOG.info('Prefix store at %s (quota %.1f GB, %d entries)', self.dir, max_bytes / 1e9, len(self._index))

    # -- lookup ---------------------------------------------------------------------------------------------------

    def match(self, tokens):
        """The longest entry of this identity whose tokens are a strict prefix of ``tokens``."""
        tokens = tuple(tokens)
        with self._mutex:
            if self._closed:
                return None
            records = [r for r in self._index.values() if r.identity == self.identity
                       and len(r.tokens) < len(tokens) and r.tokens == tokens[:len(r.tokens)]]
        return max(records, key=lambda r: len(r.tokens), default=None)

    def contains(self, tokens):
        tokens = tuple(tokens)
        with self._mutex:
            return self._path(tokens) in self._index

    def _path(self, tokens):
        digest = hashlib.sha256(struct.pack(f'<{len(tokens)}i', *tokens)).hexdigest()
        return self.dir / f'{self.identity[:16]}-{digest[:40]}{SUFFIX}'

    def remove(self, tokens):
        """Delete this identity's entry for exactly ``tokens``, if there is one."""
        self._discard(self._path(tuple(tokens)))

    def load(self, record, expected):
        """Read and verify ``record``: ``(tokens, state, {name: bytes})``, or None (the entry is dropped).
        ``expected(names_sizes, n_tokens, state_size)`` raises CorruptEntry for buffers this program cannot take."""
        started = time.perf_counter()
        with self._mutex:
            if self._closed:
                return None
            self._pinned.add(record.path)
        try:
            fd = os.open(record.path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                header, size = self._header(fd)
                if header['identity'] != self.identity:
                    raise CorruptEntry('identity changed')
                tokens = self._tokens(fd, header)
                if tokens != record.tokens:
                    raise CorruptEntry('tokens changed')
                blobs = [header['state']] + header['buffers']
                expected({b['name']: b['size'] for b in header['buffers']}, len(tokens), header['state']['size'])
                with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
                    data = list(pool.map(lambda blob: self._blob(fd, blob), blobs))
            finally:
                os.close(fd)
            try:
                os.utime(record.path)                         # least recently used: by modification time
            except OSError:
                pass
            self.stats['hits'] += 1
            self.stats['read_ms'] += (time.perf_counter() - started) * 1000
            return tokens, data[0], {b['name']: d for b, d in zip(header['buffers'], data[1:])}
        except (CorruptEntry, OSError, ValueError, KeyError, TypeError) as exc:
            self.stats['errors'] += 1
            LOG.warning('Prefix store: dropping %s (%s); prefilling instead', record.path.name, exc)
            self._discard(record.path)
            return None
        finally:
            with self._mutex:
                self._pinned.discard(record.path)

    # -- writes ---------------------------------------------------------------------------------------------------

    def submit(self, tokens, state, buffers, on_written=None, timeout=0):
        """Queue a snapshot for the writer thread; the bytes objects are shared, not copied. A queue still full after
        ``timeout`` seconds (None: as long as it takes) drops it and returns False."""
        tokens = tuple(tokens)
        if self._closed:
            self.stats['dropped'] += 1
            return False
        if self.contains(tokens):
            return True
        try:
            self._queue.put((tokens, state, buffers, on_written), timeout != 0, timeout)
            return True
        except queue.Full:
            self.stats['dropped'] += 1
            return False

    def flush(self, timeout=None):
        """Wait until every queued snapshot is written or abandoned."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def close(self):
        """Refuse new writes, finish the accepted ones and give the directory up to the next server. A closed store
        reads, writes and deletes nothing: the next owner may hold the files."""
        with self._mutex:
            self._closed = True
        self.flush()
        if self._owner is not None:
            os.close(self._owner)
            self._owner = None

    def write(self, tokens, state, buffers):
        """Write one snapshot now (the writer thread's work). Returns the entry's path, or None on any failure."""
        tokens = tuple(tokens)
        token_bytes = struct.pack(f'<{len(tokens)}i', *tokens)
        named = [('tokens', token_bytes), ('state', state)] + list(buffers.items())
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            digests = list(pool.map(lambda item: hashlib.sha256(item[1]).hexdigest(), named))
        blobs, offset = [], 0
        for (name, data), digest in zip(named, digests):
            blobs.append(dict(name=name, offset=offset, size=len(data), sha256=digest))
            offset += len(data) + _pad(len(data))
        base = ALIGN
        while True:                    # the blobs start at the first aligned offset after the header
            placed = [dict(blob, offset=blob['offset'] + base) for blob in blobs]
            head = json.dumps(dict(format=FORMAT, identity=self.identity, created=time.time(),
                                   tokens=dict(placed[0], count=len(tokens)), state=placed[1], buffers=placed[2:]),
                              sort_keys=True).encode()
            prologue = MAGIC + struct.pack('<Q', len(head)) + head
            if len(prologue) <= base:
                break
            base = len(prologue) + _pad(len(prologue))
        total = placed[-1]['offset'] + placed[-1]['size']
        final = self._path(tokens)
        name = final.name
        with self._mutex:
            self._counter += 1
            partial = self.tmp / f'{self._counter}.partial'
            self._pinned.add(final)
        started = time.perf_counter()
        fd = None
        try:
            if total > self.max_bytes or not self._make_room(total, final):
                LOG.warning('Prefix store: a %.2f GB snapshot does not fit the %.1f GB quota; not saved',
                            total / 1e9, self.max_bytes / 1e9)
                return None
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.ftruncate(fd, total)
            self._write_all(fd, prologue, 0)
            for (_, data), blob in zip(named, placed):
                self._write_all(fd, data, blob['offset'])
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(partial, final)
            dir_fd = os.open(self.dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            self.stats['errors'] += 1
            LOG.warning('Prefix store: writing %s failed (%s); continuing without it',
                        name, errno.errorcode.get(exc.errno, exc))
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(partial)
            except OSError:
                pass
            return None
        finally:
            with self._mutex:
                self._pinned.discard(final)
        with self._mutex:
            self._index[final] = Record(final, self.identity, tokens, total)
        self.stats['writes'] += 1
        self.stats['write_bytes'] += total
        self.stats['write_ms'] += (time.perf_counter() - started) * 1000
        return final

    @staticmethod
    def _write_all(fd, data, offset):
        view = memoryview(data)
        while view:
            n = os.pwrite(fd, view[:1 << 30], offset)
            view, offset = view[n:], offset + n

    def _drain(self):
        while True:
            tokens, state, buffers, on_written = self._queue.get()
            try:
                if not self.contains(tokens):
                    path = self.write(tokens, state, buffers)
                    if path is not None and on_written is not None:
                        on_written()
                elif on_written is not None:
                    on_written()
            except Exception:
                LOG.exception('Prefix store: writer failed')
            finally:
                # Hold no snapshot while waiting for the next one: the host tier may already have dropped it.
                tokens = state = buffers = on_written = None
                self._queue.task_done()

    # -- quota, scanning and cleanup -------------------------------------------------------------------------------

    def _make_room(self, nbytes, keep):
        """Unlink the least recently used entries (never one being read or written) until ``nbytes`` more fit."""
        entries = []
        for path in self.dir.glob(f'*{SUFFIX}'):
            try:
                info = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                entries.append((info.st_mtime, path, info.st_size))
        used = sum(size for _, _, size in entries)
        with self._mutex:
            pinned = set(self._pinned) | {keep}
        for _, path, size in sorted(entries):
            if used + nbytes <= self.max_bytes:
                break
            if path in pinned:
                continue
            try:
                os.unlink(path)
            except OSError:
                continue
            used -= size
            self.stats['evictions'] += 1
            with self._mutex:
                self._index.pop(path, None)
        return used + nbytes <= self.max_bytes

    def _scan(self):
        for path in sorted(self.dir.glob(f'*{SUFFIX}')):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    header, size = self._header(fd)
                    tokens = self._tokens(fd, header)
                finally:
                    os.close(fd)
                self._index[path] = Record(path, header['identity'], tokens, size)
            except (CorruptEntry, OSError, ValueError, KeyError, TypeError) as exc:
                LOG.warning('Prefix store: removing unreadable %s (%s)', path.name, exc)
                self._discard(path)

    def _discard(self, path):
        with self._mutex:
            self._index.pop(path, None)
            if self._closed:
                return
        try:
            if stat.S_ISREG(os.lstat(path).st_mode):
                os.unlink(path)
        except OSError:
            pass

    # -- the format -----------------------------------------------------------------------------------------------

    @staticmethod
    def _header(fd):
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise CorruptEntry('not a regular file')
        prologue = os.pread(fd, len(MAGIC) + 8, 0)
        if len(prologue) != len(MAGIC) + 8 or prologue[:len(MAGIC)] != MAGIC:
            raise CorruptEntry('not a prefix snapshot')
        (length,) = struct.unpack('<Q', prologue[len(MAGIC):])
        if length > MAX_HEADER or len(MAGIC) + 8 + length > info.st_size:
            raise CorruptEntry('bad header length')
        header = json.loads(os.pread(fd, length, len(MAGIC) + 8))
        if not isinstance(header, dict) or header.get('format') != FORMAT or not isinstance(header.get('identity'), str):
            raise CorruptEntry('unsupported format')
        blobs = [header.get('tokens'), header.get('state')] + list(header.get('buffers') or [])
        if not isinstance(header.get('buffers'), list) or not blobs[0] or not blobs[1]:
            raise CorruptEntry('missing blobs')
        names = set()
        for blob in blobs:
            if (not isinstance(blob, dict) or not all(isinstance(blob.get(k), int) and not isinstance(blob.get(k), bool)
                                                      for k in ('offset', 'size'))
                    or not isinstance(blob.get('sha256'), str) or not isinstance(blob.get('name'), str)
                    or blob['offset'] < len(MAGIC) + 8 + length or blob['size'] < 0
                    or blob['offset'] + blob['size'] > info.st_size or blob['name'] in names):
                raise CorruptEntry('bad blob table')
            names.add(blob['name'])
        count = header['tokens'].get('count')
        if not isinstance(count, int) or count < 1 or header['tokens']['size'] != 4 * count:
            raise CorruptEntry('bad token count')
        return header, info.st_size

    @staticmethod
    def _blob(fd, blob):
        data = os.pread(fd, blob['size'], blob['offset'])
        if len(data) != blob['size'] or hashlib.sha256(data).hexdigest() != blob['sha256']:
            raise CorruptEntry(f'checksum mismatch in {blob["name"]}')
        return data

    def _tokens(self, fd, header):
        data = self._blob(fd, header['tokens'])
        return struct.unpack(f'<{header["tokens"]["count"]}i', data)
