"""Exact-token prefix checkpoints on local disk, below PrefixCache's host copies.

An entry is one file: ``MAGIC``, the header length, a JSON header and the blobs it describes (the prompt's token
IDs as little-endian int32, the StepState and each state buffer), each at a 4 KiB-aligned offset with its SHA-256.
Files are written under ``tmp/`` and renamed into place, so a partial write is never an entry. The header names the
identity the bytes depend on (format, code, device, weights, layout; ``Session.prefix_identity``); an entry with
another identity is never matched, and one that fails any check is deleted and treated as a miss.

Entries live in ``entries/`` and outlive the process: a later server with the same identity reuses them. All
entries share one byte quota with least-recently-used eviction; entries being read (by any server: a shared flock is the read lease) or written are never evicted,
and a partial write that a dead process left is removed.
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
import secrets
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


def _pad(n):
    return -n % ALIGN


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        self.tmp = self.root / 'tmp'
        _private_dir(self.tmp)
        self._lock_fd = os.open(self.root / '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self._nonce = secrets.token_hex(4)
        self.dir = self.root / 'entries'
        _private_dir(self.dir)
        self._mutex = threading.Lock()
        self._pinned = set()          # paths being read or written
        self._index = {}              # path -> Record (this store's directory)
        self._counter = 0
        self._stamp = None            # the entries directory's mtime when this store last listed it
        self.stats = dict(hits=0, misses=0, writes=0, write_bytes=0, write_ms=0.0, read_ms=0.0,
                          evictions=0, errors=0, dropped=0)
        with self._dir_lock():
            self._remove_stale()
            self._refresh()
        self._queue = queue.Queue(maxsize=pending)
        self._writer = threading.Thread(target=self._drain, name='prefix-store', daemon=True)
        self._writer.start()
        LOG.info('Prefix store at %s (quota %.1f GB, %d entries)', self.dir, max_bytes / 1e9, len(self._index))

    # -- lookup ---------------------------------------------------------------------------------------------------

    def match(self, tokens):
        """The longest entry of this identity whose tokens are a strict prefix of ``tokens``."""
        tokens = tuple(tokens)
        self._refresh()
        with self._mutex:
            records = [r for r in self._index.values() if r.identity == self.identity
                       and len(r.tokens) < len(tokens) and r.tokens == tokens[:len(r.tokens)]]
        return max(records, key=lambda r: len(r.tokens), default=None)

    def contains(self, tokens):
        """Whether the entry for exactly ``tokens`` is on disk now. Another server sharing the root may have written
        or evicted it since this store indexed the directory: its name follows from the identity and the tokens."""
        tokens = tuple(tokens)
        path = self._path(tokens)
        try:
            info = os.lstat(path)
        except OSError:
            info = None
        with self._mutex:
            if info is None or not stat.S_ISREG(info.st_mode):
                self._index.pop(path, None)
                return False
            self._index.setdefault(path, Record(path, self.identity, tokens, info.st_size))
            return True

    def _path(self, tokens):
        digest = hashlib.sha256(struct.pack(f'<{len(tokens)}i', *tokens)).hexdigest()
        return self.dir / f'{self.identity[:16]}-{digest[:40]}{SUFFIX}'

    def remove(self, tokens):
        """Delete this identity's entry for exactly ``tokens``, if there is one."""
        tokens = tuple(tokens)
        with self._mutex:
            paths = [r.path for r in self._index.values() if r.identity == self.identity and r.tokens == tokens]
        for path in paths:
            self._discard(path)

    def load(self, record, expected):
        """Read and verify ``record``: ``(tokens, state, {name: bytes})``, or None (the entry is dropped).
        ``expected(names_sizes, n_tokens, state_size)`` raises CorruptEntry for buffers this program cannot take."""
        started = time.perf_counter()
        with self._mutex:
            self._pinned.add(record.path)
        try:
            fd = os.open(record.path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                # The read lease: no server sharing the root evicts an entry while it is read. One evicted between
                # the open and the lease is a miss, not damage: its name may already hold a new entry.
                fcntl.flock(fd, fcntl.LOCK_SH)
                if not self._is_current(fd, record.path):
                    with self._mutex:
                        self._index.pop(record.path, None)
                    return None
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
        except FileNotFoundError:                             # evicted by another server since the lookup
            with self._mutex:
                self._index.pop(record.path, None)
            return None
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
        """Finish queued writes."""
        self.flush()

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
            partial = self.tmp / f'{os.getpid()}-{self._nonce}-{self._counter}.partial'
            self._pinned.add(final)
        started = time.perf_counter()
        fd = None
        try:
            fd = self._reserve(total, final, partial)
            if fd is None:
                LOG.warning('Prefix store: a %.2f GB snapshot does not fit the %.1f GB quota; not saved',
                            total / 1e9, self.max_bytes / 1e9)
                return None
            self._write_all(fd, prologue, 0)
            for (_, data), blob in zip(named, placed):
                self._write_all(fd, data, blob['offset'])
            os.fsync(fd)
            os.close(fd)
            fd = None
            with self._dir_lock():        # another server counting the quota sees the partial or the entry
                if os.path.lexists(final):
                    # Another server wrote this prefix meanwhile: the name follows from the identity and the
                    # tokens, so its entry holds the same state. Replacing it would unlink a file under a read lease.
                    os.unlink(partial)
                else:
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

    def _dir_lock(self):
        store = self

        class Lock:
            def __enter__(self):
                fcntl.flock(store._lock_fd, fcntl.LOCK_EX)

            def __exit__(self, *exc):
                fcntl.flock(store._lock_fd, fcntl.LOCK_UN)
        return Lock()

    def _entries(self):
        """Every entry file and partial write under root: (path, size, mtime, evictable)."""
        out = []
        for path in self.dir.glob(f'*{SUFFIX}'):
            try:
                info = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                out.append((path, info.st_size, info.st_mtime, True))
        for path in self.tmp.glob('*.partial'):
            try:
                out.append((path, os.lstat(path).st_size, 0.0, False))
            except OSError:
                continue
        return out

    def _reserve(self, nbytes, keep, partial):
        """Evict least recently used entries until ``nbytes`` fit the quota, then create ``partial`` at that size
        under the same directory lock, so other processes count it. Returns its descriptor, or None."""
        if nbytes > self.max_bytes:
            return None
        with self._dir_lock():
            if not self._make_room(nbytes, keep):
                return None
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.ftruncate(fd, nbytes)
            except OSError:
                os.close(fd)
                raise
            return fd

    def _make_room(self, nbytes, keep):
        """With the directory lock held: remove partial writes that dead processes left, then unlink the least recently
        used entries (never a pinned one or a partial write) until ``nbytes`` more fit the quota."""
        entries = self._entries()
        used = sum(size for _, size, _, _ in entries)
        if used + nbytes > self.max_bytes:
            self._remove_stale()                  # a writer that died after this store started still holds quota
            entries = self._entries()
            used = sum(size for _, size, _, _ in entries)
        with self._mutex:
            pinned = set(self._pinned) | {keep}
        for path, size, _, evictable in sorted(entries, key=lambda e: e[2]):
            if used + nbytes <= self.max_bytes:
                break
            if not evictable or path in pinned or not self._unlink_unless_read(path):
                continue
            used -= size
            self.stats['evictions'] += 1
            with self._mutex:
                self._index.pop(path, None)
        return used + nbytes <= self.max_bytes

    @staticmethod
    def _unlink_unless_read(path):
        """Unlink an entry unless a server holds its read lease; True when the entry is gone."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.unlink(path)
            return True
        except OSError:                                       # BlockingIOError: being read
            return False
        finally:
            os.close(fd)

    @staticmethod
    def _is_current(fd, path):
        """Whether ``fd`` is still the file at ``path``."""
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return False
        mine = os.fstat(fd)
        return (info.st_dev, info.st_ino) == (mine.st_dev, mine.st_ino)

    def _remove_stale(self):
        """Remove partial writes whose process is gone."""
        for path in self.tmp.glob('*.partial'):
            try:
                pid = int(path.name.split('-', 1)[0])
            except ValueError:
                pid = None
            if pid is None or not _pid_alive(pid):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _refresh(self):
        """Index what other servers sharing the root wrote or evicted since this store last listed ``entries/``: a
        rename into the directory and an unlink both change its mtime, so an unchanged directory costs one stat."""
        try:
            stamp = os.stat(self.dir).st_mtime_ns
        except OSError:
            return
        if stamp == self._stamp:
            return
        self._stamp = stamp
        present = set(self.dir.glob(f'*{SUFFIX}'))
        with self._mutex:
            for path in [p for p in self._index if p not in present]:
                del self._index[path]
            new = sorted(p for p in present if p not in self._index)
        for path in new:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    header, size = self._header(fd)
                    tokens = self._tokens(fd, header)
                finally:
                    os.close(fd)
                with self._mutex:
                    self._index.setdefault(path, Record(path, header['identity'], tokens, size))
            except FileNotFoundError:
                continue                          # evicted since the listing
            except (CorruptEntry, OSError, ValueError, KeyError, TypeError) as exc:
                LOG.warning('Prefix store: removing unreadable %s (%s)', path.name, exc)
                self._discard(path)

    def _discard(self, path):
        with self._mutex:
            self._index.pop(path, None)
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
