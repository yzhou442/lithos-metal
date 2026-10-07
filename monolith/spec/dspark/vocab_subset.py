"""Draft vocabulary subset (FR-Spec style): the drafter proposes only tokens from a fixed id list.

The draft's base-logit projection (the shared vocabulary head) and the Markov W2 correction are restricted to the
list's rows, the argmax runs over the list, and its final stage maps the winning row back to the token id. Rows
are copied losslessly from the original packed slabs (same codes, scales and layout, fewer rows) into
content-addressed files outside the pack, so the target's head and every other consumer keep the full
vocabulary. Greedy output tokens never change: only the target's verify decides them; proposals outside the list
are simply never made (acceptance can drop where the target's token is outside it).

Applied to the emitted program before the drafter's region recipes, so the lm_head's tile operand layout is
derived from the subset slab. Ids are used in ascending order, so ties keep the lowest token id like the full argmax.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ...formats.base import PackLayout
from ...formats.blm import pack_blm, unpack_blm
from ...packs.packer import PackFile
from ...runtime.program import BufferSpec, KernelSpec

ROOT = Path(tempfile.gettempdir()) / 'monolith-vocab-subset'


def load_subset(spec: str, vocab: int) -> np.ndarray:
    """``path[:count]``: a JSON list of token ids, or a dict whose ``ranked`` list is cut to ``count``."""
    path, _, count = spec.partition(':')
    data = json.loads(Path(path).expanduser().read_text())
    ids = data['ranked'] if isinstance(data, dict) else data
    if count:
        ids = ids[:int(count)]
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if ids.size == 0 or ids[0] < 0 or ids[-1] >= vocab:
        raise ValueError('vocabulary subset ids must lie in [0, vocab)')
    return ids


def _slab_at(program, binding):
    """The pack slab whose bytes start at this weight binding: (PackFile, name)."""
    name, off = binding
    spec = program.buffers[name]
    if spec.role != 'weights' or spec.file is None:
        raise ValueError(f'{name}: vocabulary subsets need weights mapped from a pack file')
    absolute = spec.file_offset + off
    pack = PackFile(Path(spec.file).parent)
    hits = [n for n, s in pack.slabs.items() if s['offset'] == absolute]
    if len(hits) != 1:
        raise ValueError(f'{name}+{off}: no unique pack slab at offset {absolute}')
    return pack, hits[0]


def _subset_slab(pack: PackFile, slab: str, ids: np.ndarray):
    """A derived BLM slab holding rows ``ids`` in the original layout, plus its row scales."""
    s = pack.slabs[slab]
    key = hashlib.sha256(json.dumps([str(pack.dir.resolve()), slab, s['offset'], s['nbytes'],
                                     pack.manifest.get('created', '')]).encode() + ids.astype('<i8').tobytes()).hexdigest()
    ROOT.mkdir(exist_ok=True)
    path = ROOT / f'{key}.bin'
    info = pack.slab_info(slab)
    layout = PackLayout(rows=info.rows, lane_order=info.lane_order, scale_placement=info.scale_placement,
                        scale_order=info.scale_order)
    payload, scales = unpack_blm(pack.slab_bytes(slab).tobytes(), info)
    data, sub = pack_blm(payload[ids], None if scales is None else scales[ids], layout, format=info.format, k=info.k,
                         tensor_scale=info.tensor_scale, scale_group=info.scale_group, scale_dtype=info.scale_dtype)
    for field in ('rows', 'unit_bytes', 'payload_bytes', 'scale_bytes', 'lane_order', 'scale_placement',
                  'scale_unit_bytes', 'scale_lane_divisor', 'scale_order'):
        if getattr(sub, field) != getattr(info, field):
            raise ValueError(f'{slab}: subset changed the pack layout ({field})')
    page = os.sysconf('SC_PAGESIZE')
    aligned = (len(data) + page - 1) // page * page
    if not path.exists() or path.stat().st_size != aligned:
        with tempfile.NamedTemporaryFile(dir=ROOT, delete=False) as f:
            tmp = Path(f.name)
            try:
                f.write(data)
                f.write(bytes(aligned - len(data)))
                f.flush()
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
    rs = np.ascontiguousarray(pack.row_scales(slab)[ids], dtype=np.float32)
    return BufferSpec(aligned, role='weights', file=str(path)), sub, rs, key[:16]


def _private_kernel(program, op, suffix):
    key = f'{op.kernel}.{suffix}'
    if key not in program.kernels:
        program.kernels[key] = copy.deepcopy(program.kernels[op.kernel])
    op.kernel = key
    return program.kernels[key]


def _rebind(op, slot, name, off=0):
    op.bindings = [(s, name, off) if s == slot else (s, n, o) for s, n, o in op.bindings]


def _new_params(program, op, slot, name, patch):
    pname, poff = next((n, o) for s, n, o in op.bindings if s == slot)
    raw = bytearray(program.buffers[pname].init[poff:poff + 32])
    for offset, value in patch:
        struct.pack_into('<I', raw, offset, value)
    program.buffers[name] = BufferSpec(len(raw), bytes(raw), 'params')
    _rebind(op, slot, name)


def restrict_draft_vocab(program, ids: Sequence[int], *, vocab: int, base: str = 'draft.base_logits') -> int:
    """Rewrite the draft head, Markov W2 GEMVs and their argmaxes to the subset; returns the ops changed."""
    ids = np.asarray(ids, dtype=np.int64)
    m = int(ids.size)
    ops = program.ops
    heads = [o for o in ops if o.meta.get('kind') == 'lm_head' and any(s == 3 and n == base for s, n, _ in o.bindings)]
    if len(heads) != 1 or program.kernels[heads[0].kernel].function != 'gemm_tile':
        raise ValueError('vocabulary subset: expected one tensor-tile draft head writing ' + base)
    head = heads[0]
    w2 = [o for o in ops if o.name == 'gemv:draft.markov_w2.w2']
    if not w2 or any(program.kernels[o.kernel].function != 'gemv_T' for o in w2):
        raise ValueError('vocabulary subset: expected the Markov W2 shader GEMVs')
    changed = 0
    # 1. the head: rows ids of the shared vocabulary slab
    hk = program.kernels[head.kernel]
    pack, slab = _slab_at(program, next((n, o) for s, n, o in head.bindings if s == 0))
    spec, info, rs, tag = _subset_slab(pack, slab, ids)
    wname = f'vocab_subset.{tag}.head'
    program.buffers[wname] = spec
    _rebind(head, 0, wname)
    if 'ROW_SCALE_BITS' not in hk.macros:
        program.buffers[wname + '.row_scales'] = BufferSpec(rs.nbytes, rs.tobytes(), 'weights')
        _rebind(head, 1, wname + '.row_scales')
    tn = int(hk.macros['TN'].rstrip('u'))
    k = _private_kernel(program, head, 'vocab_subset')
    fields = dict(N_ROWS=m, N_TILES=-(-m // tn), N_BLOCKS=info.n_blocks, TILE0=0)
    for f, v in fields.items():
        if f'STATIC_GEMM_P_{f}' in k.macros:
            k.macros[f'STATIC_GEMM_P_{f}'] = f'{v}u'
    _new_params(program, head, 4, f'{wname}.params.{id(head)}', [(0, m), (4, -(-m // tn)), (20, 0), (24, info.n_blocks)])
    changed += 1
    # 2. the Markov W2 rows, residual = the subset base logits
    pack2, slab2 = _slab_at(program, next((n, o) for s, n, o in w2[0].bindings if s == 0))
    spec2, info2, rs2, tag2 = _subset_slab(pack2, slab2, ids)
    w2name = f'vocab_subset.{tag2}.w2'
    program.buffers[w2name] = spec2
    if any(next(o for s, n, o in op.bindings if s == 0) != next(o for s, n, o in w2[0].bindings if s == 0) for op in w2):
        raise ValueError('vocabulary subset: the Markov GEMVs read different slabs')
    rsname = None
    if 'ROW_SCALE_BITS' not in program.kernels[w2[0].kernel].macros:
        rsname = w2name + '.row_scales'
        program.buffers[rsname] = BufferSpec(rs2.nbytes, rs2.tobytes(), 'weights')
    full_row = vocab * 2
    for i, op in enumerate(w2):
        _rebind(op, 0, w2name)
        if rsname is not None:
            _rebind(op, 1, rsname)
        res = next((n, o) for s, n, o in op.bindings if s == 7)
        if res[0] != base or res[1] % full_row:
            raise ValueError('vocabulary subset: Markov residual is not a base-logit row view')
        _rebind(op, 7, base, (res[1] // full_row) * m * 2)
        k2 = _private_kernel(program, op, 'vocab_subset')
        for f, v in dict(N_ROWS=m, N_BLOCKS=info2.n_blocks, BLOCK0=0).items():
            if f'STATIC_GEMV_P_{f}' in k2.macros:
                k2.macros[f'STATIC_GEMV_P_{f}'] = f'{v}u'
        _new_params(program, op, 4, f'{w2name}.params.{i}', [(0, m), (4, info2.n_blocks), (28, 0)])
        changed += 1
    # 3. the argmaxes over the Markov logits: m candidates, the final stage maps the row to the token id
    table = f'vocab_subset.{tag}.ids'
    program.buffers[table] = BufferSpec(m * 4, ids.astype('<i4').tobytes(), 'weights')
    logits = {next(n for s, n, _ in op.bindings if s == 3) for op in w2}
    partials = {}
    for op in ops:
        fn = program.kernels[op.kernel].function
        if fn == 'argmax_partial' and next(n for s, n, _ in op.bindings if s == 0) in logits:
            pname = next(n for s, n, _ in op.bindings if s == 3)
            partials[pname] = op
            _new_params(program, op, 3, f'{table}.argmax.{len(partials)}', [(0, m), (12, -(-m // 256))])
            changed += 1
    for op in ops:
        if program.kernels[op.kernel].function != 'argmax_final':
            continue
        pname = next(n for s, n, _ in op.bindings if s == 3)
        if pname not in partials:
            continue
        mate = next(n for s, n, _ in partials[pname].bindings if s == 3)
        _rebind(op, 3, mate)
        k3 = _private_kernel(program, op, 'vocab_map')
        k3.macros['VOCAB_MAP'] = '1'
        op.bindings.append((4, table, 0))
        changed += 1
    return changed
