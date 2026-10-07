"""Measured attention and projection geometry for 40-core M5 Max prefill.

The serving Qwen3.8-27B sweep covers worker counts, SIMD widths, query/key
tiles, key partitions, cached-prefix loads, projection tiles and K splits.
Keep choices here rather than duplicating the decode context recipes.
"""
import copy
import os
import struct

from ....compiler.attention_fusion import specialize_attention, compact_partials
from ....compiler.region_fusion import subprogram
from ....compiler.prefill import (projection_geometry, device_attention_tiles, packed_nvfp4_projection,
                                 packed_fp8_projection, direct_bf16_projection, shared_nvfp4_layout,
                                 shared_fp8_layout, input_tile_order, transposed_nvfp4_projection)

# Numerics of the 512-row prompt graph. EXACT_ATTENTION keeps the 16-query /
# 32-key attention core of smaller chunks (per-row softmax blocks identical to
# the 128-row graph) instead of the faster 32/128 device tiles; SHARED_READS
# 'transposed' reads the decoder's packed files in the private 128-column
# reduction order (same sums as the private layouts), 'retarget' writes x' in
# the files' order (faster for none measured, changes the in-tile sum order).
EXACT_ATTENTION = os.environ.get('LITHOS_PREFILL_EXACT_ATTENTION', '0') == '1'
SHARED_READS = os.environ.get('LITHOS_PREFILL_SHARED_READS', 'retarget')

# 512-row FP8 projections reading the verification graph's packed operands
# (32-row file tiles of 32 reduction columns) through 32x16x128 matrix tiles:
# (output rows per tile, SIMD groups per worker, workers). Measured 2026-10-06
# at T=512 against the private 16x128 layout: qkv 1.05 vs 1.13 ms, z 0.63 vs
# 0.66, out 0.61 vs 0.64 (sweep2_fp8_*.json in the prefill results).
SHARED_FP8_GEOMETRY = {
    ('fp8_e4m3', 10240, 5120): (16, 16, 80),
    ('fp8_e4m3', 6144, 5120): (16, 16, 40),
    ('fp8_e4m3', 8192, 5120): (16, 16, 80),
    ('fp8_e4m3', 5120, 6144): (16, 16, 80),
}


def shared_fp8_plan(program, fp8_rows):
    """Choose shared FP8 layouts, keeping a permuted input's readers on one reduction-tile order."""
    plan, readers = {}, {}
    for op in program.ops:
        k = program.kernels[op.kernel]
        if k.function != 'gemm_tile':
            continue
        xp = next(((name, offset) for slot, name, offset in op.bindings if slot == 2), None)
        readers.setdefault(xp, []).append(op)
        shape = (op.meta.get('format'), op.meta.get('n'), op.meta.get('k'))
        if (shape not in SHARED_FP8_GEOMETRY or op.meta.get('t_variant') != 512
                or k.macros.get('T_SRC', '0') not in ('0', '1') or int(k.macros.get('TK', '0').rstrip('u')) != 128):
            continue
        binding = next(((name, offset) for slot, name, offset in op.bindings if slot == 0), None)
        record = shared_fp8_layout(program, op, fp8_rows.get(binding)) if xp is not None else None
        if record is not None:
            plan[id(op)] = record
    for ops in readers.values():
        tks = {plan[id(op)]['tk'] if id(op) in plan else None for op in ops}
        if len(tks) != 1:
            for op in ops:
                plan.pop(id(op), None)
    return plan


def optimize(program, exact=False):
    """``exact``: keep every reduction order of the smaller-chunk prompt graph (the 128-row path): the attention
    core's 16-query / 32-key blocks, and the decoder's packed files read in each projection's own 128-column K order
    (transposed reads). Rows are independent in every kernel, so the 512-row graph then reproduces the 128-row
    graph's activations bit for bit while keeping the larger chunk's GEMM efficiency and the shared weights."""
    exact_attention = exact or EXACT_ATTENTION
    reads = 'transposed' if exact else SHARED_READS
    # Work on individual core/merge pairs. Draft attention and shapes not
    # measured by the prefill sweep retain their existing implementation.
    for core in list(program.ops):
        kernel = program.kernels[core.kernel]
        if (kernel.function != 'gqa_decode_mma' or kernel.macros.get('DRAFT') == '1'
                or kernel.macros.get('DIRECT_KV', '0') != '0'
                or kernel.macros.get('ADAPTIVE_CHUNK', '0') != '0'
                or int(kernel.macros.get('D', '0').rstrip('u')) != 256):
            continue
        bindings = {slot: (name, offset) for slot, name, offset in core.bindings}
        params, offset = bindings[9]
        rows = struct.unpack_from('<I', program.buffers[params].init, offset+8)[0]
        if rows < 32:
            continue
        workspace = bindings[7][0]
        merge = next(op for op in program.ops if program.kernels[op.kernel].function == 'gqa_merge'
                     and any(name == workspace for _, name, _ in op.bindings))
        part = copy.deepcopy(subprogram(program, [core, merge]))
        for i, op in enumerate(part.ops):
            private = f'{op.kernel}.prefill.{workspace}.{i}'
            part.kernels[private] = copy.deepcopy(part.kernels[op.kernel])
            op.kernel = private
        tuned = part.kernels[part.ops[0].kernel]
        sgs, groups = ((8, 80) if rows == 512 and not exact_attention else
                       ((8, 320) if exact_attention or rows < 256 else (4, 160)))
        tuned.macros.update(MMA_SG=str(sgs), STATIC_GQA_P_N_SG=f'{groups}u')
        record = bytearray(part.buffers[params].init)
        struct.pack_into('<I', record, offset+16, groups)
        part.buffers[params].init = bytes(record)
        part.ops[0].grid = (groups, 1, 1)
        part.ops[0].threadgroup = (sgs*32, 1, 1)
        part = specialize_attention(part, sgs, True, 64)
        if rows == 512 and not exact_attention:
            device_attention_tiles(part)
        compact_partials(part)
        # Preparation has explicit writes so lifetime analysis can reuse Q.
        for op in part.ops:
            if part.kernels[op.kernel].function == 'gqa_prepare_mma':
                op.meta['writes'] = [1, 2, 10]
        # A gate projection may overlap the attention core between these ops.
        program.ops[program.ops.index(merge)] = part.ops[-1]
        start = program.ops.index(core)
        program.ops[start:start+1] = part.ops[:-1]
        program.buffers.update(part.buffers)
        program.kernels.update(part.kernels)
    from ....compiler.fp8_tiles import projection_rows
    fp8_rows = projection_rows(program)
    retile = []
    fp8_plan = shared_fp8_plan(program, fp8_rows)
    for op in program.ops:
        k = program.kernels[op.kernel]
        if k.function == 'gdn_prepare':
            name, offset = next((n, off) for slot, n, off in op.bindings if slot == 9)
            if (struct.unpack_from('<3I', program.buffers[name].init, offset) == (48, 16, 512)
                    and int(k.macros.get('DK', '0').rstrip('u')) == 128
                    and int(k.macros.get('DV', '0').rstrip('u')) == 128):
                from ....compiler.prefill import shared_gdn_preparation
                shared_gdn_preparation(program, op)
            continue
        if (op.name == 'gdn_mixer' and k.macros.get('PREPARED') == '1'
                and k.macros.get('LOCAL_PREPARE', '0') == '0'
                and int(k.macros.get('DK', '0').rstrip('u')) == 128
                and int(k.macros.get('DV', '0').rstrip('u')) == 128):
            pn, off = next((n, o) for slot, n, o in op.bindings if slot == 9)
            hv, hk, rows = struct.unpack_from('<III', program.buffers[pn].init, off)
            if (hv, hk) == (48, 16) and rows >= 32:
                # Prepared inputs use one token's registers regardless of TP.
                # Keep the recurrence resident instead of storing every 8 rows.
                key = op.kernel+f'.prefill.tp{rows}'
                private = copy.deepcopy(k)
                private.macros['TP'] = f'{rows}u'
                program.kernels[key] = private
                op.kernel = key
            continue
        if (k.function != 'gemm_tile' or op.meta.get('t_variant', 0) < 32
                or k.macros.get('T_SRC', '0') not in ('0', '1')
                or int(k.macros.get('TK', '0').rstrip('u')) != 128):
            continue
        shape = (op.meta.get('format'), op.meta.get('n'), op.meta.get('k'))
        if id(op) in fp8_plan and not (exact and fp8_plan[id(op)]['tk'] != 32):
            # The decoder's FP8 operands (6.7 GiB otherwise duplicated): 128-column
            # matrix tiles assembled from its 32-column file tiles, half a file tile
            # of rows per matrix tile. Changes the reduction order (not the values)
            # of the products: outputs within one BF16 rounding of the private path.
            shared = fp8_plan[id(op)]
            tn, sgs, groups = SHARED_FP8_GEOMETRY[shape]
            tn = min(tn, shared['tn'])
            transposed = reads == 'transposed' and shared['tk'] == 32
            projection_geometry(program, op, tm=32, tn=tn, sgs=sgs, groups=groups, q_outer=None if transposed else 0)
            program.kernels[op.kernel].macros['FP8_DECODE'] = '1'
            binding = next((name, offset) for slot, name, offset in op.bindings if slot == 0)
            packed_fp8_projection(program, op, fp8_rows[binding], tile_block=shared['tile_block'], file_tk=shared['tk'],
                                  file_tn=shared['tn'], transposed=transposed)
            if not transposed:
                retile.append((op, shared['tk']))
            continue
        if shape == ('bf16', 96, 5120) and op.meta['t_variant'] == 512 and not exact:
            # Reorder native BF16 only; FP8 projections stay eight-bit.
            projection_geometry(program, op, tm=32, tn=16, sgs=4, groups=40)
            direct_bf16_projection(program, op)
        elif shape in (('nvfp4', 34816, 5120), ('nvfp4', 5120, 17408)) and op.meta['t_variant'] == 512 and (
                shared := shared_nvfp4_layout(program, op)) is not None and not (exact and shared['tk'] != 64):
            # Read the verification graph's packed MLP operands (32x64 tiles in
            # blocks of 32) instead of a private copy: within 1.5 % of the
            # 16x128 prompt layout, bit-identical outputs, and no 9-GiB layout
            # to evict and re-read whenever a request switches programs.
            # The down projection (K = 17408) multiplies each decoded weight tile with
            # two 32-token blocks: 1.68 -> 1.56 ms per op, identical outputs.
            if reads == 'transposed' and shared['tk'] == 64:
                gate_up = shape[0] == 34816
                projection_geometry(program, op, tm=32, tn=16 if gate_up else 32, sgs=16 if gate_up else 8,
                                    groups=160 if gate_up else 80, token_blocks=1 if gate_up else 2)
                transposed_nvfp4_projection(program, op, shared)
            else:
                projection_geometry(program, op, tm=32, tn=shared['tn'], sgs=8,
                                    groups=160 if shape[0] == 34816 else 80, tk=shared['tk'], q_outer=shared['outer'],
                                    token_blocks=1 if shape[0] == 34816 else 2)
                packed_nvfp4_projection(program, op, shared['tile_block'], shared['rows'])
                if shared['tk'] != 128:
                    retile.append((op, shared['tk']))
        elif shape == ('nvfp4', 34816, 5120):
            projection_geometry(program, op, tm=32, tn=16, sgs=16, groups=160)
            if op.meta['t_variant'] == 512:
                packed_nvfp4_projection(program, op)
        elif shape == ('nvfp4', 5120, 17408):
            projection_geometry(program, op, tm=16, tn=32, sgs=4 if op.meta['t_variant'] == 512 else 8, groups=80)
            if op.meta['t_variant'] == 512:
                packed_nvfp4_projection(program, op)
        elif shape == ('fp8_e4m3', 10240, 5120):
            large = op.meta['t_variant'] == 512
            projection_geometry(program, op, tm=32, tn=16, sgs=4 if large else 16, groups=160 if large else 80)
            if large:
                program.kernels[op.kernel].macros['FP8_DECODE'] = '1'
        elif shape in (('fp8_e4m3', 5120, 6144), ('fp8_e4m3', 6144, 5120),
                       ('fp8_e4m3', 8192, 5120)) and op.meta['t_variant'] == 512:
            groups = 40 if shape[1] == 6144 else 80
            projection_geometry(program, op, tm=32, tn=16, sgs=16, groups=groups)
            program.kernels[op.kernel].macros['FP8_DECODE'] = '1'
        if program.kernels[op.kernel].macros.get('FP8_DECODE') == '1':
            binding = next((name, offset) for slot, name, offset in op.bindings if slot == 0)
            packed_fp8_projection(program, op, fp8_rows[binding], tile_block=8 if shape[1] == 5120 else 1)
    for op, tk in retile:
        input_tile_order(program, op, tk)
    if any(op.meta.get('prefill_packed_nvfp4') or op.meta.get('prefill_packed_fp8')
           or op.meta.get('prefill_direct_bf16') for op in program.ops):
        from ....compiler.prefill import packed_projection_tails
        from ....compiler.weight_windows import compact_weights
        packed_projection_tails(program)
        compact_weights(program)
    program.kernels = {op.kernel: program.kernels[op.kernel] for op in program.ops}
    return program
