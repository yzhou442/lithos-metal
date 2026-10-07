// Stochastic sampling on the GPU (design D7; issue #23), appended to argmax.metal's helpers. Four dispatches:
//   sample_hist   — a histogram of the BF16 logits' monotone 16-bit keys per token (device atomics; 256 KB per token);
//   sample_select — one SIMD-group per token: scans the histogram from the top key down (lane ℓ owns a stripe of
//                   2048 keys, lane-prefix across stripes) for the logit thresholds of top-k (the k-th largest value:
//                   ties kept, like the HF warper), top-p (the bucket at which the descending cumulative softmax mass
//                   of logits / temperature first reaches p · Z) and min-p (max + temperature · log(min_p)); writes
//                   tau[t] = the most restrictive of the enabled ones and clears its stripe of the histogram;
//   sample_gumbel — argmax over the vocabulary of  logit / temperature + Gumbel noise  for logits ≥ tau[t] (Gumbel-max
//                   = a draw from the warped softmax), the noise from a counter-based hash keyed by (seed, step,
//                   token, index) so a run is reproducible bit for bit; partials as argmax_partial;
//   argmax_final  — as for greedy.
// Greedy (temperature 0) uses argmax_partial + argmax_final alone. Exact for BF16 logits: the thresholds are values
// of the 65536 possible logit codes, so no sort is needed.
struct SampleParams { uint vocab; uint t_active; uint n_sg; uint n_spans; uint top_k; float temperature; float top_p; float min_p;
                      uint seed_lo; uint seed_hi; uint step; uint flags; };   // flags: 1 top_k, 2 top_p, 4 min_p,
                                                                             // 8 top-p over the top-k renormalized (HF / sglang order)
#ifndef SPEC_STATS
#define SPEC_STATS 0                 // sample_select also writes stats[t] = (max logit, softmax mass of the kept logits)
#endif
#define HIST_KEYS 65536u
#define LANE_KEYS (HIST_KEYS / 32u)

static inline uint key16(ushort bits) { return (bits & 0x8000u) ? uint((~bits) & 0xFFFFu) : uint(bits | 0x8000u); }
static inline float key_val(uint key) { ushort bits = (key & 0x8000u) ? ushort(key & 0x7FFFu) : ushort(~key & 0xFFFFu); return bf16f(bits); }

static inline float gumbel(uint seed_lo, uint seed_hi, uint step, uint t, uint i) {
  ulong x = (ulong(seed_hi) << 32) | ulong(seed_lo);
  x += ulong(step) * 0x9E3779B97F4A7C15ul + ulong(t) * 0xC2B2AE3D27D4EB4Ful + ulong(i) * 0x165667B19E3779F9ul;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; x ^= x >> 31;   // splitmix64
  const float u = (float((x >> 41) & 0x7FFFFFul) + 0.5f) * (1.0f / 8388608.0f);   // (0, 1): 23 bits, so u < 1 stays representable in float32 (24 bits would round to 1 → −log(−log 1) = ∞)
  return -log(-log(u));
}

kernel void sample_hist(device const ushort* logits [[buffer(0)]], device atomic_uint* hist [[buffer(1)]], constant SampleParams& p [[buffer(3)]],
#if STEP_STATE
                        device const StepState* st [[buffer(15)]],
#endif
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = st->t_this_step;
#else
  const uint T_act = p.t_active;
#endif
  for (uint t = 0; t < T_act; t++) {
    device const ushort* row = logits + (ulong)t * p.vocab;
    device atomic_uint* h = hist + (ulong)t * HIST_KEYS;
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      for (uint e = 0; e < 8u; e++) {
        const uint i = base + e;
        if (i < p.vocab) atomic_fetch_add_explicit(h + key16(row[i]), 1u, memory_order_relaxed);
      }
    }
  }
}

// the softmax mass (of logits / temperature, relative to vmax) of this lane's stripe at keys whose value is >= th
static inline float stripe_mass_above(device atomic_uint* h, uint hi, float th, float vmax, float inv_t) {
  float m = 0.0f;
  for (uint j = 0; j < LANE_KEYS; j++) {
    const uint k = hi - j;
    const float v = key_val(k);
    if (v < th) break;                                               // keys descend within the stripe
    const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
    if (c != 0u) m += float(c) * exp((v - vmax) * inv_t);
  }
  return m;
}

kernel void sample_select(device atomic_uint* hist [[buffer(1)]], device float* tau [[buffer(2)]], constant SampleParams& p [[buffer(3)]],
#if SPEC_STATS
                          device float* stats [[buffer(4)]],
#endif
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
#if STEP_STATE
  if (st->done || t >= st->t_this_step) return;
#else
  if (t >= p.t_active) return;
#endif
  device atomic_uint* h = hist + (ulong)t * HIST_KEYS;
  const uint hi = HIST_KEYS - 1u - lane * LANE_KEYS;                 // this lane's stripe: keys (hi - LANE_KEYS, hi], descending
  const float inv_t = 1.0f / p.temperature;
  // pass a: the maximum logit (the highest non-empty key)
  uint kmax = 0u;
  for (uint j = 0; j < LANE_KEYS; j++) { const uint k = hi - j; if (atomic_load_explicit(h + k, memory_order_relaxed) != 0u) { kmax = k; break; } }
  const uint gmax = simd_max(kmax);
  const float vmax = key_val(gmax);
  // pass b: per-stripe count and softmax mass (of logits / temperature, relative to the max)
  uint cnt = 0u;
  float mass = 0.0f;
  for (uint j = 0; j < LANE_KEYS; j++) {
    const uint k = hi - j;
    const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
    if (c != 0u) { cnt += c; mass += float(c) * exp((key_val(k) - vmax) * inv_t); }
  }
  const uint cnt_before = simd_prefix_exclusive_sum(cnt);
  const float mass_before = simd_prefix_exclusive_sum(mass);
  const float z = simd_sum(mass);
  // top-k: the bucket where the descending cumulative count reaches k
  float tau_k = -INFINITY, tau_p = -INFINITY, tau_m = -INFINITY;
  if (p.flags & 1u) {
    const uint k_target = p.top_k;
    if (cnt_before < k_target && cnt_before + cnt >= k_target) {
      uint run = cnt_before;
      for (uint j = 0; j < LANE_KEYS; j++) { const uint k = hi - j; run += atomic_load_explicit(h + k, memory_order_relaxed); if (run >= k_target) { tau_k = key_val(k); break; } }
    }
    tau_k = simd_max(tau_k);
  }
  if (p.flags & 2u) {
    // flag 8: top-p within the top-k set, renormalized (HF warpers / sglang: top-k first, then top-p on its softmax)
    const float zk = ((p.flags & 9u) == 9u) ? simd_sum(stripe_mass_above(h, hi, tau_k, vmax, inv_t)) : z;
    const float target = p.top_p * zk;
    if (mass_before < target && mass_before + mass >= target) {
      float run = mass_before;
      for (uint j = 0; j < LANE_KEYS; j++) {
        const uint k = hi - j;
        const uint c = atomic_load_explicit(h + k, memory_order_relaxed);
        if (c != 0u) { run += float(c) * exp((key_val(k) - vmax) * inv_t); if (run >= target) { tau_p = key_val(k); break; } }
      }
    }
    tau_p = simd_max(tau_p);
  }
  if (p.flags & 4u) tau_m = vmax + p.temperature * log(p.min_p);
  const float th = max(max(tau_k, tau_p), tau_m);
  if (lane == 0) tau[t] = th;
#if SPEC_STATS
  // the kept distribution p(v) = exp((v - vmax) / T) / zkept for v >= th: what the exact speculative-sampling rule needs
  const float zkept = simd_sum(stripe_mass_above(h, hi, th, vmax, inv_t));
  if (lane == 0) { stats[2u * t] = vmax; stats[2u * t + 1u] = zkept; }
#endif
  for (uint j = 0; j < LANE_KEYS; j++) atomic_store_explicit(h + (hi - j), 0u, memory_order_relaxed);   // clear for the next step
}

kernel void sample_gumbel(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(2)]], constant SampleParams& p [[buffer(3)]],
                          device float* part_val [[buffer(4)]], device uint* part_idx [[buffer(5)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = st->t_this_step, step = st->step;
  const uint seed_lo = st->rng_lo, seed_hi = st->rng_hi;
#else
  const uint T_act = p.t_active, step = p.step, seed_lo = p.seed_lo, seed_hi = p.seed_hi;
#endif
  const float inv_t = 1.0f / p.temperature;
  for (uint t = 0; t < T_act; t++) {
    const float th = tau[t];
    float best = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    device const ushort* row = logits + (ulong)t * p.vocab;
    for (uint s = sg; s < p.n_spans; s += p.n_sg) {
      const uint base = s * 256u + lane * 8u;
      for (uint e = 0; e < 8u; e++) {
        const uint i = base + e;
        if (i < p.vocab) {
          const float v = bf16f(row[i]);
          if (v >= th) better(best, bi, v * inv_t + gumbel(seed_lo, seed_hi, step, t, i), i);
        }
      }
    }
    simd_argmax(best, bi);
    if (lane == 0) { part_val[t * p.n_sg + sg] = best; part_idx[t * p.n_sg + sg] = bi; }
  }
}


// ---- exact speculative sampling with sampled drafts (spec campaign; sglang chain_speculative_sampling semantics) ----
// The draft side samples d_k ~ q_k = softmax(corrected_k / T) over the whole vocabulary (no truncation), Gumbel-max
// with noise stream 2000 + k, and keeps the corrected logits row (q_logits[k]) and log Σ exp(corrected_k / T)
// (q_lse[k]) in persistent state. The verify side computes p (the target's kept softmax, from sample_select's stats)
// and accepts drafter row r with probability min(1, p(d_r) / q(d_r)) (uniform stream 3000 + r); rows a context lookup
// filled (r >= StepState.gamma: point-mass proposals) keep "accept iff the target's sample equals the draft", whose
// rejection already yields p without the draft. At the first rejected drafter row the correction is drawn from
// norm(max(p - q, 0)) (Gumbel stream 4000); with every draft accepted the bonus stays the target's sample at row L.
// The accept scan then sees token[] = [d_0 … d_{j-1}, correction] and needs no change.
static inline float uniform01(uint seed_lo, uint seed_hi, uint step, uint t, uint i) {
  ulong x = (ulong(seed_hi) << 32) | ulong(seed_lo);
  x += ulong(step) * 0x9E3779B97F4A7C15ul + ulong(t) * 0xC2B2AE3D27D4EB4Ful + ulong(i) * 0x165667B19E3779F9ul;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; x ^= x >> 31;
  return (float((x >> 41) & 0x7FFFFFul) + 0.5f) * (1.0f / 8388608.0f);
}

#ifndef DRAFT_K
#define DRAFT_K 0u
#endif
#define Q_STREAM 2000u
#define U_STREAM 3000u
#define R_STREAM 4000u

// one draft position: Gumbel-max over corrected / T, the online log-sum-exp, and a copy of the row into q_logits[k]
kernel void draft_q_partial(device const ushort* logits [[buffer(0)]], device float* part_val [[buffer(1)]], device uint* part_idx [[buffer(2)]],
                            constant SampleParams& p [[buffer(3)]], device float* part_m [[buffer(4)]], device float* part_s [[buffer(5)]],
                            device ushort* q_logits [[buffer(6)]],
#if STEP_STATE
                            device const StepState* st [[buffer(15)]],
#endif
                            uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg) return;
#if STEP_STATE
  if (st->done) return;
  const uint step = st->step, seed_lo = st->rng_lo, seed_hi = st->rng_hi;
#else
  const uint step = p.step, seed_lo = p.seed_lo, seed_hi = p.seed_hi;
#endif
  const float inv_t = 1.0f / p.temperature;
  device ushort* qrow = q_logits + (ulong)DRAFT_K * p.vocab;
  float best = -INFINITY, m = -INFINITY, sum = 0.0f;
  uint bi = 0xFFFFFFFFu;
  for (uint s = sg; s < p.n_spans; s += p.n_sg) {
    const uint base = s * 256u + lane * 8u;
    for (uint e = 0; e < 8u; e++) {
      const uint i = base + e;
      if (i < p.vocab) {
        const ushort bits = logits[i];
        qrow[i] = bits;
        const float x = bf16f(bits) * inv_t;
        better(best, bi, x + gumbel(seed_lo, seed_hi, step, Q_STREAM + DRAFT_K, i), i);
        if (x > m) { sum = sum * exp(m - x) + 1.0f; m = x; } else { sum += exp(x - m); }
      }
    }
  }
  simd_argmax(best, bi);
  const float mm = simd_max(m);
  const float ss = simd_sum((m == -INFINITY) ? 0.0f : sum * exp(m - mm));
  if (lane == 0) { part_val[sg] = best; part_idx[sg] = bi; part_m[sg] = mm; part_s[sg] = ss; }
}

kernel void draft_q_final(device const float* part_val [[buffer(0)]], device const uint* part_idx [[buffer(1)]], device int* token [[buffer(2)]],
                          constant SampleParams& p [[buffer(3)]], device const float* part_m [[buffer(4)]], device const float* part_s [[buffer(5)]],
                          device float* q_lse [[buffer(6)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if (st->done) return;
#endif
  float best = -INFINITY, m = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = lane; i < p.n_sg; i += 32u) { better(best, bi, part_val[i], part_idx[i]); m = max(m, part_m[i]); }
  simd_argmax(best, bi);
  const float mm = simd_max(m);
  float s = 0.0f;
  for (uint i = lane; i < p.n_sg; i += 32u) if (part_m[i] > -INFINITY) s += part_s[i] * exp(part_m[i] - mm);
  s = simd_sum(s);
  if (lane == 0) { token[0] = int(bi); q_lse[DRAFT_K] = mm + log(s); }
}

#if STEP_STATE                       // the verify-side kernels read the step's rows from StepState
static inline float spec_p(device const ushort* lrow, device const float* tau, device const float* stats, uint row, uint v, float inv_t) {
  const float z = bf16f(lrow[v]);
  return (z >= tau[row]) ? exp((z - stats[2u * row]) * inv_t) / stats[2u * row + 1u] : 0.0f;
}

// one SIMD-group: the accept test for every verified draft, then (lane 0) the token rewrite for the accept scan;
// flag[0] = the absolute row of a rejected drafter row whose correction the residual pass draws (0xFFFFFFFF: none)
kernel void spec_q_accept(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(1)]], device const float* stats [[buffer(2)]],
                          constant SampleParams& p [[buffer(3)]], device const ushort* q_logits [[buffer(4)]], device const float* q_lse [[buffer(5)]],
                          device int* token [[buffer(6)]], device uint* flag [[buffer(7)]],
                          device const StepState* st [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
  if (st->done) return;
  if (lane == 0) flag[0] = 0xFFFFFFFFu;
  if (st->prefill_left > 0u) return;
  const uint L = st->verify_len, t = st->t_this_step;
  if (L == 0u || t < L + 1u) return;
  const uint base = t - 1u - L;
  const uint drafted = min(L, st->gamma);                            // rows past it hold point-mass lookup proposals
  const float inv_t = 1.0f / p.temperature;
  bool ok = true;                                                    // this lane's row verdict
  if (lane < L) {
    const uint r = lane, row = base + r;
    const int d = st->pending_tokens[row + 1u];
    if (r < drafted) {
      const float pd = spec_p(logits + (ulong)row * p.vocab, tau, stats, row, uint(d), inv_t);
      const float qd = exp(bf16f(q_logits[(ulong)r * p.vocab + uint(d)]) * inv_t - q_lse[r]);
      const float u = uniform01(st->rng_lo, st->rng_hi, st->step, U_STREAM + r, 0u);
      ok = u * qd < pd;
    } else {
      ok = token[row] == d;
    }
  }
  const uint verdicts = uint(ulong(simd_ballot(!ok && lane < L)));  // bit r: row r rejects
  if (lane == 0) {
    const uint j = (verdicts != 0u) ? ctz(verdicts) : L;             // the first rejection (L: all accepted)
    for (uint r = 0; r < j; r++) token[base + r] = st->pending_tokens[base + r + 1u];
    if (j < L && j < drafted) flag[0] = base + j;                    // a drafter row: the correction comes from the residual
    // (a lookup row keeps the target's own sample, already != the draft; all accepted: token[base + L] is the bonus)
  }
}

// the correction at the rejected drafter row: Gumbel-max over log max(p - q, 0) (noise stream 4000); partials as argmax
kernel void spec_residual_partial(device const ushort* logits [[buffer(0)]], device const float* tau [[buffer(1)]], device const float* stats [[buffer(2)]],
                                  constant SampleParams& p [[buffer(3)]], device const ushort* q_logits [[buffer(4)]], device const float* q_lse [[buffer(5)]],
                                  device const uint* flag [[buffer(7)]], device float* part_val [[buffer(8)]], device uint* part_idx [[buffer(9)]],
                                  device const StepState* st [[buffer(15)]],
                                  uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  if (sg >= p.n_sg || st->done) return;
  const uint row = flag[0];
  if (row == 0xFFFFFFFFu) return;
  const uint r = row - (st->t_this_step - 1u - st->verify_len);     // the draft position of that row
  const float inv_t = 1.0f / p.temperature;
  device const ushort* lrow = logits + (ulong)row * p.vocab;
  device const ushort* qrow = q_logits + (ulong)r * p.vocab;
  const float lse = q_lse[r];
  float best = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint s = sg; s < p.n_spans; s += p.n_sg) {
    const uint base = s * 256u + lane * 8u;
    for (uint e = 0; e < 8u; e++) {
      const uint i = base + e;
      if (i < p.vocab) {
        const float res = spec_p(lrow, tau, stats, row, i, inv_t) - exp(bf16f(qrow[i]) * inv_t - lse);
        if (res > 0.0f) better(best, bi, log(res) + gumbel(st->rng_lo, st->rng_hi, st->step, R_STREAM, i), i);
      }
    }
  }
  simd_argmax(best, bi);
  if (lane == 0) { part_val[sg] = best; part_idx[sg] = bi; }
}

kernel void spec_residual_final(device const float* part_val [[buffer(0)]], device const uint* part_idx [[buffer(1)]], device int* token [[buffer(2)]],
                                constant SampleParams& p [[buffer(3)]], device const uint* flag [[buffer(7)]],
                                device const StepState* st [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
  if (st->done) return;
  const uint row = flag[0];
  if (row == 0xFFFFFFFFu) return;
  float best = -INFINITY;
  uint bi = 0xFFFFFFFFu;
  for (uint i = lane; i < p.n_sg; i += 32u) better(best, bi, part_val[i], part_idx[i]);
  simd_argmax(best, bi);
  if (lane == 0 && bi != 0xFFFFFFFFu) token[row] = int(bi);
}
#endif  // STEP_STATE
