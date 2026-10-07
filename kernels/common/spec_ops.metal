#ifndef EOS_TEST
#define EOS_TEST (p.eos >= 0 && tok == p.eos)
#endif
// The DSpark round's small ops (design §5.8; issue #24). The compiler prepends the program's StepState struct.
//
// tap_concat:    x[t] = [tap_0[t] | tap_1[t] | … | tap_{N_SRC-1}[t]] for the rows the drafter injects this step —
//                the concatenated target features the feature projection (a plain gemv) reads. One SIMD-group per
//                (row, source); each tap is [T, k] BF16 (k % 8 == 0), the output [T, N_SRC·k]. Row count by T_SRC.
// confidence:    conf[k] = σ(w · [h_k ; emb_k] + b) for the γ block positions — h_k the drafter's normed block hidden
//                [γ, hidden] BF16, emb_k the Markov embedding W₁[prev_k] [γ, rank] BF16 (rank 0 = no Markov part),
//                w [hidden + rank] and b in FP32 (the BF16-valued checkpoint parameters, widened). The dot product
//                accumulates in FP32 and rounds to BF16 before the bias like the reference's BF16 linear; the logit
//                is divided by the position's STS temperature (params, 1 = uncalibrated) and the sigmoid runs in
//                FP32. One SIMD-group per position.
// verify_select: one thread. Advances the drafter's context length by the positions the draft pass injected, copies
//                the block's drafts and confidences into StepState and, outside a prefill chunk, chooses the verify
//                length L ≤ min(γ, t_max − 1): mode 0 = the reference's confident-prefix rule (the leading positions
//                with confidence ≥ threshold; threshold ≤ 0 verifies the whole block), mode 1 = the cost-aware rule
//                of design §5.8, L = argmax_l (1 + Σ_{i≤l} a_i) / cost[l] with a_i = Π_{j≤i} c_j the survival
//                probability of draft i and cost[l] the profile's cost of a (1 + l)-token target pass relative to a
//                single-token pass, mode 2 = a fixed L = threshold (the measurement's baseline). Then it sets the next
//                step's tokens pending_tokens = [anchor, d_0 … d_{L-1}] and t_this_step = 1 + L, and logs the block's
//                confidences at conf_log[step % log_cap][0 … 15] for the calibration. During a prefill chunk the
//                host feeds the next chunk and only the bookkeeping runs. ctx_cap > 0 (the target's KV capacity)
//                clamps L so the verify rows position … position + L stay inside the caches.
// accept_scan:   one thread; the step's closing SERIAL op when a drafter is wired in (replaces `advance`). A prefill
//                chunk advances the position and marks its t positions for injection. Otherwise the step fed
//                pending_tokens = [prompt tail …, anchor, d_0 … d_{L-1}] (L = verify_len; the anchor is at row
//                base = t − 1 − L, 0 in decode): the target's sampled tokens are compared with the drafts — accepted
//                = the matching prefix, bonus = the target's token after it; the accepted drafts and the bonus go to
//                the ring (sequence-tagged, stopping at the first EOS), position advances by the committed count
//                (the KV entries of rejected positions are overwritten by later steps), the anchor becomes the last
//                committed token, n_inject = checkpoint_index = base + committed (the rows whose target features
//                the drafter injects next and the state commit passes replay), t_this_step returns to 1 and `done`
//                is set at EOS. Each step logs (committed << 16) | (verify_len << 8) | accepted at log[step % log_cap]
//                (a prefill chunk logs (t << 16) | 0xFFFF) for the acceptance statistics. ctx_cap > 0 is the
//                program's context capacity (the target's KV rows, and the drafter's less the block it appends
//                after the context): a step whose first position reaches it sets error = 2 and done — the
//                pump's over-run past a request stops there instead of writing past a cache.
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef T_SRC
#define T_SRC 0
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
#ifndef N_SRC
#define N_SRC 1
#endif
#ifndef HIST
#define HIST 0                       // accept_scan: record every committed token at its position in hist (context lookup)
#endif
#ifndef HIST_CAP
#define HIST_CAP 0u
#endif
#ifndef LOOKUP
#define LOOKUP 0                     // verify_select: extend a whole-block verify with the context lookup's continuation
#endif
#ifndef LOOKUP_NMIN
#define LOOKUP_NMIN 2u               // shortest suffix match that proposes
#endif
#ifndef LOOKUP_NMAX
#define LOOKUP_NMAX 4u               // longest suffix compared (a longer match ranks first)
#endif
#ifndef LOOKUP_BASE
#define LOOKUP_BASE 64u              // drafter rows before the lookup continuation (min with the block: the whole block)
#endif
#ifndef LOOKUP_ROWS
#define LOOKUP_ROWS 16u              // continuation tokens the lookup reports
#endif
#ifndef LOOKUP_Q
#define LOOKUP_Q 0.6f                // cost rule: prior per-token acceptance of a continuation token (match >= 3)
#endif
#ifndef LOOKUP_Q2
#define LOOKUP_Q2 0.45f              // ... of a two-token match
#endif
#ifndef LOOKUP_ADAPT
#define LOOKUP_ADAPT 1               // cost rule: q from this request's measured continuation acceptance (prior LOOKUP_Q, weight 2)
#endif
#ifndef LOOKUP_DECAY
#define LOOKUP_DECAY 0.9f            // per extended round decay of the acceptance counts
#endif
#ifndef LOOKUP_PREV_SURVIVAL
#define LOOKUP_PREV_SURVIVAL 0.85f   // cost rule, ext_enable bit 1: after a wholly accepted block, the block's survival is at least this
#endif
#ifndef LOOKUP_MIN_SURVIVAL
#define LOOKUP_MIN_SURVIVAL 0.0f     // fixed rule: extend only when the block's survival (product of confidences) reaches this
#endif

struct ConcatParams { uint k; uint t_active; uint pad0; uint pad1; };
struct ConfParams { uint gamma; uint hidden; uint rank; uint pad; float sts[16]; };
struct SelectParams { uint gamma; float threshold; uint t_max; uint mode; float cost[16]; uint log_cap; uint ctx_cap; uint lm; uint ext_enable; };
struct AcceptParams { uint ring_cap; int eos; uint log_cap; uint ctx_cap; uint lm; uint pad0; uint pad1; uint pad2; };
// lm = 1: an LM drafter (design §5.8): drafter_ctx_len is the length of the committed prefix it has processed; the accept scan
// leaves in n_inject the committed rows it has not (the last ones of the step: 0 in decode unless every draft was accepted,
// the chunk in prefill) and in n_chain whether the step drafts (0 in a prefill chunk); the select records where the chain got to.

kernel void tap_concat(device const uint4* s0 [[buffer(0)]], device const uint4* s1 [[buffer(1)]], device const uint4* s2 [[buffer(2)]],
                       device const uint4* s3 [[buffer(3)]], device const uint4* s4 [[buffer(4)]], device const uint4* s5 [[buffer(5)]],
                       device const uint4* s6 [[buffer(6)]], device const uint4* s7 [[buffer(7)]], device uint4* out [[buffer(8)]],
                       constant ConcatParams& p [[buffer(9)]],
#if STEP_STATE
                       device const StepState* st [[buffer(15)]],
#endif
                       uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / N_SRC, s = sg % N_SRC;
#if STEP_STATE
  if (st->done || t >= ((T_SRC == 1) ? st->n_inject : ((T_SRC == 3) ? st->n_chain : ((T_SRC == 4) ? st->n_inject + st->n_chain : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step))))) return;
#else
  if (t >= p.t_active) return;
#endif
  const uint kw = p.k / 8u;
  device const uint4* src = s0;
  switch (s) {
    case 1: src = s1; break; case 2: src = s2; break; case 3: src = s3; break; case 4: src = s4; break;
    case 5: src = s5; break; case 6: src = s6; break; case 7: src = s7; break; default: break;
  }
  for (uint j = lane; j < kw; j += 32u) out[(ulong)t * (N_SRC * kw) + s * kw + j] = src[(ulong)t * kw + j];
}

kernel void confidence(device const ushort* hidden [[buffer(0)]], device const ushort* emb [[buffer(1)]], device const float* w [[buffer(2)]],
                       device const float* b [[buffer(3)]], device float* conf [[buffer(4)]], constant ConfParams& p [[buffer(5)]],
#if STEP_STATE
                       device const StepState* st [[buffer(15)]],
#endif
                       uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint k = gid / sw;
#if STEP_STATE
  if (st->done) return;
#endif
  if (k >= p.gamma) return;
  float acc = 0.0f;
  for (uint i = lane; i < p.hidden; i += 32u) acc = fma(as_type<float>(uint(hidden[k * p.hidden + i]) << 16), w[i], acc);
  for (uint i = lane; i < p.rank; i += 32u) acc = fma(as_type<float>(uint(emb[k * p.rank + i]) << 16), w[p.hidden + i], acc);
  acc = simd_sum(acc);
  if (lane == 0) {
    uint u = as_type<uint>(acc); u += 0x7FFFu + ((u >> 16) & 1u); const float r = as_type<float>(u & 0xFFFF0000u);   // the BF16 linear
    float z = r + b[0];
    uint v = as_type<uint>(z); v += 0x7FFFu + ((v >> 16) & 1u); z = as_type<float>(v & 0xFFFF0000u);                // BF16 + BF16
    conf[k] = 1.0f / (1.0f + exp(-z / p.sts[k]));
  }
}

kernel void verify_select(device const int* drafts [[buffer(0)]], device const float* conf [[buffer(1)]], device StepState* st [[buffer(2)]],
                          constant SelectParams& p [[buffer(3)]], device float* conf_log [[buffer(4)]],
#if LOOKUP
                          device const int* lookup [[buffer(5)]],
#endif
                          uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  if (p.lm) st->drafter_ctx_len = st->position + ((st->prefill_left > 0u) ? 0u : p.gamma);   // the ingest reached position; the chain added gamma rows
  else st->drafter_ctx_len = st->drafter_ctx_len + st->n_inject;    // the draft pass appended the injected positions
  st->n_inject = 0u;
  if (st->prefill_left > 0u) return;                          // the host feeds the next chunk
  for (uint k = 0; k < p.gamma; k++) {
    st->draft_tokens[k] = drafts[k];
    st->confidence[k] = conf[k];
    if (p.log_cap) conf_log[(st->step % p.log_cap) * 16u + k] = conf[k];
  }
  uint L = 0u;
  uint lmax = min(p.gamma, p.t_max - 1u);
  uint rows_cap = min(p.t_max, 16u) - 1u;                     // verify rows past the anchor: the decode bound (cost[16]) and the caches
  if (p.ctx_cap) {                                            // the verify rows must fit the target's caches
    if (st->position >= p.ctx_cap) { st->error = 2u; st->done = 1u; return; }
    lmax = min(lmax, p.ctx_cap - st->position - 1u);
    rows_cap = min(rows_cap, p.ctx_cap - st->position - 1u);
  }
#if LOOKUP
  // the lookup's continuation after the whole block: lookup[0] = suffix match length (0 = none), then the tokens
  // (a deeper block may instead hand the rows past its first LOOKUP_BASE drafts to the continuation)
  const uint base = min(p.gamma, LOOKUP_BASE);
  uint ext = 0u;
  const uint m = uint(max(lookup[0], 0));
  if (p.ext_enable && m >= LOOKUP_NMIN && lmax >= base) {
    while (base + ext < rows_cap && ext < LOOKUP_ROWS && lookup[1u + ext] >= 0) ext++;
  }
  float q = (m >= 3u) ? LOOKUP_Q : LOOKUP_Q2;
#if LOOKUP_ADAPT
  q = clamp((st->lookup_hits + 2.0f * q) / (st->lookup_trials + 2.0f), 0.0f, 0.99f);
#endif
#endif
  uint use_ext = 0u;                                          // lookup tokens appended after the block this step
#if LOOKUP
  // ext_enable bit 1: rounds that follow a wholly accepted block extend (code / file rewrites run in long accepted
  // stretches; the confidence chain alone under-predicts them). st->accepted is the round just verified.
  const bool prev_full = (p.ext_enable & 2u) != 0u && base == p.gamma && st->accepted >= p.gamma;
#endif
  if (p.mode == 2u) {
    L = min(uint(max(p.threshold, 0.0f)), lmax);
#if LOOKUP
    if (L == base && ext > 0u) {
      float surv = 1.0f;
      for (uint k = 0; k < base; k++) surv *= conf[k];
      if ((p.ext_enable & 2u) ? prev_full : surv >= LOOKUP_MIN_SURVIVAL) use_ext = ext;
    }
#endif
  } else if (p.mode == 1u) {
    float a = 1.0f, expect = 1.0f, best = 1.0f / p.cost[0];
#if LOOKUP
    float a_base = 1.0f, e_base = 1.0f;                       // survival and expectation through the lookup's base
#endif
    for (uint l = 1; l <= lmax; l++) {
      a *= conf[l - 1u];
      expect += a;
      const float score = expect / p.cost[l];
      if (score > best) { best = score; L = l; }
#if LOOKUP
      if (l == base) { a_base = a; e_base = expect; }
#endif
    }
#if LOOKUP
    if (ext > 0u) {                                           // the base's survival carries into the continuation
      // after a wholly accepted block the chain under-predicts the next one: lift its survival (the step's cost and
      // the request's measured q still decide, so deep contexts with a costly 9-16 step and poor lookups do not extend)
      float qa = prev_full ? max(a_base, LOOKUP_PREV_SURVIVAL) : a_base, e2 = e_base;
      for (uint x = 1; x <= ext; x++) {
        qa *= q;
        e2 += qa;
        const float score = e2 / p.cost[base + x];
        if (score > best) { best = score; L = base; use_ext = x; }
      }
    }
#endif
  } else if (p.threshold <= 0.0f) {
    L = lmax;
  } else {
    while (L < lmax && conf[L] >= p.threshold) L++;
  }
  st->gamma = p.gamma;                                        // with a continuation: the drafter rows before it (accept_scan)
  st->pending_tokens[0] = st->anchor;
  for (uint k = 0; k < L; k++) st->pending_tokens[k + 1u] = drafts[k];
#if LOOKUP
  for (uint k = 0; k < use_ext; k++) st->pending_tokens[L + 1u + k] = lookup[1u + k];
  if (use_ext) st->gamma = L;
  L += use_ext;
#endif
  st->verify_len = L;
  st->t_this_step = 1u + L;
}

kernel void accept_scan(device const int* token [[buffer(0)]], device StepState* st [[buffer(1)]], device ulong* ring [[buffer(2)]],
                        constant AcceptParams& p [[buffer(3)]], device uint* log [[buffer(4)]],
#if HIST
                        device int* hist [[buffer(5)]],
#endif
                        uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  const uint t = st->t_this_step;
#if HIST
  if (st->prefill_left > 0u) {                                // every prompt row is a committed token at its position
    for (uint r = 0; r < t; r++) if (st->position + r < HIST_CAP) hist[st->position + r] = st->pending_tokens[r];
  }
#endif
  if (st->prefill_left > 0u) {                                // a prefill chunk: nothing to verify, inject its positions
    if (p.log_cap) log[st->step % p.log_cap] = (t << 16) | 0xFFFFu;
    st->position = st->position + t;
    st->step = st->step + 1u;
    st->n_inject = t;
    st->checkpoint_index = t;
    st->n_chain = 0u;                                         // no drafting from a prefill chunk
    if (p.ctx_cap && st->position >= p.ctx_cap) { st->error = 2u; st->done = 1u; }   // the context is full
    return;
  }
  const uint L = st->verify_len;                              // 0 on the step that samples the first token
  const uint base = t - 1u - L;                               // the anchor's row (the last prompt position on the first step)
  uint acc = 0u;
  while (acc < L && token[base + acc] == st->pending_tokens[base + acc + 1u]) acc++;
  const int bonus = token[base + acc];
  uint head = st->ring_head;
  if (head + acc + 1u - st->ring_tail > p.ring_cap) { st->error = 1u; st->done = 1u; return; }   // ring overflow: the host fell behind
  uint committed = 0u;
  int last = bonus;
  bool stop = false;
  for (uint k = 0; k <= acc; k++) {
    const int tok = (k < acc) ? st->pending_tokens[base + k + 1u] : bonus;
    ring[head % p.ring_cap] = (ulong(head + 1u) << 32) | ulong(uint(tok));
    head++;
    committed++;
    last = tok;
    if (EOS_TEST) { stop = true; break; }   // nothing after the first EOS is committed
  }
  if (p.log_cap) log[st->step % p.log_cap] = (committed << 16) | (L << 8) | acc;
#if HIST
  if (L > st->gamma && acc >= st->gamma) {                    // a context-lookup continuation followed a whole accepted block
    const float hits = float(acc - st->gamma), trials = hits + ((acc < L) ? 1.0f : 0.0f);
    st->lookup_hits = st->lookup_hits * LOOKUP_DECAY + hits;
    st->lookup_trials = st->lookup_trials * LOOKUP_DECAY + trials;
  }
  for (uint r = 0; r <= base; r++) if (st->position + r < HIST_CAP) hist[st->position + r] = st->pending_tokens[r];   // prompt tail + anchor
  for (uint k = 0; k < committed; k++) {                      // the accepted drafts and the bonus, after the anchor
    const uint at = st->position + base + 1u + k;
    if (at < HIST_CAP) hist[at] = (k < acc) ? st->pending_tokens[base + k + 1u] : bonus;
  }
#endif
  st->ring_head = head;
  if (st->stop_at && head >= st->stop_at) stop = true;        // the request is served: the queued steps return at once
  st->accepted = acc;
  st->anchor = last;                                          // pending_tokens keeps the step's rows: an LM drafter's ingest reads them; the select rewrites them
  st->position = st->position + base + committed;
  st->step = st->step + 1u;
  st->verify_len = 0u;
  st->t_this_step = 1u;
  st->n_inject = base + committed;
  st->checkpoint_index = base + committed;
  st->n_chain = 1u;
  if (p.lm) st->n_inject = (st->position > st->drafter_ctx_len) ? st->position - st->drafter_ctx_len : 0u;   // the committed rows the LM drafter has not seen
  if (stop) st->done = 1u;
  if (p.ctx_cap && st->position >= p.ctx_cap) { st->error = 2u; st->done = 1u; }     // the context is full
}

// ngram_lookup: the context-lookup proposal for the next verify (spec campaign; prompt lookup / copy drafting).
// S = hist[0 .. position] (every committed token; hist[position] = the anchor) ++ the drafter's block d_0 … d_{γ-1}.
// Over every earlier end index j < |S| - 1 it measures how many trailing tokens of S match the tokens ending at j
// (up to LOOKUP_NMAX), keeps the longest match (the most recent on ties) and writes out[0] = its length (0 when
// shorter than LOOKUP_NMIN) and out[1 .. LOOKUP_ROWS] = S[j + 1 …], -1 past the end of S. One threadgroup.
struct LookupParams { uint gamma; uint cap; uint pad0; uint pad1; };
static inline int lookup_tok(device const int* hist, device const int* drafts, uint P, uint i) {
  return (i <= P) ? hist[i] : drafts[i - P - 1u];
}
kernel void ngram_lookup(device const int* hist [[buffer(0)]], device const int* drafts [[buffer(1)]], device int* out [[buffer(2)]],
                         constant LookupParams& p [[buffer(3)]],
#if STEP_STATE
                         device const StepState* st [[buffer(15)]],
#endif
                         uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint best;
#if STEP_STATE
  if (st->done) return;
  const uint P = min(st->position, p.cap - 1u);
  const bool active = st->prefill_left == 0u;
#else
  const uint P = 0u;
  const bool active = false;
#endif
  if (tid == 0) atomic_store_explicit(&best, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = P + 1u + p.gamma;                            // |S|
  if (active && n > LOOKUP_NMIN) {
    int tail[LOOKUP_NMAX];
    for (uint k = 0; k < LOOKUP_NMAX; k++) tail[k] = (k < n) ? lookup_tok(hist, drafts, P, n - 1u - k) : -2;
    for (uint j = tid; j + 1u < n; j += tpg) {
      uint m = 0;
      while (m < LOOKUP_NMAX && m <= j && lookup_tok(hist, drafts, P, j - m) == tail[m]) m++;
      if (m >= LOOKUP_NMIN) atomic_fetch_max_explicit(&best, (m << 24) | j, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint b = atomic_load_explicit(&best, memory_order_relaxed);
    const uint m = b >> 24, j = b & 0xFFFFFFu;
    out[0] = int(m);
    for (uint k = 0; k < LOOKUP_ROWS; k++) out[1u + k] = (m > 0u && j + 1u + k < n) ? lookup_tok(hist, drafts, P, j + 1u + k) : -1;
  }
}
