# AGENTS.md

Guidance for coding agents and automated reviewers working in lithos-metal (an LLM inference engine for Apple
silicon; Python package `monolith`). Architecture, design decisions, working rules and the reading order are in
[CLAUDE.md](CLAUDE.md); contribution rules and test tiers are in [CONTRIBUTING.md](CONTRIBUTING.md). Read those first.

Checks that CI runs (all CPU-only, any OS): `python -m pytest tests/contract`, `python tools/ci/hygiene.py`,
`python -m compileall -q monolith tools tests` (Python 3.10 and 3.13 must both work). GPU tiers (`tests/kernels`,
`tests/runtime`, `tests/models`) need a Mac with the native module built and cannot run on hosted runners.

## Code Review Rules

This is the section Codex follows when reviewing pull requests. Flag these, in priority order:

1. **Greedy token identity.** A change presented as lossless (kernels, scheduling, fusion, recipes, speculative
   decoding, prefill) must leave greedy output token-identical to the engine before it. Speculative decoding is
   exact: accept/commit, rollback, verify-length selection and drafter changes may change speed, never the tokens.
   Flag any path that can commit a token the target did not produce, skip a rollback of recurrent (GDN) state or
   KV, or read a draft row past the verified length.
2. **Numerics changes only behind flags.** Anything that alters floating-point results (reduction order, tile or
   K-split shapes, chunk sizes, number formats, re-quantization, fast math) must be opt-in through an explicit option
   or recipe key that defaults to off, and the PR must say so. The numerics contract is in CLAUDE.md (BF16 residual
   stream, FP32 accumulators and recurrent state, layers cos > 0.999 vs the HF reference).
3. **Metal-specific correctness traps.**
   * Every kernel store must stay inside its binding (buffers are separate allocations; an overrun corrupts a
     neighbour such as StepState or a params record and shows up later as wrong tokens or a hang). Check index math
     against the buffer sizes the compiler allocates, including ragged tails and the largest `t_max` / context.
   * Indirect command buffer bindings: a buffer bound with an offset (`setKernelBuffer:offset:atIndex:`) must use
     the offset the compiler recorded for that slot; a params record belongs to one program and is never shared.
   * Threadgroup memory: static plus dynamic threadgroup memory must stay within the device limit (32 KB on current
     Apple GPUs) for every geometry the compiler can choose, not only the default recipe.
   * Ordering: a dispatch that reads another's output needs the ICB barrier (`barrier_before`) or a dispatch
     boundary; threadgroup-memory reuse needs `threadgroup_barrier`. Correctness may depend only on documented Metal
     semantics, never on observed scheduling.
   * GPU loops must be bounded; no dispatch may spin on a condition another dispatch sets.
4. **StepState layout stability.** `monolith/core/step_state.py` fields are read by kernels at fixed offsets. New
   fields are appended (never inserted or reordered), the golden offsets in `tests/contract/test_step_state.py` are
   updated in the same change, and every kernel/host reader uses the layout's offsets rather than literals.
5. **No device- or machine-specific paths.** No absolute paths, user names, host names or local checkpoint
   locations in code, tests, docs or tool defaults; chip-specific values live in the backend profiles and recipes
   under `monolith/backends/metal/<chip>/`, selected through the registry. Model names appear only under
   `monolith/models/<name>/` (`tools/ci/hygiene.py`).
6. **Defaults and compatibility.** New serving options must keep the previous behaviour by default unless the PR
   states otherwise with measurements; recipe JSON changes must keep older keys readable. Performance claims need
   a paired A/B table (same machine, alternating runs, several repetitions).

Style nits, naming and formatting are not worth a finding unless they hide a bug.
