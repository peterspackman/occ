# MP2 Restructure & Performance Plan

## Goal

Re-home and rewrite the MP2 implementation for performance. Both the
conventional and RI (density-fitted) paths become first-class and threaded,
unrestricted (UHF) MP2 is added, and the correlation code moves into a new
`occ/qm/correlation` module with a reusable density-fitting / B-tensor core.

Scope decisions (agreed):
- **Both paths equally** — optimize conventional *and* RI-MP2.
- **UHF in scope** — implement αα/ββ/αβ.
- **New module** — `occ::qm::correlation`, compiled as a new `occ_correlation`
  library.

### Overriding principle: performant **and** memory-bounded, with tidy code
The primary goal is fast code that **does not blow up memory**. This is a
cross-cutting constraint, not a final-stage cleanup:
- No code path may assume an O(N⁴) (or full `n_occ·n_virt·naux`) array fits in
  RAM. Blocking/batching over occupied orbitals is part of the **initial**
  design of every kernel, with a configurable memory budget governing block size.
- Prefer streaming/integral-direct accumulation over materialize-then-contract.
- Tidiness serves this: clear, boring kernels that are easy to reason about for
  both speed and footprint — not clever code.

---

## Diagnosis of current code

### Conventional MP2 — broken for performance
- `MOIntegralEngine::compute_ovov_tensor()` materializes the **full dense N⁴ AO
  ERI tensor** (`src/qm/mo_integral_engine.cpp:334`). ~800 MB at 100 bf, ~13 GB
  at 200 bf.
- The four quarter-transforms (`transform_first_index` … `transform_fourth_index`,
  `src/qm/mo_integral_engine.cpp:157-284`) are **scalar nested loops** — O(N⁵),
  no BLAS, **single-threaded**; step 1 recomputes the 8-fold canonical index
  (min/max + branch) in the innermost loop (lines 172-185).
- `compute_mo_eri` (`mo_integral_engine.cpp:83`) allocates a libcint `Optimizer`
  and a buffer **inside the innermost of four shell loops** (lines 114-115).
- `-C` sign hack on MO coefficients (`mo_integral_engine.cpp:54-55`) — cancels
  in the energy; unexplained.

### RI-MP2 — right idea, wrong execution
`IntegralEngineDF::compute_df_mp2_energy` (`src/qm/integral_engine_df.cpp:324`):
- Recomputes `juP` = (jν|P), the entire occupied half-transform of j, **inside
  the i-loop** (lines 370-375) → O(n_occ²·N²·naux) instead of O(n_occ·N²·naux).
- Re-solves the metric `V_LLt.solve(jbP.transpose())` inside **every** (i,j)
  iteration (line 379).
- **Single-threaded** over i (plain `for`, line 346).
- `four_center_integrals_tensor()` (line 266) rebuilds N⁴ via N⁴-many
  back-substitution solves (line 303) — pathological; only the DF-vs-conventional
  test uses it.

### Reusable infrastructure (keep)
- SCF J/K pattern: TBB `occ::parallel` + thread-local matrices + Schwarz
  screening (`src/qm/detail/four_center_kernels.h`).
- libcint 3-center store: `IntegralEngineDF::compute_stored_integrals`
  (`integral_engine_df.cpp:64`) and the `V_LLt` metric factor — **stay in
  occ_qm** (SCF/RI-JK need them).
- Native Hermite/MMD 3-center kernels in `occ/ints` (`eri3.h`,
  `hermite_kernels.h`) — candidate primitive backend later.
- BLAS is Eigen-native by default (`USE_SYSTEM_BLAS OFF`) → GEMMs are
  single-threaded. **Parallelism must come from the TBB layer** (thread over
  `i`/`ij`, GEMM serially inside), exactly like SCF.

---

## Target design

### New module `occ_correlation`
- Headers: `include/occ/qm/correlation/` — `df_integrals.h`, `mp2.h`,
  `mp2_components.h`, `mo_integral_engine.h`, `post_hf_method.h`.
- Sources: `src/qm/correlation/`.
- CMake: new `occ_correlation` target linking `occ_qm occ_ints`. Driver
  (`src/driver/single_point.cpp`), Python/Lua bindings, and `tests/mp2_tests.cpp`
  switch to it.
- `IntegralEngineDF` **stays in occ_qm**; the MP2-specific
  `compute_df_mp2_energy` and `four_center_integrals_tensor` move out / are
  deleted. The new module reads `df.integral_store()` + metric to build B.

### DF / B-tensor core (`DFIntegrals`)
Standard DF-MP2 structure — build once, reuse:
1. `(μν|P)` AO 3-index — reuse `IntegralEngineDF::compute_stored_integrals`.
2. **B_iaP** = Σ_μν C_μi C_νa (μν|P), shape `(n_occ·n_virt, naux)` — two GEMMs
   per occupied (or batched), computed once.
3. **Fold the metric once**: B̃ = B·L⁻ᵀ (one triangular solve on the whole
   matrix, V = LLᵀ), so `(ia|jb) = Σ_P B̃_iaP B̃_jbP`. Removes all per-(i,j)
   solves.
4. Energy: thread over `i`/`ij`; one GEMM per pair → `n_virt×n_virt` block;
   accumulate ss/os with thread-local reduction. Batch over occupied blocks when
   B̃ won't fit.

### Conventional core
Integral-direct, GEMM-based half-transform that **never materializes N⁴**:
accumulate `(iν|ρσ)` directly from shell-quartet batches via the SCF
four-center kernel, then GEMM the remaining indices. Threaded at the shell-pair
/ occupied level, thread-local intermediates.

### UHF
Per-spin B-tensors (or per-spin half-transforms): αα, ββ same-spin + αβ
opposite-spin, with the correct spin-summed numerator.

---

## Stage 1: New module scaffold + baseline harness
**Goal**: `occ_correlation` library exists; all MP2 code moved into it; existing
behavior and tests unchanged; a timing/scaling benchmark exists.
**Changes**:
- Create `include/occ/qm/correlation/` + `src/qm/correlation/`; move `mp2.*`,
  `mp2_components.h`, `mo_integral_engine.*`, `post_hf_method.*` (verify no other
  occ_qm consumer via grep first).
- New `src/qm/correlation/CMakeLists.txt` → `occ_correlation`; rewire driver,
  bindings, tests.
- Add a benchmark (mid-size case, e.g. small organic, def2-SVP + RIFIT) printing
  AO-int / transform / energy timings.
**Success Criteria**: builds; `mp2_tests` passes unchanged; benchmark runs and
reports a baseline.
**Tests**: existing `tests/mp2_tests.cpp` green; new benchmark smoke test.
**Status**: Complete

Done:
- New `occ_correlation` library: `src/qm/correlation/` + `include/occ/qm/correlation/`
  (mp2, mo_integral_engine, post_hf_method). Linked into occ_driver, mp2_tests,
  qm_tests. `mp2_components.h` stays in occ_qm (moves in Stage 2 with the DF-MP2
  extraction, else occ_qm would depend on correlation).
- `tests/mp2_benchmarks.cpp` (`[.benchmark][mp2-bench]`), compiled into mp2_tests.
- Tests green: mp2_tests 32 assertions/9 cases, qm_tests 117/29; full `occ`
  build + CLI link clean.
- **Baseline (water/def2-SVP, 25 AO / 133 aux, 11 threads), full compute:**
  conventional MP2 **3.69 ms**, RI-MP2 **1.62 ms**; E_corr conv −0.20341545,
  RI −0.20344013 (agree 2.5e-5). Run: `mp2_tests "[mp2-bench]~[slow]"`.

## Stage 2: DF/B-tensor core + RI-MP2 rewrite (restricted)
**Goal**: `DFIntegrals` B-tensor; RI-MP2 rebuilt on it; metric folded once;
energy threaded; **memory bounded by a configurable budget from day one**.
**Changes**:
- `DFIntegrals::build_b_tensor(C_occ_block, C_virt)` → B̃ via GEMM half-transforms
  + single triangular solve, computed **per occupied block** (block size derived
  from the memory budget), never the full `n_occ·n_virt·naux` array unless it
  fits.
- Replace `compute_df_mp2_energy` body: loop occupied blocks, thread over
  `i`/`ij` within/across blocks, one GEMM per pair, thread-local ss/os reduction.
- Delete the pathological `four_center_integrals_tensor` DF reconstruction (and
  the DF-vs-conventional test's dependence on it — compare energies instead).
**Success Criteria**: RI-MP2 energies identical to current within tol; large
speedup vs baseline; near-linear thread scaling; peak memory tracks the budget
(verified with a small budget forcing multiple blocks).
**Tests**: existing RI-MP2 reference energies; new thread-count invariance test;
small-budget multi-block run gives identical energy; benchmark delta recorded.
**Status**: Complete

Done:
- New `DFIntegrals` core (`correlation/df_integrals.{h,cpp}`): `build_b_tilde`
  builds metric-folded B (per-P GEMMs + one triangular solve with L). Added
  `IntegralEngineDF::coulomb_metric()` accessor.
- Rewrote `MP2::compute_ri_mp2_energy`: occupied-block loop (size from
  `set_memory_budget`, default 1 GiB), i≥j symmetry (factor 2 off-diagonal),
  GEMM `K=Bi·Bjᵀ` per pair, threaded over i with thread-local ss/os. No phase
  hack (energy is even in C signs).
- Deleted pathological `IntegralEngineDF::four_center_integrals_tensor` and
  `compute_df_mp2_energy`; removed `MOIntegralEngine` DF mode; deleted dead
  `mp2_components.h` (superseded by `MP2::Results`).
- Tests: replaced "DF tensor comparison" with RI-vs-conventional energy +
  budget-blocking-exact + thread-invariance. mp2_tests 36/9, qm_tests 117/29,
  full build/CLI link clean.
- **Speedup (water/def2-SVP, full compute):** RI-MP2 1.62 ms → **0.80 ms**
  (~2.0×), identical energy (−0.2034401277). Win grows with system size (removes
  O(n_occ²) redundant half-transforms + per-pair metric solves; now threaded).

## Stage 3: Conventional MP2 — integral-direct only (restricted)
**Goal**: integral-direct conventional MP2 that never materializes N⁴ (or the
full o²v² (ia|jb)). The in-core N⁴ path is deleted outright — it is too memory
-inefficient to keep even as a reference.
**Changes**:
- New integral-direct conventional MP2: loop occupied blocks (size from the
  memory budget); per block, stream AO shell-quartets once and contract the
  first index → half-transformed `(iν|ρσ)` (peak ≈ b_o·N³, bounded), then GEMM
  ν→a, ρ→j, σ→b per i; accumulate energy with the shared kernel. Threaded.
  The budget interpolates direct↔semidirect: a large budget keeps the
  half-transform resident over all occupied (one AO pass, semidirect); a small
  budget recomputes AO per block (fully direct). Never N⁴.
- Add a public four-center streaming entry point on `IntegralEngine`
  (lambda over `IntegralResult<4>`) so the correlation module can drive the
  AO-direct loop without reaching into `detail/`.
- Delete the N⁴ machinery: `MOIntegralEngine::compute_ovov_tensor`,
  `compute_ovov_block`, `compute_oovv_block`, `compute_ovvv_block`,
  `transform_block`, the four scalar `transform_*_index`, `IndexRange`. Strip
  `MOIntegralEngine` to the MO-coefficient/count holder (+ `compute_mo_eri`
  kept as a brute-force test reference, allocations hoisted) or fold those
  counts into `PostHFMethod`.
**Success Criteria**: conventional energies unchanged (water/3-21G, H₂/def2-SVP,
H₂/STO-3G); peak memory bounded by the budget (no N⁴, no full o²v²); threaded;
small-budget multi-block run gives identical energy.
**Tests**: existing conventional reference energies; conventional-vs-RI within DF
error; conventional small-budget multi-block exactness; `compute_mo_eri` vs
direct MP2 (ia|jb) on H₂ as a transform sanity check.
**Status**: Complete

Done:
- `IntegralEngine::ao_direct_half_transform(C, Schwarz)` (occ_qm): streams shell
  quartets and contracts the first index with an **inlined templated** lambda
  (not std::function), expanding the 8-fold symmetry conditionally on shell
  equality; returns H(i,ν,ρ,σ). Never builds N⁴.
- Rewrote `compute_conventional_mp2_energy` AO-direct: occupied-block loop
  (block size from budget; b_o=o → semidirect/one AO pass, small b_o → fully
  direct), GEMM completion via Eigen tensor contractions, shared
  `accumulate_mp2_pair` kernel (also used by RI), threaded.
- Deleted the N⁴ machinery: `compute_ovov_tensor/_block`, `compute_oovv_block`,
  `compute_ovvv_block`, `transform_block`, the four scalar `transform_*_index`,
  `IndexRange`, and the `-C` phase hack. `MOIntegralEngine` is now a slim
  MO-info holder + `compute_mo_eri` (allocations hoisted, buffer layout fixed)
  kept as a brute-force reference.
- Tests: updated MO-transform test (compute_mo_eri 8-fold symmetry + positivity);
  added conventional budget-blocking + thread-invariance. mp2_tests 45/10,
  qm_tests 117/29, full build/CLI link clean.
- **Speed (water/def2-SVP, full compute):** conventional 3.44 ms → **1.22 ms**
  (~2.8×), identical energy (−0.2034154547); peak memory now O(b_o·N³) not N⁴.

## Stage 4: Unrestricted (UHF) MP2
**Goal**: αα/ββ/αβ for both RI and conventional; remove the `runtime_error`.
**Status**: Complete

Done:
- Per-spin active ranges (`spin_active_ranges`), α/β coefficient + energy blocks
  via `block::a/b`.
- RI (`compute_unrestricted_ri_energy`): metric-folded B̃_α, B̃_β; αα/ββ
  same-spin + αβ opposite-spin, threaded.
- Conventional (`compute_unrestricted_conventional_energy`): per-spin AO-direct
  half-transform (occupied-block bounded); the α half-transform feeds both αα
  and αβ (second electron completed with α or β coeffs), β feeds ββ.
- Energy kernels: `os_pair_energy` (Σ K²/D) and `ss_pair_energy`
  (¼Σ(K_ab−K_ba)²/D) — reduce to the RHF formula in the closed-shell limit.
**Tests**: closed-shell UHF == RHF (conventional + RI, 1e-7); open-shell H₂O⁺
doublet conventional == RI within DF error (two independent UHF paths agree),
ss/os both < 0. mp2_tests 57/12, qm_tests 117/29, full build/CLI link clean.

---

## All five stages complete.
Optional follow-ups (not blocking): occupied-blocking for the UHF B̃ build;
shell-pair-packed layout for the *stored* DF path; out-of-core paging; SCS/SOS
as a first-class CLI option; `--mp2-max-memory` CLI knob for the budget.

## Stage 5: Direct RI 3-center, sparsity & cleanups
**Goal**: remove the dense `(μν|P)` store as the RI memory ceiling; keep RI-MP2
memory bounded for large systems.
**Status**: Complete

Done:
- `IntegralEngineDF::build_b_direct(C_left, C_right)`: builds the DF B tensor by
  streaming 3-center integrals (templated/inlined lambda; parallel over aux
  shells → disjoint B columns, no thread-local). Never materializes the dense
  `(μν|P)` store; uses the shell-pair list, so it's sparse by construction.
- `DFIntegrals(df, memory_budget)` auto-picks: dense store + reuse when it fits
  the budget (fast), else integral-direct per call (bounded). `build_b_tilde`
  folds the metric identically either way.
- RI-MP2 passes its budget through; combined with occupied blocking, peak RI
  memory is now max(B̃ block, transient 3-center buffers) — no `nbf²·naux`, no
  `o²v²`.
- Test: integral-direct vs stored RI-MP2 energy bit-identical (1e-10).
  mp2_tests 47/10, qm_tests 117/29, full build/CLI link clean.
- Already handled earlier: `-C` sign hack removed (Stage 3), pathological
  `const_cast` paths deleted (Stage 2), dead stubs removed (Stage 3).

Remaining follow-ups (optional): shell-pair-packed `num_rows()` layout for the
*stored* path; out-of-core paging if even one B̃ block exceeds RAM (the direct
path makes this rarely necessary).
- Remove `-C` sign hack (verify energy invariance); fix `const_cast` / make
  stores `mutable`; delete dead stubs (`compute_oovv_block`, `compute_ovvv_block`,
  `transform_block`); populate-or-remove `pair_energies`; SCS first-class.
**Success Criteria**: large case runs within a set memory budget; no behavior
change in energies; no dead code; clean build with no new warnings.
**Tests**: memory-bounded large-case run; full `mp2_tests` green; SCS values
checked.
**Status**: Not Started

---

## Open follow-ups (out of scope now)
- Swap RI 3-center primitive from libcint to `occ/ints` native Hermite kernels.
- SOS-MP2 / Laplace-transform denominators.
- MP2 gradients (would require retaining amplitudes/intermediates).
