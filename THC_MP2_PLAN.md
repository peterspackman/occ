# THC-MP2 Implementation Plan

Add tensor-hypercontraction MP2 (LS-THC-MP2) alongside the existing DF-MP2
(`occ_correlation`) and THC-CCSD(T) (`occ_cc`). The scaling win comes from the
**Laplace transform** of the energy denominator, which decouples the (i,a) and
(j,b) sums so the THC factors collapse the orbital sums into small
`n_isdf x n_isdf` Gram contractions per quadrature point.

Scope (agreed with user): **full MP2** (opposite- *and* same-spin, so plain
MP2 / SCS / SOS all work) and **restricted + unrestricted**.

---

## STATUS: COMPLETE (2026-06-02)

Built as exploratory infrastructure toward a possible PNO-CCSD(T)+THC effort
(not as a production MP2 -- see the verdict below).

**Delivered**
- `laplace.{h,cpp}` -- least-squares Laplace quadrature for 1/x (fixed log nodes
  + QR weights).
- `thc_mp2.{h,cpp}` (`cc::thc_mp2`) -- restricted + unrestricted LS-THC-MP2,
  full OS+SS, with `opposite_spin_only` SOS fast path. Cubic O(P^3) Coulomb +
  O(o^2 v^2 P) orbital-form exchange (parallel over occ-pairs).
- `thc.{h,cpp}`: factored `thc_select_collocation` + shared `reg_inverse`/
  `solve_core`; added `fit_core_ov` (occ-virt-restricted core fit). `fit_core`
  and the CCSD path are unchanged.
- Driver/CLI: `--mp2-backend thc`, `--mp2-thc-c`, `--mp2-thc-method`,
  `--mp2-laplace-points`; SOS auto-enables the OS-only path.
- `share/basis/cc-pvdz-rifit.json` (from BSE).
- Tests: `[laplace]`, `[thc][mp2]`, `[thc][mp2][uhf]` (cc_tests). Suites green:
  cc_tests 135, mp2_tests 66, qm_tests 117.

**Verdict (measured, benzene cc-pVTZ, 1 core unless noted)**
- THC-MP2 matches DF-MP2 to <0.5 mHa after the ov-fit (was 4-12 mHa).
- But THC-MP2 ~12.7s vs DF-MP2 ~3.3s and **cannot beat DF-MP2** at these sizes:
  the THC factorization (ISDF selection ~7.9s, single-threaded) is unamortized
  overhead, and MP2 only needs the cheap `ovov` block where DF already wins.
  THC's niche is large-N / SOS, and its real payoff is THC-CCSD(T) (where the
  factorization is amortized over the solve + enables the O(v^4) ladder).
- Next high-value work (deferred): parallelize the ISDF point selection in
  thc.cpp -- benefits THC-CCSD(T) directly. Extend the ov-fit to the
  unrestricted path. Reusable PNO building blocks: per-pair K_ij kernel,
  Laplace quadrature, ov-space THC factorization.

## Key facts / constraints
- `occ_cc` depends on `occ_correlation` (one-directional). The `MP2` class lives
  in `occ_correlation`, so it cannot call THC. THC-MP2 therefore lives in
  `occ_cc` as a free function `occ::qm::cc::thc_mp2(...)`, mirroring `thc_eris`.
- Reuse `DFIntegrals::build_b_tilde` + `build_thc_from_B` / `build_uthc`, and the
  `freeze_core` / active-MO slicing patterns already in `integrals.cpp` /
  `uintegrals.cpp` (`extract`).
- Energy split (matches `accumulate_mp2_pair` in mp2.cpp). Per Laplace point the
  scaled collocation is `Xo~_iP = Xo_iP e^{+e_i t/2}`, `Xv~_aP = Xv_aP e^{-e_a t/2}`;
  Gram `Go = Xo~^T Xo~`, `Gv = Xv~^T Xv~` (both PxP); `B = Go o Gv` (Hadamard).
  Positive Laplace sums:
    `coul = sum_t w_t <B_t, V B_t V>_F`            (O(P^3)/pt)
    `exch = sum_t w_t Xkernel(V, Go_t, Gv_t)`       (O(P^4)/pt)
  with `Xkernel = sum_PQRS V_PQ V_RS Go_PR Go_QS Gv_PS Gv_QR`.
  - Restricted: `opposite_spin = -coul`, `same_spin = -coul + exch`.
  - Unrestricted: `same_spin = 1/2(exch_a - coul_a) + 1/2(exch_b - coul_b)`,
    `opposite_spin = -coul_ab` where `coul_ab = sum_t w_t <Ba_t, Vab Bb_t Vab^T>`,
    `Ba_t = Goa o Gva`, `Bb_t = Gob o Gvb` (cross core Vab, no exchange).
- Exchange kernel done as P GEMMs, O(P^2) memory (per interp index R:
  `F = V*diag(Gv[:,R])`, `G = Gv*diag(V[R,:])`, `H = F*Go`,
  `x += sum_P Go[P,R] * sum_S H[P,S] G[P,S]`).

## Stage 1: Laplace quadrature utility
**Goal**: `laplace_grid(xmin, xmax, n)` -> `{t, w}` with `sum_k w_k e^{-x t_k} ~ 1/x`
over `[xmin, xmax]`, plus Gauss-Legendre nodes (Golub-Welsch via Eigen).
Method: canonical `1/x' on [1,R]` (R=xmax/xmin) = `int_0^1 u^{x'-1} du`, GL on
[0,1]; `t_k = -ln(u_k)/xmin`, `w_k = omega_k/(u_k*xmin)`. All nodes/weights > 0.
**Files**: `include/occ/qm/cc/laplace.h`, `src/qm/cc/laplace.cpp` (+ CMake).
**Success**: max relative error vs `1/x` over a log grid of x in `[xmin,xmax]`
below tol for n~10-13 (< 1e-5 for R up to ~200).
**Tests**: `[laplace]` unit test in cc_tests.cpp.
**Status**: Complete

Done:
- `laplace.{h,cpp}` in occ_cc: `laplace_grid(xmin,xmax,n)` uses FIXED log-spaced
  nodes bracketing [1/xmax, 1/xmin] and solves a linear least-squares (QR) for the
  weights to fit `1 - x*sum_k w_k e^{-x t_k}` over a dense log grid of x. Robust,
  well-conditioned, monotone in n (the nonlinear-fit approach stalled ~6e-5
  regardless of n -- exponential-sum ill-conditioning). `laplace_max_rel_error`
  diagnostic. Realistic gap range (R~25) -> 2.4e-5 rel at n=13 (~5e-6 Ha, two
  orders below THC error). `[laplace]` test green (26 assertions).

## Stage 2: Restricted THC-MP2 opposite-spin (cubic)
**Goal**: `thc_mp2` restricted path computing `opposite_spin = -coul`.
**Files**: `include/occ/qm/cc/thc_mp2.h`, `src/qm/cc/thc_mp2.cpp` (+ CMake).
**Success**: OS energy matches DF-MP2 `results().opposite_spin_correlation`
within ~1e-3 for water/def2-svp.
**Tests**: `[thc][mp2]` OS-only check vs DF-MP2.
**Status**: Complete (folded into Stages 2+3 together)

## Stage 3: Restricted THC-MP2 same-spin (quartic exchange) -> full restricted
**Goal**: add exchange kernel; `same_spin = -coul + exch`. Full restricted MP2.
**Success**: total + SS + OS all match DF-MP2 within ~1e-3 (water, HF).
**Tests**: `[thc][mp2]` full restricted vs DF-MP2.
**Status**: Complete

Done (Stages 2+3):
- `thc_mp2.{h,cpp}` in occ_cc: `thc_mp2(basis, aux, mo, ThcMP2Options)` ->
  `ThcMP2Result{same_spin, opposite_spin, total, n_isdf, n_laplace, ...}`.
  Restricted path: freeze_core active space, DF B + `build_thc_from_B`, per
  Laplace point scale Xo/Xv rows, form Go/Gv Grams, `coulomb_term` (<B,VBV>) and
  `exchange_term` (P-GEMM O(P^4) complete-graph kernel). `opposite_spin=-coul`,
  `same_spin=-coul+exch`.
- Validated vs DF-MP2 (water/def2-svp, c_isdf=8, n_isdf=200): os Δ2.2e-4,
  ss Δ6.6e-6, total Δ2.2e-4; Laplace err 2.1e-5. `[thc][mp2]` green.

## Stage 4: Unrestricted THC-MP2
**Goal**: UHF path via `build_uthc`; assemble per spin channel.
**Success**: UHF closed-shell == restricted within tol; open-shell vs DF-UMP2 ~1e-3.
**Tests**: `[thc][mp2][uhf]`.
**Status**: Complete

Done:
- Unrestricted path: `build_uthc` (Xa,Xb,Vaa,Vbb,Vab); per Laplace point form
  αα/ββ Grams + cross Ba/Bb Hadamards; `same_spin = 1/2(exch_a-coul_a) +
  1/2(exch_b-coul_b)`, `opposite_spin = -coul_ab` with
  `coul_ab = <Ba, Vab Bb Vab^T>`. Laplace range spans all three channels.
- Validated vs DF-UMP2: closed-shell water == restricted within 1e-3; open-shell
  OH doublet total Δ6.5e-5. `[thc][mp2][uhf]` green. Full cc_tests 132/19 green.

## Stage 5: Driver + CLI wiring
**Goal**: `--mp2-backend {df|thc}` (+ `--mp2-thc-c`, `--mp2-thc-method`,
`--mp2-laplace-points`) in occ_scf.cpp; `MethodInput` fields in occ_input.h;
`run_mp2_method` branch in single_point.cpp calling `cc::thc_mp2` and logging
OS/SS/SCS like the existing MP2 path.
**Success**: `occ scf mol.xyz --method mp2 --ri-basis ... --mp2-backend thc` runs
and prints correlation energy consistent with DF-MP2; existing MP2 tests unaffected.
**Status**: Complete

Done:
- `MethodInput`: `mp2_backend{"auto"}`, `mp2_thc_c_isdf{6}`, `mp2_thc_method`,
  `mp2_laplace_points{14}`. CLI: `--mp2-backend`, `--mp2-thc-c`,
  `--mp2-thc-method`, `--mp2-laplace-points`.
- `run_mp2_method`: early THC branch (aux = --ri-basis or def2-universal-jkfit,
  chemical frozen core via `cc::num_frozen_core`), logs interp/Laplace points +
  error, same/opposite spin, SCS/SOS scaling, sets method "THC-MP2"/"SCS-THC-MP2"
  /"SOS-THC-MP2". `auto` keeps prior RI/conventional behaviour.
- CLI verified (water/def2-svp, fc=1): DF-MP2 -0.20359582 vs THC-MP2 -0.20369566
  (Δ1.0e-4); SCS-THC-MP2 runs. mp2_tests 66/13 + cc_tests 132/19 green.

---

## All five stages complete.

### Performance tuning (done 2026-06-02; benzene/cc-pVDZ)
- **Same-spin exchange rewrite O(P^4) -> O(o^2 v^2 P)**: replaced the
  complete-graph interpolation-point kernel with the orbital-intermediate form
  (`exchange_orbital`): per Laplace point `MvO_i = Mv .* Mo(i,:)`,
  `Z_i = MvO_i V`, `K_ij = Z_i MvO_j^T = (ia|jb)`, accumulate
  `sum_ab K(a,b)K(b,a)`. Parallel over occupied pairs (i>=j). Benzene c=6 THC-MP2
  **35s -> 3.7s single / 2.2s threaded**. Same-spin still matches DF-MP2 to 6e-5.
- **SOS fast path** (`ThcMP2Options::opposite_spin_only`): `--mp2-spin-scaling
  sos` skips the exchange entirely -> pure O(P^3) cubic Coulomb. The genuine
  large-system THC win. (Full THC-MP2 same-spin is O(N^5) and cannot beat
  DF-MP2; THC's niche is SOS/large-N.)
- **RI-MP2 vs ORCA finding**: OCC RI-MP2 is competitive (~0.18s vs ORCA 0.365s
  module). The apparent slowness was (a) OCC defaulting to CARTESIAN d-functions
  (use `--spherical` to match ORCA's 5d and the -0.7831 energy), (b) oversized
  `def2-universal-jkfit` aux vs `cc-pvdz-rifit` (now in share/basis from BSE),
  (c) the 25s conventional SCF when `--df-basis` is not set.

### occ-virt-restricted core fit (done 2026-06-02)
- **`fit_core_ov`**: fit the THC core V over the occ-virt (ovov) block only --
  the integrals MP2 needs -- instead of all nmo^2 pairs. Metric S = (Xo^T Xo) o
  (Xv^T Xv); reference B_ov = build_b_tilde(C_occ, C_virt) (o*v x naux). Factored
  `thc_select_collocation` (selection half of build_thc) + shared `reg_inverse`/
  `solve_core` out of thc.cpp so MP2 reuses the AO point selection but its own fit.
- **~50x more accurate at fixed c** (it fits exactly what MP2 uses, no compromise
  across all pairs): water ~1e-7; benzene cc-pVDZ Δ8e-5 (was 4.2 mHa); cc-pVTZ
  Δ0.34 mHa (was 12.4 mHa) -- now matches DF-MP2. Also cheaper: B-build o*v vs
  nmo^2 (cc-pVTZ DF 1.06->0.30s, core fit 2.25->1.16s). benzene cc-pVDZ 2.6->1.9s,
  cc-pVTZ 16.4->12.7s.
- Accuracy is steep in c (benzene cc-pVDZ: c=4 -> 4.4 mHa, c=6 -> 0.08 mHa), so
  the point COUNT (~6*nbf) is set by ISDF interpolation, not the fit -- can't drop
  c. fit_core (all-pairs) unchanged -> CCSD path identical (tests green).

### Remaining bottleneck / follow-ups
- **ISDF point selection** (pivoted Cholesky, single-threaded: ~1.2s cc-pVDZ,
  ~7.9s cc-pVTZ) now dominates THC-MP2 *and* the THC-CCSD factorization. Shared;
  parallelizing it (`thc.cpp` `pivoted_cholesky_points` -- the W*wp GEMV and L
  update over npts) is the next win and benefits CCSD. [item #2]
- Unrestricted THC-MP2 still uses the all-pairs `build_uthc` fit; extend the ov
  fit to UHF (Vaa/Vbb over same-spin ov, Vab over cross ov) for the same gain.
- Could share one DF/THC factorization between THC-MP2 and a later THC-CCSD run.
