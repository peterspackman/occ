# Native GFN2-xTB in occ

Goal: replace the `tblite_wrapper` with an in-tree GFN2-xTB. Reference: Grimme's `xtb` (`~/git/xtb`), DOI 10.1021/acs.jctc.8b01176.

## Phase 1 — Foundations  ✅
Param loader, STO-NG basis, overlap diagonal validated.

## Phase 2 — H0 + repulsion + iso-Coulomb + charge-only SCC  ✅
Charge-only SCC for water within ~7 mHa of full GFN2.

## Phase 3 — CAMM multipoles + third-order  ✅
Full GFN2 (no dispersion) for water within 4 µHa of xtb.

## Phase 4abc — D4 dispersion + native backend + SCC-coupled D4  ✅
- 4a: post-SCF D4 with EEQ charges
- 4c: SCC-coupled D4 (per-iteration `weight_references` with Mulliken charges)
- 4b: `NativeCalculator` mirrors the `TbliteCalculator` API
- Refactor: `Gfn2Calculator` owns the basis/integrals/H0/etc.; SCC drivers reduced to thin wrappers

**Status**: Water full GFN2 = -5.07026 Ha (xtb: -5.07026, Δ < 1 µHa). Methane Δ ≈ 30 µHa.
Committed in df5757e87 + 5fafb5aa7.

## Phase 4d — Crystal / periodic support  (in progress)

### Done
- **4d.1** `LatticeImage`, `build_lattice_images`, `PeriodicSystem::from_crystal`
- **4d.2** `gfn_coordination_numbers_periodic`, `repulsion_energy_periodic` (CN + repulsion sum over translations; equal molecular at large cell to 1e-12)
- **4d.3** `periodic_overlap_blocks`, `periodic_h0_blocks`, `bloch_sum`, `bloch_sum_gamma` (per-T real-space S^T, H0^T blocks via two-cell merged AOBasis; Bloch sum at any k)
- **4d.4** `periodic_klopman_ohno_gamma` (Ewald-summed shell-resolved γ at Γ).
  Splits γ = (γ - 1/R) + 1/R: residual sum over real-space lattice (1/R³
  decay, fixed 60-Bohr default cutoff); Coulomb tail Ewald-summed (real erfc,
  reciprocal G-sum, background, self). API in `include/occ/xtb/periodic_gamma.h`.
  Validated: α-invariance of γ matrix to 1e-9, energy α-invariance for neutral
  density to 1e-10, 2×1×1 supercell consistency for charged ionic pair to 1e-7,
  large-cell limit reduces to molecular γ (modulo Madelung shift) to 1e-4.
- **4d.5–4d.7** Γ-only periodic SCC + k-point sampling + crystal-driver
  `NativeCalculator(Crystal)`. SCC converges on real molecular crystals
  (BENZEN, ACENAP03, ACSALA07, ANTCEN14, BPHENO10, CITRAC10) with
  multipoles on, no catastrophic divergence.
- **4d.8 — periodic AES Ewald + Bra/Ket atom-centered AO multipoles + DIIS
  on (qsh; dipm; qpat).** `build_multipole_ewald_tensors` (real + reciprocal
  Ewald split for sd/dd/sq pair tensors), `build_periodic_multipole_ao`
  (per-T merged-basis dipole/quadrupole AO with origin shifts to row/col
  atom), `compute_camm_moments_periodic` (Bra-only Mulliken partition to
  match tblite/dftbplus convention), `apply_anisotropic_h1_periodic` (H1
  with side-specific Ket/Bra integrals).

### AES sign-convention rework (resolved; see commits 397de5a99 + 10bb0630c)

Previous OCC AES code carried OCC's own sign convention for `t.sd[α](i,j)`
and the gauge-corrected molecular `anisotropic_potentials` formula, which
were internally consistent at the molecular limit but did not line up with
tblite's `coulomb/multipole.f90` term-by-term. On crystals the convention
mismatch produced catastrophic divergence (initial implementation) and
later, after the convention drift was cancelled, a small but real ~1 mHa
gap on water vs tblite.

Root causes (all now fixed):
1. **Q_bra/Q_ket were full Cartesian.** tblite makes the AO quadrupole
   integrals traceless (1.5·Q − 0.5·tr·δ) at the integral level; we did
   the trace removal post-CAMM on `qpat` only. Mathematically equivalent
   for the Mulliken partition itself, but the H1 contribution
   `0.5·Q_bra·vq` is **not** invariant under post-CAMM trace removal
   when the trace of Cartesian Q is non-zero. Fix: apply
   `apply_traceless_quadrupole_transform` to Q_ket/Q_bra in both
   builders; drop the post-CAMM transform from `compute_camm_moments_periodic`.
2. **CT (kernel) potential sign** was `vd -= 2·dkernel·dipm` (matching an
   older OCC apply convention). tblite uses `+= 2·dkernel·dipm` paired
   with `H1 -= 0.5·D·v`. Fix: flip the CT sign and the apply sign
   together so the H1 contribution from the on-site polariz kernel
   matches tblite exactly.
3. **Apply sign** `H1 += eh1` (was OCC convention) → `H1 -= eh1`
   (matching tblite's `add_vmp_to_h1` + `add_vao_to_h1`). Done together
   with (2) so the net H1 contribution is unchanged, but each individual
   term now has tblite's sign.
4. **Molecular `anisotropic_energy` e01 sign** was the opposite of tblite's
   e_qd_v1 because OCC's pair loop uses `rij = R_j - R_i` while tblite's
   formula uses `(R_i - R_j)`. Fix: flip the sign of the two ed terms
   (and the same in `aes_pair_energy` for the periodic legacy path).

Validation:
- `AES potential: tblite reference for water_mol` (xtb_native_tests.cpp)
  feeds tblite's converged charges/dipoles/quadrupoles into OCC's
  `anisotropic_potentials_ewald` + `anisotropic_energy_ewald` and
  verifies e_aes and polariz match tblite to <0.05 mHa.
- Water molecule total: −5.07036943 Eh vs tblite −5.07036967 Eh
  (Δ = 0.3 µHa). Benzene molecule: 3 µHa.
- All 53 xtb_native_tests pass.

### Open issue — periodic crystal energies

Molecular energies match tblite to single-µHa. Crystals do not:

| Crystal   | Atoms | OCC total       | tblite total    | diff (mHa) | Δ/atom (mHa) |
|-----------|------:|-----------------|-----------------|-----------:|-------------:|
| BENZEN    |    48 | −63.59235       | −63.57247       | −19.9      | −0.41        |
| ACENAP03  |    88 | −123.31840      | −123.27769      | −40.7      | −0.46        |
| ACSALA07  |    84 | −158.66909      | −158.62756      | −41.5      | −0.49        |
| ANTCEN14  |    48 | −70.05947       | −70.03133       | −28.1      | −0.59        |
| BPHENO10  |    96 | −147.98072      | −147.94322      | −37.5      | −0.39        |
| CITRAC10  |    84 | −181.69979      | −181.61219      | −87.6      | −1.04        |

OCC is consistently ~0.4–1 mHa/atom *more* bound than tblite's reported
total. **Some of this is almost certainly tblite's broken-looking
periodic D4** — tblite reports `dispersion energy = +0.021 to +0.048 Eh`
(positive) for these crystals while a proper lattice-summed D4 should
be negative O(−0.1 Eh). On molecular benzene tblite gives 3e-11 Eh
correctly; the periodic D4 path looks suspect.

To split the discrepancy:
- OCC `--no-multipoles` BENZEN: −63.6486 Eh → multipole correction
  contributes +56 mHa (destabilising; intermolecular dipoles ↔ charges
  cancel some attractive E_iso).
- tblite has no equivalent flag, so we can't directly compare
  multipole-off to multipole-off.

### Plan to close the periodic gap (pending)

1. **Verify or replace tblite's reported "dispersion".** Re-run a couple
   of crystals against dftb+ (which is the periodic reference dftbplus
   was written for) or against xtb's own periodic D4 to determine the
   correct lattice-summed value. If the gap shrinks once tblite's
   misleading dispersion is removed from the comparison, the AES path
   is fine and only D4 needs reworking.
2. **Wigner–Seitz vs raw Ewald gamma.** tblite's `get_amat_3d` uses
   Wigner–Seitz cell averaging on top of the Ewald split; OCC uses
   straight Ewald (`periodic_klopman_ohno_gamma`). For neutral cells
   both should give the same total energy in the limit but they can
   differ at finite cell — check if WSC averaging closes a few mHa
   on the harder cases (CITRAC10).
3. **Periodic CN sensitivity.** Verify `gfn_coordination_numbers_periodic`
   matches tblite's get_coordination_number for the same cell at sub-µ
   accuracy. Small CN errors propagate through H0 → ε → P.
4. **End-to-end vs dftb+.** Build dftb+ locally (or use the conda
   package), feed it the same gen file, and compare. dftb+ implements
   GFN2-xTB via a different code path so it's a stronger reference for
   the periodic case than a possibly-broken tblite.

### Pending — original 4d roadmap items (mostly subsumed by 4d.5–4d.8)

**4d.5 — Γ-only periodic SCC** (~150 lines plumbing)
Add `Gfn2Calculator(Crystal)` constructor that:
- Stores `PeriodicSystem`
- Computes `lattice_images` for the SCC cutoff
- Builds periodic CN (4d.2), periodic γ (4d.4), per-T S/H0 blocks (4d.3)
Add `single_point_periodic(opts)` method:
- Bloch sum at Γ → real S, H0
- SCC iteration: same as molecular but with periodic V/H1 contributions (needs careful handling of multipole AO ints — may also need per-T blocks)
- Repulsion + dispersion already periodic
Validates by setting cell large enough that result equals molecular.

**4d.6 — k-point sampling + complex eigensolver** (~250 lines)
- `MonkhorstPackGrid(n1, n2, n3, sys.reciprocal())` → vector of (k, weight)
- Use crystal symmetry to reduce to IBZ (optional — full grid works as start)
- Per k-point: `S(k) = bloch_sum(S_per_T, k)`, `H(k) = bloch_sum(H_per_T, k)`
- Solve `H(k) C(k) = ε(k) S(k) C(k)` with `Eigen::GeneralizedSelfAdjointEigenSolver<CMat>` (need to check Eigen actually supports the complex generalized variant — may need to symmetrize via S^(-1/2) trick)
- Aufbau across all k with weights → density matrix and band-energy sum
- Fermi smearing for metals (reuse existing `qm::orbital_smearing` if it accepts complex)
- Mulliken populations summed across k (each k contributes a complex matrix; the diagonal is real)

**4d.7 — `NativeCalculator(Crystal)` driver + validation** (~150 lines)
- Crystal constructor on NativeCalculator routes to Gfn2Calculator's periodic path
- Pick a small molecular crystal (e.g. urea, ice Ih primitive cell) for validation
- Validation strategy: needs working tblite reference (Python `tblite` package or local libtblite build). Currently broken on this machine — would need to fix before validating end-to-end.
- Acceptance: total energy within ~1 mHa per atom of tblite, charges/orbital energies qualitatively correct.

### Open design questions for 4d
- Should the Γ-point SCC be a fast-path (real-only) or always go through the complex eigensolve at k=0? Real-only is ~2× faster but needs a separate code path.
- Are AO-resolved multipole shifts (CAMM H1) k-dependent? In principle yes — apply_anisotropic_h1 needs Bloch-summed dipole/quadrupole AO matrices at each k. Easy extension once 4d.3 is generalized.
- For metals, Fermi smearing across k requires entropy-aware free energy. Most molecular crystals are insulating, so this can be deferred.
- Crystal-symmetry-reduced k-mesh: optional optimization, not needed for correctness.

### Estimated effort
~800–1000 new lines, 3–5 days of focused work, a working tblite reference for validation.

## Phase 6 — Native dispersion (in progress)

Replace the cpp-d4 dependency with an in-tree DFT-D4 (and D3) implementation
under `src/disp/`. Reference data lives in `share/dftd4/`.

### Done
- **6a** Native D4 for GFN2-xTB: `occ::disp::Dispersion` reads the GFN2-xTB
  reference data extracted from xtb's `param_ref.fh`. Energy matches xtb to
  **0.28 µHa on water, 94 µHa on rubrene** (after multi-Gaussian ref weights
  and ATM sign fixes).
- **6b** Hirshfeld refq variant for DFT-D4 (`RefqMode::DFT`). Refq + refh
  tables come from cpp-d4's `refq_eeq` / `refsq` (the modern dftd4 fortran
  port's data). Per-functional damping presets in
  `share/dftd4/functionals.json` (152 functionals, e.g. pbe, b3lyp, wb97x,
  blyp, b97-3c). `set_charges_eeq()` populates atomic charges from EEQ.
  Existing tests for water+pbe/blyp, benzene+wb97x/b3lyp+1 all pass.
- **6c** Numerical gradient via `energy_and_gradient()` (5-point central
  diff). Slow but correct. Analytical version is **6f**.
- **6d** Migrated all D4 use sites to native (`qm/gradients.cpp`,
  `driver/single_point.cpp`, `driver/geometry_optimization.cpp`).
- **6e** Dropped cpp-d4 CMake dependency entirely.

- **6g** Native D3-BJ. `occ::disp::DispersionD3` with refdata in
  `share/dftd3/refdata.json` (94 elements, 5×5 ref C6 blocks per pair) and
  per-functional damping in `share/dftd3/functionals.json` (94 functionals,
  BJ variant). Validates against s-dftd3 to <1 µHa for water+pbe.
- **6f** Analytical D4 + D3 gradients (2-body BJ + ATM angular + ∂C6/∂CN
  chain). Plus EEQ derivative (`occ::core::charges::eeq_partial_charges_and_gradient`)
  → ∂q/∂R chain in D4 for full DFT-D4 forces. All gradient tests match FD
  to 1e-9 Ha/Bohr; full DFT-D4 force matches re-equilibrated-EEQ FD to
  1e-8 Ha/Bohr.

### Pending
(none — phase 6 complete)

## Phase 5 — Gradients + frequencies (in progress)

User requested **analytical** gradients (not numerical). Numerical kept only as a finite-difference oracle for validating the analytical implementations.

### Done
- **5a** `NativeCalculator::compute_gradient_numerical(step_bohr)` — central differences, used as the validation reference.
- **5b** Easy analytical gradients:
  - `repulsion_energy_and_gradient(atoms, params)` — closed-form pair derivative; matches FD to 1e-9 Ha/Bohr.
  - `gfn_coordination_numbers_with_gradient(atoms)` — CN values plus ∂CN_i/∂R; matches FD to 1e-8.
- **5c** `h0_scc_gradient` — analytical H0 + Pulay + V_q-via-S contribution.
  Z-matrix assembly: `Z = P·X − W − ½·P·(V+V)` where V is the converged shell
  shift potential V_q + V_3rd. Matches the full reconverged-SCC FD to <1 µHa.
  γ-gradient factor-of-2 fix (`klopman_ohno_gamma_energy_gradient`) included.
  SCC iteration switched to charge DIIS (water: 16→11 iters; rubrene: was
  diverging → 19 iters).

### Done (continued)

- **5d-partial — γ matrix gradient.** `klopman_ohno_gamma_energy_gradient`
  in `src/xtb/gamma.cpp` covers the explicit ∂(½ q^T γ q)/∂R term and is
  validated against FD to 1e-9. CAMM and anisotropic gradient pieces are
  still pending — see "Pending" below.

- **NativeCalculator::compute_gradient_analytical()** assembles the
  analytical pieces into a 3×N gradient. Runs charge-only SCC (multipoles
  off) + native D4 dispersion. Returns the energy that matches the
  gradient (charge-only SCC + repulsion + dispersion) so opt uses a
  consistent (E, ∂E) pair. Self-consistency vs FD: ≤ 5×10⁻⁵ Ha/Bohr; gap
  is the missing CPSCF response of D4 through the SCC charges (xtb has the
  same gap and accepts it as part of the SCC-coupled-D4 convention).

- **GFN2 wired into `optimization_step_driver`.** `MethodKind::GFN2`
  branch in `src/driver/geometry_optimization.cpp` calls
  NativeCalculator's analytical gradient and converts to Hartree/Å.
  `occ scf --driver opt water.xyz gfn2` runs end-to-end and converges in
  4 steps (water from a slightly displaced geometry).

### Pending — Phase 5

**5d-rest — CAMM + anisotropic multipole gradients** (~700 lines)
Needed for full multipoles-on analytical gradient (currently the analytical
path uses charge-only SCC for consistency, ~1 mHa energy gap vs full GFN2
on water). Components:
- ∂(CAMM dipole)/∂R: needs `IntegralEngine::one_electron_operator_grad(Op::dipole)`
  returning ∂dipole_int/∂R, plus ∂S/∂R chain.
- ∂(CAMM quadrupole)/∂R: same with `Op::quadrupole`.
- ∂(aniso ES + polariz)/∂R: chain through ∂(gab3, gab5)/∂R → ∂(radcn)/∂R
  → ∂CN/∂R, plus the explicit ∂(dipole·dipole interaction)/∂R kernels.

**5e — Hessian + frequencies** (~200 lines)
Plug NativeCalculator into `HessianEvaluator<Proc>` template (finite-diff
of analytical gradient, 0.005 Bohr step + acoustic sum rule). Then
mass-weighted diagonalization → frequencies via `core::VibrationalModes`.
Wire into `occ freq -m gfn2`.

### Known limitations / follow-ups

- **CPSCF for D4 through Mulliken charges**: The analytical gradient skips
  the ∂E_disp/∂q · ∂q_SCC/∂R chain — for SCC-coupled D4 this term needs
  the coupled-perturbed SCF response, same gap as xtb. ~5–10 µHa/Bohr on
  water. Negligible at typical opt tolerances but worth doing if we add
  CPSCF for other pieces.

- **xyz-file unit detection**: OCC reads .xyz strictly as Å, while xtb
  auto-detects Bohr when values are large (e.g. urea was distributed in
  Bohr; OCC's opt then fails with `trust radius too small` because the
  geometry is unphysical when interpreted as Å). Either add Bohr detection
  to `occ::io::xyz` (e.g. by atomic-distance heuristic) or document that
  inputs must be in Å.

- **Opt slowness**: A full opt step does (charge-only SCC) × (~10 iters) +
  one analytical gradient. Each SCC iteration rebuilds H + diagonalises;
  the integral matrices are cached. The dominant per-step cost is SCC
  iteration count × O(nbf³) eigensolve. For a tight loop over many opt
  steps the obvious wins are: (a) reuse the previous step's qsh as the
  initial guess (currently we restart from zero charges), (b) cache the
  D4 reference α-tables across calls (currently rebuilt every gradient),
  (c) avoid reconstructing the integral engine in `Gfn2Calculator::
  update_positions` — recompute only the integral matrices.

---
Out of scope for v1: GFN1/GFN0 (parameter file is structured to allow GFN1 later), solvation.
