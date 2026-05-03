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

### Pending — concrete plan for next session (Phase 4d)

**4d.4 — Periodic shell-resolved γ via Ewald** (~300 lines, hardest piece)
The Klopman-Ohno γ_ij(R) decays as 1/R for large R (when the global α_ewald → ∞ in the formula `(R^α + 1/η^α)^(-1/α)`), so a direct lattice sum is conditionally convergent. Use Ewald splitting:

    γ_ij^periodic(r_AB) =
        Σ_T [γ_ij(|r_AB + T|) - erf(α_ewald · |r_AB + T|) / |r_AB + T|]   (real space, short range)
      + (4π/V) Σ_{G≠0} [exp(-G²/(4 α_ewald²)) / G²] cos(G · r_AB)         (reciprocal space)
      + self term                                                          (G=0 + on-site)

- Real-space loop: only includes T-images such that |r_AB + T| < cutoff (typical ~25 Bohr for 1e-8 convergence).
- Reciprocal-space loop: G = h·b1 + k·b2 + l·b3, with G_max determined by α_ewald and tolerance (typical ~50 G-vectors per direction).
- On-site (i=j, T=0) gets only the η_ii diagonal (Klopman-Ohno self-energy), no Ewald subtraction.
- Choose α_ewald to balance real and reciprocal work — standard formula: α = √π · (V·N²)^(1/6) / V^(1/3) where N is atom count.
- Adapt `src/mults/ewald_sum.cpp` for the structure (real loop, G loop, self term). Note: that code uses charges/dipoles, not shell-resolved γ — adaptation is non-trivial.

Output: a function `periodic_gamma(sys, shells, params, translations, ewald_params)` → `Mat` (n_shells × n_shells) that gives the full periodic γ matrix at the Γ point. For k≠0, the Bloch phase enters the real-space term (and the reciprocal sum gets `cos(G·r_AB) → exp(i G·r_AB)` style). For an SCC at general k, this becomes `Mat γ(k)` — but γ for the **electrostatic energy** is k-independent (because charges live in the cell), only the **AO matrices** (S, H) become k-dependent. So we just need the Γ-point γ matrix.

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

### Pending
- **6f** Analytical D4 + D3 gradients (2-body BJ derivative + ATM angular
  derivative + ∂C6/∂CN chain rule). Optional EEQ chain rule `∂q_EEQ/∂R` for
  full DFT-D4 forces (small effect at typical geometries; numerical FD is
  acceptable meanwhile).

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

### Pending — concrete plan for next session (Phase 5)

The SCF gradient backbone. For an SCC at convergence:

    dE/dR = Σ_μν P_μν · (dH0_μν/dR)        (one-electron, density-weighted)
          - Σ_μν W_μν · (dS_μν/dR)         (Pulay term)
          + ½ q^T (dγ/dR) q                (electrostatic explicit deriv — Phase 5d)
          + d(third-order)/dR              (zero — Γ are geometry-independent)
          + d(aniso multipole)/dR          (Phase 5d)
          + d(repulsion)/dR + d(disp)/dR   (Phase 5b — done)

W = mo.energy_weighted_density_matrix() is already in occ::qm::MolecularOrbitals.

For dH0_μν/dR specifically, three contributions chain in:
  (a) Direct ∂H0/∂R via ∂S/∂R (libcint provides via `IntegralEngine::one_electron_operator_grad(Op::overlap)` → `MatTriple`).
  (b) ∂H0/∂R via the distance polynomial Π(R_AB) — analytic, depends on shellPoly coefficients and atomicRad.
  (c) ∂H0/∂R via ∂(self_energy_with_CN_shift)/∂R = -kCN · ∂CN/∂R  (use 5b's `dcn`).

Files to create/modify:
  - `include/occ/xtb/h0_gradient.h` / `src/xtb/h0_gradient.cpp`
  - Test: drive a numerical-vs-analytical comparison on water (3-atom case) using compute_gradient_numerical as oracle.

**5d — γ matrix + CAMM + anisotropic gradients** (~700 lines, 2-3 days)
- ∂γ_ij/∂R for the Klopman-Ohno function (closed form): `γ = (R^α + 1/η^α)^(-1/α)`; chain-rule with α = globals.alphaj.
- ∂(CAMM dipole)/∂R: needs `IntegralEngine::one_electron_operator_grad(Op::dipole)` returning ∂dipole_int/∂R, plus ∂S/∂R chain.
- ∂(CAMM quadrupole)/∂R: same with `Op::quadrupole`.
- ∂(aniso ES)/∂R: chain through ∂(gab3, gab5)/∂R → ∂(radcn)/∂R → ∂CN/∂R, plus the explicit ∂(dipole·dipole interaction)/∂R kernels.

**5e — Hessian + frequencies** (~200 lines, 1 day)
Once 5b–5d are wrapped: plug NativeCalculator into `HessianEvaluator<Proc>` template (finite-diff of analytical gradient, 0.005 Bohr step + acoustic sum rule). Then mass-weighted diagonalization → frequencies via `core::VibrationalModes`. Wire into `occ freq -m gfn2`.

Once 5c is done, also:
- Wire GFN2 case into `optimization_step_driver` so `occ opt -m gfn2 input.xyz` works.
- Convert returned Mat3N from Ha/Bohr → Ha/Angstrom for `BernyOptimizer::update`.

### Estimated effort
~1500-2000 new lines, 5-7 days of focused work. Validation at each phase via finite-difference of total energy.

---
Out of scope for v1: GFN1/GFN0 (parameter file is structured to allow GFN1 later), solvation.
