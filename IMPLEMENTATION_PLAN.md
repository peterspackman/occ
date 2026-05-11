# Native GFN2-xTB in occ

Goal: replace the `tblite_wrapper` with an in-tree GFN2-xTB. Reference:
Grimme's `xtb` (`~/git/xtb`), DOI 10.1021/acs.jctc.8b01176.

## Status at a glance

**Working today**

- **Molecular SCC** (water, methane, larger): full GFN2 incl. CAMM
  multipole AES, on-site polarisation, third-order, native D4 — matches
  xtb to single µHa.
- **Periodic SCC**: Γ-only and Monkhorst-Pack k-point sampling, with
  multipole AES, Ewald-summed γ, lattice-summed D4. Converges on real
  molecular crystals (BENZEN, ACENAP03, ACSALA07, ANTCEN14, BPHENO10,
  CITRAC10).
- **Analytical gradient** (charge-only SCC) for molecular geometry opt;
  validated to 5×10⁻⁵ Ha/Bohr against FD.
- **Native D4 + D3-BJ dispersion** with analytical gradients.
- **Threading**: TBB-parallel per-T builders (overlap, H0, multipole AO),
  per-k Bloch sums + eigensolves, D4 2-body, Ewald γ. ~3–4× wall-clock
  speedup at 8 threads on real crystals.
- **Bindings**: `XtbCalculator` exposed via Python (nanobind) and
  JS / WASM (emscripten / embind). Top-level `occpy.XtbCalculator` and
  `Module.XtbCalculator` (factories `fromMolecule` / `fromCrystal` on JS).

**Open — high priority**

1. **Phase 5e — Hessian + frequencies** (~200 lines).
   FD of analytical gradient + acoustic sum rule, plug into the existing
   `HessianEvaluator<Proc>` template. Wire `occ freq -m gfn2`. Validate
   against `xtb --ohess` on water/methane. Prerequisite for elastic
   tensors / phonons.
2. **Periodic gradient + crystal opt**. Currently only molecular gradient
   is wired. Needs ∂E_periodic/∂R: lattice sums for repulsion gradient,
   periodic γ gradient (∂(½ qᵀ J q)/∂R with the Ewald split), periodic D4
   gradient + ∂CN_periodic/∂R chain. Once done, `occ scf --driver opt`
   works on crystals and FD-of-energy elastic tensors are trivial.
3. **Periodic crystal energies are ~0.4–0.6 mHa/atom too bound vs tblite.**
   See "Known issues" below for the leading suspects (WSC averaging is
   the top one).

**Open — medium priority**

4. **Phase 5d-rest — CAMM + anisotropic multipole gradients**.
   Closes the ~1 mHa energy gap between `single_point` (multipole-on) and
   `gradient` (charge-only SCC for consistency).

   Status: **steps 1+2 done as standalone validated routines**:
   - `anisotropic_pair_gradient` — closed-form ∂E_AES/∂R at frozen
     (q, μ, Q, R_co). Matches FD-of-`anisotropic_energy` to <1e-7 Ha/Bohr.
   - `anisotropic_pair_gradient_with_dcn` — extends step 1 with the
     CN chain through `mp_radii(CN)`. Matches FD to <1e-7 Ha/Bohr when
     mp_radii flows with CN(R).
   - `multipole_radii_with_gradient` — closed-form `dr/dCN` for the
     sigmoid in CN.

   Remaining work for full multipole-on `gradient()`:
   - **Step 3 — Z-matrix update with V_AES**. Promote V_shell's iso
     shift to per-AO so it can absorb the per-atom AES vs contribution.
     This makes the `Z·∂S/∂R` Pulay term capture the SCC density response
     to vs.
   - **Step 4 — AO multipole integral derivatives**. ∂D_ket/∂R,
     ∂D_bra/∂R, ∂Q_ket/∂R, ∂Q_bra/∂R for the explicit "centering changes
     with R" chain. Needed because with multipoles on, the Fock has
     terms `½ D_ket·vd + ½ D_bra·vd + ½ Σ_l Q_ket[l]·vq_l + ½ Σ_l Q_bra[l]·vq_l`
     whose ∂/∂R doesn't go through ∂S alone. Probably the largest piece.
   - **Step 5 — on-site polariz ∂/∂R**. Mostly falls out of step 4 via
     the (μ, Q) chain.

   Pilot integration (running multipole-on SCC + steps 1+2 inside
   `gradient()`, with vs folded into V_shell as a partial step 3) showed
   a ~14 mHa/Bohr gap to FD-of-energy on water — confirming step 4 is
   the dominant missing piece. Reverted; steps 1+2 stay as standalone
   functions ready to wire in once 3+4 land.

5. **Eigensolve threading**. Largest remaining serial cost on real
   crystals (~4 s of 9 s wall on QQQCIG11 at 8 threads). Direct `dsygv_`
   call against Apple Accelerate / OpenBLAS via a small wrapper.

**Open — low priority / nice-to-haves**

6. **CPSCF for D4 through Mulliken charges**. The analytical gradient skips
   ∂E_disp/∂q · ∂q_SCC/∂R for SCC-coupled D4. ~5–10 µHa/Bohr gap on water,
   matches xtb's own convention.
7. **xyz-file unit detection**. OCC reads .xyz strictly as Å; xtb
   auto-detects Bohr when values are large (e.g. urea distributed in Bohr).
   Either heuristic-detect or document Å-only.
8. **Opt warm-start**. Reuse previous step's qsh as initial guess (currently
   restarts from zero charges); cache D4 reference α-tables across calls.
9. **Open-shell GFN2** (`set_num_unpaired_electrons` is currently a stub
   that throws unless n=0). GFN1/GFN0 also out of scope for v1.
10. **Continuum solvent** — see Phase 7 below (in progress).

**Out of scope for v1**: GFN1/GFN0.

---

## Next-session sketches

Concrete plans for picking up open items cold. Each lists files to
touch, the exact chain rule, what to leverage, and an FD validation
pattern.

### Phase 5e — numerical Hessian + frequencies — STATUS: shipped (53e3ad5e0)

`XtbCalculator::compute_hessian_numerical(step)` + `compute_vibrational_modes(...)`
exist and bind to Python (`calc.hessian()`, `calc.vibrational_modes()`)
and JS (`calc.hessian()`, `calc.vibrationalModes()`). Caveat: built on
top of `gradient()` which runs charge-only SCC, so the Hessian is of
the charge-only-SCC energy. Water frequencies land at 1521/3538/3644 cm⁻¹
vs xtb's 1574/3578/3671 (~50 cm⁻¹ shift from missing multipole-on
gradient — closes once Phase 5d-rest steps 3-5 land).

If you want full-multipole frequencies *now* (without waiting for 5d-rest),
the cheap fix is **FD of `single_point_energy` directly** instead of
FD-of-`gradient`:
- New method `compute_hessian_numerical_from_energy(step)` doing 4N²
  central-difference single-point evaluations.
- Slow (O(N²) SCC calls vs O(N) gradient calls), but multipole-correct.
- Add a `from_energy=false` flag on `compute_vibrational_modes`.

### Step 4 — AO multipole integral derivatives (the big remaining piece)

This is what unblocks the multipole-on `gradient()`. The Fock under
multipoles has terms

```
F_μν -= ½ (D_ket(μ,ν)·vd_A_μ + D_bra(μ,ν)·vd_A_ν)
F_μν -= ½ Σ_l (Q_ket[l](μ,ν)·vq(l, A_μ) + Q_bra[l](μ,ν)·vq(l, A_ν))
```

so the gradient has Pulay-like contributions Σ μν P_μν · vd_A · ∂D_ket/∂R
that don't go through ∂S/∂R alone.

**Files to touch**:
- `include/occ/qm/integral_engine.h` — should already have
  `one_electron_operator_grad(Op::dipole)` returning
  ∂D(origin=0)/∂R. Verify; if missing, the dipole-derivative kernel
  needs to be wired into `IntegralEngine::compute_one_electron_grad`.
- `include/occ/xtb/periodic_integrals.h` — add new builder
  `build_molecular_multipole_ao_with_gradient(atoms, params)` returning
  the existing `PeriodicMultipoleAO` plus per-axis 3 × N "translation
  derivative" tensors. The centering chain is:
    D_ket(μ, ν) = D_origin0(μ, ν) − R_atom_of_μ · S(μ, ν)
    ∂D_ket(μ, ν) / ∂R_atom_of_μ = ∂D_origin0/∂R_atom_of_μ
                                  − S(μ, ν) (direct centering term)
                                  − R_atom_of_μ · ∂S/∂R_atom_of_μ
  Same shape for D_bra (col-side atom), Q_ket, Q_bra (with the
  ∂(R·R)/∂R = 2R chain on the quad outer product).
- `src/xtb/anisotropic.cpp` — extend `apply_anisotropic_h1_periodic`
  with a "build the H1 contribution as an AO matrix at one (μ, ν)
  pair" helper that the gradient assembly can iterate over.
- `src/xtb/h0_gradient.cpp` — the Z-matrix construction stays the
  same shape but now Z absorbs vs (per-atom) at the AO level rather
  than per-shell. Easiest path: change `V_shell` argument to
  `V_ao` (length nbf) so the caller can fold `vs(atom_of(μ))` in.

**Validation pattern** (independent of the SCC):
1. Pick frozen P (e.g., from converged charge-only SCC).
2. Pick frozen vs/vd/vq (any reasonable values).
3. Compute analytical Σ μν P_μν · vd_A · ∂D_ket(μ,ν)/∂R_iα.
4. FD: displace atom i by ±h, recompute D_ket, re-evaluate
   Σ μν P_μν · vd_A · D_ket(μ,ν), compare to (E+ − E−)/(2h).
5. Should match to <1e-7 Ha/Bohr.

Once step 4 passes, integrating it back into `gradient()` is mostly
re-running the pilot from `8bccec802` with the additional Pulay-like
∂D/∂R, ∂Q/∂R contributions added.

Estimated effort: ~400 lines of integral derivative + Z-matrix code +
~150 lines of tests, 1–2 focused sessions.

### Step 3 — Z-matrix update with V_AES — small follow-up to step 4

After step 4 lands, this is bookkeeping:
- Promote `V_shell` to `V_ao` (length nbf) in `h0_scc_gradient`
  signature. Each AO inherits its shell's iso V plus its atom's vs.
- The vd / vq pieces already covered by step 4's ∂D_ket, ∂Q_ket
  contributions (they're not S-coupled).

Estimated effort: ~50 lines, ½ session.

### Periodic gradient + crystal opt

Mostly a parallel of the molecular gradient with periodic kernel
sums. Independent of Phase 5d-rest — can be done in parallel.

**Files to touch / add**:
- `include/occ/xtb/periodic_repulsion.h` (new) — analytical periodic
  repulsion gradient: lattice sum of pair derivatives over
  `rep_images`. Mirrors `repulsion_energy_and_gradient` but with
  translation loop.
- `src/xtb/periodic_gamma.cpp` — extend `klopman_ohno_gamma_energy_gradient`
  to a periodic variant. The Ewald split makes this fiddly: residual
  γ - 1/R has analytical r-derivatives (already in the molecular
  routine), the Ewald 1/R tail needs its own real-space + reciprocal
  G-sum derivative. tblite's `coulomb/effective_3d.f90 get_gradient`
  is a usable reference.
- `src/xtb/h0_gradient.cpp` — the Pulay assembly works per-T if the
  caller hands it per-T S^T, P^(0,T), W^(0,T). Add a periodic variant
  `h0_scc_gradient_periodic(...)` that iterates translations and
  Bloch-sums correctly.
- `src/xtb/xtb_calculator.cpp` — periodic branch in `gradient()`
  that wires the above + the existing periodic D4 gradient
  (`D4Dispersion::energy_and_gradient_periodic` — already exists).

**Validation**: vacuum-padded crystal (e.g., water at 30 Bohr cubic)
should match the molecular gradient to ~1e-6 Ha/Bohr.

Estimated effort: ~600 lines, 2–3 sessions.

### Periodic energy gap (~0.4 mHa/atom vs tblite)

WSC averaging is the leading suspect (see "Known issues" below).
Concretely:
- `src/xtb/periodic_gamma.cpp::periodic_klopman_ohno_gamma` and
  `src/xtb/multipole_ewald.cpp::build_multipole_ewald_tensors`:
  in tblite, both functions average per-image contributions over
  Wigner–Seitz equidistant images of each pair vector with weight
  `1/nimg`. We don't.
- For low-symmetry molecular crystals `nimg = 1` for most pairs, but
  the cumulative effect across O(N²) pair-images is the suspected
  source of the 0.4 mHa/atom gap.
- Implementing requires a WSC neighbor enumerator: for each pair
  vector R_ij, walk the lattice, find all images at the minimum image
  distance (within tolerance), and weight by 1/nimg in both real and
  reciprocal Ewald sums.

Bisection plan:
1. Add a `wsc_average=true/false` toggle to both routines.
2. With `wsc_average=false` (current), confirm we still match
   tblite to the same ~0.4 mHa/atom on the 5 reference crystals.
3. With `wsc_average=true`, see if the gap closes. If it does, ship.
   If not, the gap is somewhere else (multipole AO cutoff conventions
   are the next suspect).

Estimated effort: ~300 lines + lots of careful debugging, 1–2
sessions.

### Phase 7 — Implicit solvation (CPCM-X + SMD)

End goal: per-surface-element solvation contributions exposed through
`occ cg` so the crystal-growth energy decomposition can attribute
electrostatic + CDS energy to each surface patch and each neighbour.

Decisions locked in for this work:
- **CPCM-X first**, SMD second. CPCM-X has a tblite reference; SMD's CDS
  is additive on top of an ES backbone we get for free from CPCM-X.
- **Classical COSMO solver** (reuse `occ::solvent::COSMO`), not ddCOSMO.
  Documented small (~µHa) discrepancy vs tblite's ddCOSMO.
- **Atom-resolved** solvation potential (matches tblite; simplifies the
  Fock-shift to a per-shell add inside the existing iso V).
- **Per-element data** lives on a `SolventSurface`-shaped struct exposed
  through `XtbResult` so the cg layer can consume it directly.

#### Phase 7A — Solvation plumbing (no physics) — STATUS: in progress

Goal: any `XtbSolvationModel` (initially a `NullSolvationModel`) plugs
into the SCC without breaking gas-phase numbers.

Touches:
- `include/occ/xtb/solvation_interface.h` (new) — abstract
  `XtbSolvationModel { initialize(positions, Z); update(atom_q);
  atom_potential() const; energy() const; name() const; }` plus a
  `NullSolvationModel`.
- `include/occ/xtb/gfn2_engine.h`,
  `src/xtb/gfn2_engine.cpp` — engine holds
  `std::shared_ptr<XtbSolvationModel>`. SCC calls `initialize()` at the
  top of `single_point()`, `update(atom_q_iter)` at the start of each
  iteration, folds `atom_potential()[atom_of_shell(s)]` into the
  per-shell `V`, and adds `energy()` to `scc_energy` in the breakdown.
- `include/occ/xtb/xtb_calculator.h`,
  `src/xtb/xtb_calculator.cpp` — `set_solvation_model(shared_ptr)`
  forwards to the engine. Public `set_solvent(name)` stays a stub
  (returns false) until built-in models exist (7B/7C).
- `tests/xtb_native_tests.cpp` — gate test: `NullSolvationModel`
  attached to water + methane must produce identical
  `single_point_energy()` and `charges()` to gas phase (<1e-10 Ha,
  <1e-12 e). Multipole-on path identical.

Gradient interaction: with `NullSolvationModel` the analytical gradient
is unchanged because `atom_potential()` and `energy()` are zero. Real
models will need a gradient hook in Phase 7B+; for now the engine warns
once if a non-null model is attached and `gradient()` is called.

#### Phase 7B — CPCM-X (tblite-parity electrostatics)

- `include/occ/xtb/cpcmx.h`, `src/xtb/cpcmx.cpp` (new) —
  `CpcmXSolvationModel : XtbSolvationModel`. Cavity via the existing
  `occ::solvent::surface` Lebedev builder, ASC via `occ::solvent::COSMO`,
  atom-resolved potential = σ contracted against atom→cavity Coulomb
  blocks.
- Validate water / methanol / formamide single-point against tblite
  CPCM/GFN2 (ε=78.4). Target <5e-5 Ha; document the residual as
  classical-vs-ddCOSMO.
- Analytical gradient via adjoint (`s · ∂A/∂R · σ` + nuclear-side
  ∂φ/∂R). FD-validate to <1e-7 Ha/Bohr.

#### Phase 7C — SMD on top of the CPCM-X backbone

- `include/occ/xtb/smd_xtb.h`, `src/xtb/smd_xtb.cpp` (new) —
  `SmdSolvationModel : XtbSolvationModel`. Reuses the ES solve from 7B
  with SMD radii and ε; CDS energy adds non-self-consistent atomic-σ +
  macroscopic-γ × surface-area terms (cavity rebuilt with SMD radii).
- Validate SMD/GFN2 hydration free energy on a 5-molecule benchmark
  against published numbers; qualitative match is sufficient.

#### Phase 7D — Per-element exposure

- `XtbResult` grows `std::optional<SolventSurface> solvent_surface;`
  carrying `{ positions, areas, atom_index, e_coulomb_per_element,
  e_cds_per_element }`. Mirrors `occ::cg::SMDSolventSurfaces` so cg
  consumers can swap backends without conditionals.

#### Phase 7E — cg integration

- `XTBCrystalGrowthCalculator` pulls the per-element surface from
  `XtbResult` directly (no DFT post-processing pass needed).
- Replace scalar accumulation in `SolventSurfacePartitioner` with a
  per-element-preserving variant; preserve the per-dimer scalars as a
  derived view for backward compatibility.
- Extend `cg_json.h` to dump per-element JSON:
  `{position, area, atom_index, neighbor_index, e_coulomb, e_cds,
  e_total}`.

### Eigensolve threading

Largest remaining serial cost on real crystals (~4 s of 9 s on
QQQCIG11 at 8 threads — see `git log` for the perf table).

**Files to add**:
- `src/xtb/lapack_eigensolve.cpp` (new) — small wrapper around
  Apple Accelerate / OpenBLAS `dsygv_`. Real generalized eigensolve
  for the Γ-only periodic SCC. About 80 lines.
- `src/xtb/lapack_zheevd.cpp` (or extend the same file) — complex
  variant `zhegv_` for k-point eigensolves.
- Replace `Eigen::GeneralizedSelfAdjointEigenSolver<Mat>` calls in
  `gfn2_engine.cpp` (molecular) and `gfn2_periodic_calculator.cpp`
  (periodic) with the LAPACK wrapper. Gate behind `USE_SYSTEM_BLAS`.

Validate: 5 reference crystals' totals must remain identical to
12 decimals. Speedup target: ~3–4× on the eigensolve phase.

Estimated effort: ~150 lines, 1 session.

---

## Phase history (archived)

### Phase 1 — Foundations ✅
Param loader (152-element JSON), STO-NG basis, overlap diagonal validated.

### Phase 2 — H0 + repulsion + iso-Coulomb + charge-only SCC ✅
Charge-only SCC for water within ~7 mHa of full GFN2.

### Phase 3 — CAMM multipoles + third-order ✅
Full GFN2 (no dispersion) for water within 4 µHa of xtb.

### Phase 4abc — D4 dispersion + native backend + SCC-coupled D4 ✅
- 4a: post-SCF D4 with EEQ charges
- 4b: `XtbCalculator` (was `NativeCalculator`) mirrors `TbliteCalculator`
- 4c: SCC-coupled D4 (per-iteration `weight_references` with Mulliken
  charges)
- Refactor: `Gfn2Engine` (was `Gfn2Calculator`) owns the basis / integrals
  / H0 / γ; SCC drivers reduced to thin wrappers

Water full GFN2 = -5.07026 Ha (xtb: -5.07026, Δ < 1 µHa).
Methane Δ ≈ 30 µHa.

### Phase 4d — Crystal / periodic support ✅

- **4d.1** `LatticeImage`, `build_lattice_images`,
  `PeriodicSystem::from_crystal`.
- **4d.2** `gfn_coordination_numbers_periodic`,
  `repulsion_energy_periodic` (CN + repulsion sum over translations;
  equal molecular at large cell to 1e-12).
- **4d.3** `periodic_overlap_blocks`, `periodic_h0_blocks`, `bloch_sum`,
  `bloch_sum_gamma` (per-T real-space S^T, H0^T blocks via two-cell
  merged AOBasis; Bloch sum at any k).
- **4d.4** `periodic_klopman_ohno_gamma` (Ewald-summed shell-resolved γ
  at Γ). γ = (γ - 1/R) + 1/R: residual sum over real-space lattice
  with `fsmooth(r, 10 Bohr)` blend matching tblite; Coulomb tail
  Ewald-summed.
- **4d.5–4d.7** Γ-only periodic SCC + k-point sampling + crystal-driver
  `XtbCalculator(Crystal)`. SCC converges on real molecular crystals,
  no catastrophic divergence.
- **4d.8** Periodic AES Ewald + Bra/Ket atom-centered AO multipoles +
  DIIS on (qsh; dipm; qpat). `build_multipole_ewald_tensors`,
  `build_periodic_multipole_ao`, `compute_camm_moments_periodic`,
  `apply_anisotropic_h1_periodic`.
- **4d.9** k-point multipoles. Per-T D_ket/D_bra/Q_ket/Q_bra blocks,
  `bloch_sum_triple` / `bloch_sum_array6`, `apply_anisotropic_h1_kpoint`,
  `accumulate_camm_kpoint`. Validated against Γ-only on water (1×1×1 to
  1e-9 Ha, 2×2×2 vacuum-padded to 1e-7 Ha) and BENZEN (1×1×1 = Γ-only
  to 12 decimals; 2×2×2 = +0.4 mHa BZ correction).
- **4d.10** Fused S/D/Q cross-block kernel + TBB threading. Per-T loops
  + per-k Bloch sums + per-k eigensolves all parallel. Determinism
  verified across 4 crystals (12-decimal-identical to single-thread).

#### AES sign-convention rework (resolved)

See commits `397de5a99` + `10bb0630c`. Root causes were (1) trace removal
applied post-CAMM rather than at the integral level, (2) CT kernel
potential sign mismatch, (3) H1 apply sign mismatch, (4) e01 sign mismatch
from the `rij = R_j - R_i` vs `(R_i - R_j)` pair-loop convention. After
the fix: water molecule total -5.07036943 Ha vs tblite -5.07036967 Ha
(Δ = 0.3 µHa).

#### Known issue — periodic crystals ~0.4–0.6 mHa/atom more bound than tblite

| Crystal   | Atoms | OCC total       | tblite total    | Δ/atom (mHa) |
|-----------|------:|-----------------|-----------------|-------------:|
| BENZEN    |    48 | −63.59211       | −63.57247       | −0.41        |
| ACENAP03  |    88 | −123.31813      | −123.27769      | −0.46        |
| ACSALA07  |    84 | −158.66089      | −158.62756      | −0.40        |
| ANTCEN14  |    48 | −70.05917       | −70.03133       | −0.58        |
| CITRAC10  |    84 | −181.64994      | −181.61219      | −0.45        |

Single-µHa molecular agreement remains the strongest correctness signal.
tblite is the only public GFN2 implementation (dftb+ delegates to tblite),
so we don't have a third reference. tblite's "dispersion energy" line is
only the 3-body ATM term — the 2-body C6/C8 piece is folded into the
"electronic energy" — confirmed by reading tblite/disp/d4.f90.

Suspect order:
1. **WSC averaging** (top suspect). tblite's `get_amat_3d` and
   `get_multipole_matrix_3d` average Ewald per-image contributions over
   equidistant Wigner–Seitz images of each pair vector with weight
   `1/nimg`. For low-symmetry molecular crystals `nimg = 1` for most
   pairs, but accumulates over O(N²) pair-images.
2. **Multipole AO cutoff conventions** (untested). Worth confirming we
   match tblite's `get_cutoff(calc%bas, accuracy)` ≈ 17–20 Bohr.
3. **Periodic CN sensitivity** (resolved — algorithmic match verified
   ka=10, kb=20, r_shift=2 Bohr; cutoff differences far enough that the
   count function is ≲1e-4).

### Phase 5 — Gradients + frequencies (in progress)

- **5a** ✅ `XtbCalculator::gradient_numerical(step_bohr)` — central
  differences, used as the validation reference.
- **5b** ✅ Easy analytical gradients: `repulsion_energy_and_gradient`,
  `gfn_coordination_numbers_with_gradient`. Match FD to 1e-9.
- **5c** ✅ `h0_scc_gradient` — analytical H0 + Pulay + V_q-via-S
  contribution. Z = P·X − W − ½·P·(V+V) where V is the converged shell
  shift potential. γ-gradient factor-of-2 fix included. Matches the
  full reconverged-SCC FD to <1 µHa.
- **5d-partial** ✅ γ matrix gradient (`klopman_ohno_gamma_energy_gradient`)
  validated to FD 1e-9.
- **`XtbCalculator::gradient()`** ✅ assembles analytical pieces + native
  D4 dispersion. Runs charge-only SCC for self-consistency (so the
  reported energy matches the gradient — multipole contributions are
  excluded; ~1 mHa energy gap vs full GFN2 on water). Self-consistency
  vs FD: ≤ 5×10⁻⁵ Ha/Bohr; gap is the missing CPSCF response of D4
  through Mulliken charges (xtb has the same gap).
- **GFN2 wired into geometry opt** ✅. `MethodKind::GFN2` branch in
  `src/driver/geometry_optimization.cpp`.
- **5d-rest** ⏳ CAMM + anisotropic multipole gradients (see Open work
  list above).
- **5e** ⏳ Hessian + frequencies (see Open work list above).

### Phase 6 — Native dispersion ✅

In-tree DFT-D4 + DFT-D3-BJ replacing the cpp-d4 dependency.

- **6a** Native D4 for GFN2-xTB. Matches xtb to 0.28 µHa on water,
  94 µHa on rubrene (after multi-Gaussian ref weights and ATM sign fixes).
- **6b** Hirshfeld refq variant for DFT-D4 (`RefqMode::DFT`); 152
  per-functional damping presets in `share/dftd4/functionals.json`.
- **6c** Numerical gradient via 5-point central diff.
- **6d** All D4 use sites migrated to the native backend.
- **6e** Dropped cpp-d4 CMake dependency.
- **6f** Analytical D4 + D3 gradients (2-body BJ + ATM angular + ∂C6/∂CN
  chain). Plus EEQ derivative for full DFT-D4 forces. Match FD to
  1e-9 Ha/Bohr.
- **6g** Native D3-BJ. Refdata (`share/dftd3/refdata.json`, 94 elements)
  + 94-functional damping. Validates against s-dftd3 to <1 µHa.

### Bindings + ergonomics ✅ (post-Phase 4d)

- API tidy on `XtbCalculator`: unified `XtbResult` struct (`SccResult` /
  `PeriodicSccResult` are aliases), getter for every setter, `to_crystal`,
  `update_structure(positions, lattice)`, doxygen docstrings, stubs for
  `set_solvent` / `set_num_unpaired_electrons`.
- Renames: `NativeCalculator` → `XtbCalculator`,
  `Gfn2Calculator` → `Gfn2Engine`. Files renamed via `git mv` so blame
  history is preserved.
- Dropped `shared_ptr<Gfn2Parameters>` (no sharing was happening) and
  `unique_ptr<Gfn2Engine>` (only there for forward-declare); now plain
  value + `std::optional<Gfn2Engine>`. Move/copy semantics now explicit.
- Python (nanobind): `occpy.XtbCalculator`, `occpy.XtbResult`,
  `occpy.XtbMethod`. Top-level (matching `occpy.Molecule` etc.).
- JS (emscripten / embind): `Module.XtbCalculator.fromMolecule(mol)`.
  Constructors are factories because embind can't overload by argument
  type, only arity.
- Test cleanup: 54 → 50 cases, helper geometries (`water_atoms()` /
  `methane_atoms()`), section headers; dropped 4 obsolete sanity tests
  subsumed by stronger checks.
