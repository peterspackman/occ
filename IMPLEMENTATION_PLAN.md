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

#### Phase 7A — Solvation plumbing (no physics) — STATUS: shipped (e7c81e081)

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

#### Phase 7B — CPCM-X (tblite-parity electrostatics) — STATUS: math + SCC integration done

Shipped:
- `include/occ/xtb/cpcmx.h`, `src/xtb/cpcmx.cpp`. Cavity via the existing
  `occ::solvent::surface` 146-point Lebedev builder; A-matrix and atom→
  cavity Coulomb B built and LU-factored once per `initialize()`; the
  symmetric negative-definite atom-resolved response operator
  `J_solv = −f(ε) B^T A^{-1} B` is precomputed so per-SCC `update(q)` is
  two GEMVs.
- CPCM ideal-conductor convention (`x = 0` in `f(ε) = (ε−1)/(ε+x)`); the
  Klamt classical-COSMO `x = 0.5` is opt-in via `CpcmXOptions::x`.
- Wired through `XtbCalculator::set_solvation_model`. Water in water
  converges in 11 SCC iters with ΔE_solv ≈ −3.7 kcal/mol — sensible vs
  the SMD experimental ΔG_solv of −6.3 kcal/mol (the rest is the missing
  CDS term that Phase 7C adds).
- Tests cover: math invariants (zero charges → zero E, vacuum limit
  ε=1 → exact gas, conductor limit saturates monotonically,
  variational consistency V_solv = ∂E/∂q to <1e-9), SCC integration
  (E_solv < E_gas, polarisation of charges in the expected direction).

Open (deferred, none blocking Phase 7C):
- Bit-validate against tblite CPCM/GFN2 on water / methanol / formamide.
  Needs `WITH_TBLITE=ON`; was off in this build. Target <5e-5 Ha after
  matching tblite's exact `keps` convention (currently CPCM ideal-cond).
- Increase Lebedev grid order from 146 to 590 to match tblite (currently
  capped by `solvent_surface()` calling `lebedev(146)`). Likely a small
  energy shift; revisit when bit-parity matters.

#### Phase 7F — Analytical solvation gradients (frozen cavity) — STATUS: shipped

Shipped:
- `occ::xtb::cosmo::gradient(positions, surface, q, σ, f(ε)) -> Mat3N`
  implements the standard frozen-cavity CPCM formula
  `σ·∂φ/∂R + 1/(2f(ε))·σ·∂A/∂R·σ`. Shared between CPCM-X and SMD ES.
- `XtbSolvationModel::gradient()` virtual hook (default returns empty).
  `CpcmXSolvationModel::gradient()` delegates to `cosmo::gradient`.
  `SmdSolvationModel::gradient()` = `cosmo::gradient` for ES + numerical
  FD on `atomic_surface_tension(R)` for CDS (cavity frozen, only
  σ_atom(R) varies).
- Models cache atom positions in `initialize()` and the latest atomic
  charges in `update()` so `gradient()` needs no parameters.
- `solvent_surface()` grew an `axis_aligned` flag (default `true` —
  preserves existing HF/DFT SMD numerics). The xtb path passes `false`
  so cavity points sit rigidly on their parent atoms, which is what the
  closed-form gradient assumes.
- `XtbCalculator::gradient()` folds V_solv into V_shell before
  `h0_scc_gradient` (so the Pulay/Z chain absorbs the density response
  to solvation, the same way it does for AES vs and γ·q) and then adds
  the explicit `m_solvation->gradient()` contribution at the end.
- FD-validated to <2×10⁻⁴ Ha/Bohr at 2 mBohr FD step on water (CPCM-X
  and SMD) and methane (SMD). Tighter than that runs into the SCF
  tolerance / FD noise floor (1e-7 Ha ÷ 2e-3 Bohr ≈ 5e-5 Ha/Bohr) and
  occasional cavity-mask flips in FD.

Open:
- Periodic solvation gradient. Currently the periodic path warns and
  ignores solvation entirely (Phase 7A note).

#### Phase 7G — Smooth cavity → continuous gradients — STATUS: shipped at 0.1 Bohr

Shipped:
- `occ::solvent::surface::solvent_surface(..., smoothing_width_bohr)`
  replaces the boolean mask with `weight_j = Π_{k ≠ atom_j} smoothstep(
  |r_j − R_k|, r_k, w)` when `w > 0`. `w = 0` (default) preserves the
  legacy boolean cavity bit-for-bit; ContinuumSolvationModel (SMD HF/DFT)
  callers see no change.
- `cosmo::gradient` gains an opt-in diagonal-A term:
  `∂A_ii/∂R_c = −½ A_ii · ∂ln(weight_i)/∂R_c`, with the smoothstep'
  flowing rigid-attachment chain rules into the same per-atom grad
  accumulator as the off-diagonal pieces.
- `CpcmXOptions::smoothing_width_bohr` and the SMD model's internal
  width default to **0.1 Bohr**. The CDS branch of SMD's gradient
  rebuilds the CDS cavity at each FD step so smooth-weight areas
  propagate.
- FD validation at the default 0.1 Bohr:
    • CpcmX model FD (isolated, fixed q): <3e-5 Ha/Bohr — essentially
      at the FD truncation floor.
    • CpcmX water (full pipeline): <2e-4 Ha/Bohr.
    • SMD water (full pipeline): <2e-4 Ha/Bohr.
    • SMD methane (full pipeline): <2e-4 Ha/Bohr.

Known limitation:
- Widening the smoothing past ~0.15 Bohr causes a sharp residual
  climb (≈1e-2 Ha/Bohr at 0.2 Bohr). Bisection localised it to the
  *off-diagonal* `cosmo::gradient` paths under smooth cavity — flipping
  the diagonal-A sign changes the residual by only ~7e-4. The leading
  hypothesis is that a second-order chain through the (now-geometry-
  dependent) cavity weights leaks into σ at the variational fixed
  point in a way Hellmann-Feynman doesn't immediately cancel. Not
  blocking — 0.1 Bohr gives FD-clean gradients and the cavity is C∞
  there.

#### Phase 7C — SMD on top of the CPCM-X backbone — STATUS: shipped

Shipped:
- Extracted the cavity-response builder into `occ::xtb::cosmo::build()`
  (`include/occ/xtb/cosmo_response.h`, `src/xtb/cosmo_response.cpp`) so
  the CPCM-X and SMD models share one implementation. `CpcmXSolvationModel`
  refactored to call it; tests unchanged.
- `include/occ/xtb/smd_xtb.h`, `src/xtb/smd_xtb.cpp`. Two cavities:
  the ES surface uses SMD `intrinsic_coulomb_radii` and feeds through
  `cosmo::build` exactly like CPCM-X; the CDS surface uses
  `smd::cds_radii` and is purely geometric — every element carries a
  fixed `(σ_atom + γ_macro) × area` contribution in Hartree. The
  per-element CDS energy vector is exposed verbatim (Phase 7D will plug
  this into `XtbResult`).
- `update(q)` only refreshes the ES branch; CDS rides along in
  `energy()`. `atom_potential()` is the ES Fock shift (CDS is q-free).
- Tests cover: math invariants (CDS sum matches per-element sum;
  variational consistency for the ES branch to <1e-9), SCC on water in
  water (energy stabilises within a few-kcal/mol window),
  hydrophobic-methane sanity (E_cds > 0). 28 new assertions in 3 cases.

Open (deferred):
- Bit-validate hydration free energies against published xtb SMD numbers
  on a 5-molecule benchmark. Needs reference data; not blocking 7D.
- Optionally switch the ES branch from classical-COSMO to IEFPCM to
  match the canonical SMD formulation; both produce similar magnitudes
  in this regime so it can wait.

#### Phase 7D — Per-element exposure — STATUS: shipped

Shipped:
- `occ::xtb::SolvationSurface { positions, areas, atom_index, energies }`
  and `SolvationSurfaces { coulomb?, cds? }` in
  `include/occ/xtb/solvation_interface.h`.
- `XtbSolvationModel::surfaces()` virtual hook (default `std::nullopt`).
- `cosmo::Response` now keeps `B` so models can recover the per-element
  source potential `φ = B·q` and compose `½ σ_i · φ_i` per-element ES
  energies that sum exactly to `½ q · V_solv`.
- `CpcmXSolvationModel::surfaces()` populates `coulomb` only; SMD
  populates both `coulomb` (ES) and `cds` (geometric).
- `XtbResult` grew `std::optional<SolvationSurfaces> solvation_surfaces`;
  `Gfn2Engine` refreshes `update(q_converged)` at the end of SCC so the
  per-element decomposition reflects the same charges the energy uses,
  then snapshots `surfaces()` into the result.
- Tests cover: per-element coulomb sum == E_es (CPCM-X + SMD),
  per-element cds sum == E_cds (SMD), total sum == model.energy(),
  shape invariants (lengths match, atom_index in valid range), no
  surfaces for null / gas-phase paths. 23 new assertions / 4 cases.

What's left for cg consumption:
- `occ::xtb::SolvationSurface` and `occ::cg::SolventSurface` are
  shape-compatible; Phase 7E supplies the adapter + cg pipeline
  changes.

#### Phase 7E — cg integration — STATUS: adapter shipped; driver wiring follow-up

Shipped:
- `occ::cg::from_xtb_surfaces(const xtb::SolvationSurfaces&)` adapter
  in `include/occ/cg/solvent_surface.h`, `src/cg/solvent_surface.cpp`.
  Maps xtb's per-element shape onto `cg::SMDSolventSurfaces`:
  positions/areas/energies copy verbatim; xtb's `coulomb.energies`
  already carries the full per-element ES contribution so the bundle's
  `electronic_energies` stays zero (cg's partitioner sums both). CPCM-X
  (no CDS) lands with an empty cds branch; `total_solvation_energy` is
  the xtb total.
- End-to-end test on the acetic-acid crystal: `XtbCalculator(mol)` with
  `SmdSolvationModel("water")` → `XtbResult.solvation_surfaces` →
  `from_xtb_surfaces` → `SolventSurfacePartitioner.partition(neighbours,
  surface)`. Forward sum of partitioned Coulomb / CDS contributions
  agrees with the model's `e_es()` / `e_cds()` to 1e-9 Ha. 17 assertions
  / 2 cases. Full cg suite at 218/218.

Driver migration shipped:
- `XTBCrystalGrowthCalculator` in `src/driver/crystal_growth.cpp` now
  drives the in-tree backend. `init_monomer_energies` runs
  `xtb::XtbCalculator` (gas + SMD-solvated) per monomer, stashes the
  per-element surfaces via `from_xtb_surfaces` into
  `m_solvated_surface_properties`. `converge_lattice_energy` no longer
  runs solvated-dimer calculations — the per-monomer surface partitioned
  over the crystal's neighbour list replaces that scheme entirely.
  `process_neighbors_for_symmetry_unique_molecule` now uses
  `cg::SolventSurfacePartitioner` exactly the way the CE model does
  (forward/reverse Coulomb + CDS contributions feeding
  `DimerSolventTerm.ab/ba/total`).
- `m_solvated_dimer_energies` and `m_partial_charges` deleted from the
  XTB calculator's state — both were unique to the weight scheme.

Open:
- `cg_json.h` per-element JSON dump: `{position, area, atom_index,
  neighbor_index, e_coulomb, e_cds, e_total}`. The data is already
  flowing through `SMDSolventSurfaces` + the partitioner; emitting a
  per-element JSON view is a straightforward serialise step that
  doesn't need any new physics.

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

### Wavefunction persistence — `occ tb -o` + periodic `to_wavefunction`

GFN2 results today reach the rest of occ in two ways: (a) molecular
single-points routed through `single_point_driver` (`occ scf
--method gfn2`) already produce a `Wavefunction` and `write_output_files`
saves a `.owf.json` reloadable by `occ cube`, `occ isosurface`, and
`occ cg`; (b) `occ tb` runs the SCC but never persists a wavefunction,
and `XtbCalculator::to_wavefunction()` crashes on periodic inputs
because `m_calc` is unemplaced for the Crystal ctor. These two gaps
block the natural workflow "run `occ tb`, then make a cube / isosurface
from the result". The lossiness of the JSON round-trip (xTB-specific
energy decomposition + charges collapse into `Energy::total` and
`Energy::nuclear_repulsion`) is a separate, lower-priority concern.

#### Stage 1 — periodic-safe `to_wavefunction()`

**Files to touch**:
- `src/xtb/xtb_calculator.cpp::to_wavefunction()` (line 631).

**Plan**:
- Replace `m_calc->basis()` / `m_calc->atoms()` / `m_calc->shell_table()`
  with locally built equivalents: `make_atoms(m_positions_bohr,
  m_atomic_numbers)`, then `build_aobasis(atoms, m_params)` and
  `build_shell_table(atoms, m_params)`. Both helpers already exist
  (`xtb/basis.h`, `xtb/gamma.h`) and are the same builders used by
  `Gfn2Engine` and the periodic SCC drivers, so molecular results are
  bit-identical.
- For periodic, the existing periodic SCC drivers
  (`gfn2_periodic_calculator.cpp:334-381`, `:704-749`) already write
  `density_matrix`, `orbital_coefficients`, `orbital_energies` at Γ
  into `XtbResult`, so the body of `to_wavefunction` is unchanged once
  basis/atoms come from a periodic-safe source.
- Docstring: clarify that the periodic case is a Γ-only central-cell
  snapshot — downstream cube/isosurface treats it as a molecular
  wavefunction (no Bloch sum at evaluation time). Suitable for
  visualisation; not a full periodic wavefunction.

**Tests** (new `tests/xtb_wavefunction_io_tests.cpp`):
- Molecular `XtbCalculator(water)` → `to_wavefunction()` → save .owf.json
  → reload → compare D, C, ε, atoms, nbf to the original.
- Same for `XtbCalculator(benzene_crystal)` Γ-only.
- After reload, `wfn.electron_density(test_points)` matches the
  in-memory `wfn.electron_density(test_points)` to 1e-10 — this is
  what the cube/isosurface pipelines actually call.

#### Stage 2 — wire `occ tb -o` to write wavefunctions

**Files to touch**:
- `include/occ/main/occ_tb.h` — add `std::vector<std::string> formats{"json"};`
  to `TbConfig`.
- `src/main/occ_tb.cpp` — CLI plumbing + write call sites.

**Plan**:
- Add `tb->add_option("-o,--output", cfg->formats, "wavefunction output formats (json/fchk; empty disables)")`,
  mirroring `occ scf`'s `OutputInput::formats`.
- Factor a `write_wavefunction(const XtbCalculator&, const std::string &input_path, const std::vector<std::string> &fmts, const std::string &suffix = "")` helper:
  for each format, build path `<stem><suffix>.owf.<fmt>` and call
  `calc.to_wavefunction().save(path)`. Skip when `fmts` is empty or
  contains only blanks.
- Call sites:
  - `run_molecular` single-point: write after `calc.single_point_energy()`.
  - `run_molecular --opt`: `geometry_optimization` already returns a
    `Wavefunction`; just call `wfn.save(...)` directly with suffix
    `_opt` instead of running another SCC.
  - `run_periodic`: write after the SCC. For `--lattice-energy`, write
    only the crystal wfn (one file); skip per-monomer to keep the
    output set predictable.
  - `--freq`: vibrational analysis doesn't change the wavefunction;
    the converged single-point wfn is what we save.

**Success criteria / smoke tests**:
- `occ tb water.xyz` produces `water.owf.json`; loading it back with
  `Wavefunction::load` succeeds and `wfn.method == "GFN2-xTB"`.
- `occ tb benzene.cif` produces `benzene.owf.json`.
- `occ tb water.xyz --opt` produces `water_opt.owf.json` with the
  optimised geometry.
- `occ tb water.xyz -o` (empty list) writes nothing; energy summary
  unchanged.
- End-to-end smoke: `occ tb water.xyz && occ cube water.owf.json
  --property rho` produces a non-empty cube.

#### Stage 3 (optional) — preserve xTB-specific extras in the JSON

**Why**: `to_wavefunction()` currently folds `repulsion_energy` into
`Energy::nuclear_repulsion`, drops `dispersion_energy` and
`scc_energy` (only `total = scc + rep + disp` survives), and discards
`atomic_charges`, `shell_charges`, `converged`, `n_iterations`,
`orbital_occupations`, `solvation_surfaces`. Round-trip loses the
breakdown a user would expect to inspect after reload.

**Plan (option A — JSON-only, recommended)**:
- In `wavefunction_json.cpp::to_json/from_json`, when `wfn.method`
  starts with "GFN2", emit/consume an `"xtb"` block holding
  `{ scc_energy, repulsion_energy, dispersion_energy, atomic_charges,
    shell_charges, converged, n_iterations }`. Store it on
  `Wavefunction` as either an `std::optional<nlohmann::json>
  method_extras` (free-form) or a typed `XtbExtras` (clean but adds
  a header dependency).
- Recommend: `std::optional<nlohmann::json> method_extras` — keeps
  `wavefunction.h` free of xTB-specific types and lets us extend the
  block later (e.g. AES energy, dispersion components) without
  breaking the public struct.

**Plan (option B — typed)**: add a small `XtbExtras` struct to
`qm::Wavefunction`. Cleaner C++ API, but pulls xTB concepts into
the core qm header. Skip unless the consumers (e.g. `occ cg`) need
typed access without going through `nlohmann::json`.

**Tests**: round-trip the extras and assert equality field-by-field
against the original `XtbResult`. Skip in the molecular wavefunction
JSON test until this stage lands.

#### Sequencing

Stage 1 is a prerequisite for Stage 2 on the periodic side (otherwise
`occ tb crystal.cif` would crash inside the new write step). Stage 1
is also small (~20-line refactor + a test file). Stage 2 is the
user-visible change. Stage 3 is independent and easy to defer.

Estimated effort: Stage 1 ≈ 50 lines + tests, Stage 2 ≈ 80 lines
including CLI, Stage 3 ≈ 60 lines + tests if pursued.

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
