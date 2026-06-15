# Particle size/shape-dependent energy

Turns `occ cg` crystal-growth energies into a **size- and shape-dependent particle free
energy**, broken down into surface / edge / corner contributions, and compares the **relative
stability of two polymorphs in a given solvent** as a function of particle size.

The per-structure computation now lives in **occ (C++)** — `occ cg --morphology` writes a
`morphology` block into the cg results JSON (`occ::driver::compute_crystal_morphology`,
`src/driver/crystal_morphology.cpp`). The Python here is the thin "compare" layer:
`occ_morphology.py` reads one or two of those JSONs and reports the shape, surface/edge/corner
energies, and the polymorph crossover.

```bash
occ cg paracetamol_I.cif  --morphology --solvent water -m gfn2-xtb
occ cg paracetamol_II.cif --morphology --solvent water -m gfn2-xtb
python scripts/morphology/occ_morphology.py \
    paracetamol_I_water_cg_results.json paracetamol_II_water_cg_results.json --plot out.png
# -> crossover at N ~ 1118 (~3.7 nm): Form I favoured below, Form II above
```

The original pure-Python prototype (`model.py`, `shape.py`, `energy.py`, `polymorph.py`,
`run_morphology.py`) remains as the reference implementation the C++ was validated against
(it reproduces the C++ numbers exactly). The one-off `*_crossover.py` / `make_plots.py`
scripts are worked examples (acetic acid, urea, aspirin, paracetamol).

## Usage

Generate cg results with surface energies (needs enough facets to close the Wulff shape):

```bash
export OCC_DATA_PATH=$HOME/git/occ/share
occ cg mycrystal.cif --surface-energies 80 --solvent water -m ce-b3lyp
```

Then run the prototype with the project venv (which has `occpy` + numpy/scipy):

```bash
# one structure: shape + per-facet surface energies + size-dependent excess
.venv/bin/python scripts/morphology/run_morphology.py mycrystal_water_cg_results.json

# two polymorphs: also the stability crossover size
.venv/bin/python scripts/morphology/run_morphology.py A_water_cg_results.json B_water_cg_results.json --plot out.png
```

The CIF is inferred from the JSON `title`; override with `--cif a.cif,b.cif`.

## Physics

For a finite crystalline particle of `N` molecules,

```
G(N) = N * mu_bulk + E_excess(N)
E_excess(N) = sum_f gamma_f A_f  +  sum_e lambda_e L_e  +  sum_v eps_v   (surface + edge + corner)
```

The equilibrium (Wulff) shape is fixed by the facet energies `gamma_f`; scaling linear size
makes area ~ s^2, edge length ~ s, corner count fixed, and `N ~ s^3`.  For two polymorphs the
bulk term dominates at large `N` (the thermodynamic form wins) while the cheaper-surface form
can win for small particles - the crossover size is where `G/N` crosses.

Two routes to `E_excess`, cross-validated:

* **Exact cluster** (`energy.cluster_excess`) - tile the crystal inside the scaled
  polyhedron and sum the solvated energy of every nearest-neighbour bond crossing the
  boundary, `E_excess = 1/2 * sum_broken E_ij`.  The lattice/face **registry is minimised**
  (a finite crystal adopts its lowest-energy terminations), which realises occ's optimal cuts
  and removes the size-dependent registry noise of an arbitrary convex slice.
* **Analytic surface** (`energy.analytic_surface`) - `sum_f sigma_f A_f` with
  `sigma_f = gamma_f / KJ2J` (kJ/mol/A^2) and Wulff face areas.

## Modules

| module | role |
|--------|------|
| `cg_data.py` | parse the cg results JSON; facet gamma reconstruction; geometry/cell helpers |
| `model.py` | occpy crystal + cg energies; stamps `interaction_id`/energy onto uc dimers via occ's `InteractionMapper`; per-molecule neighbour bonds |
| `shape.py` | Wulff / user-morphology polyhedron (scipy half-space intersection); faces/edges/corners, areas, inside-test |
| `energy.py` | exact cluster enumeration (+registry minimisation), analytic surface, feature decomposition, size fit |
| `polymorph.py` | `mu_bulk`, `G(N)`, crossover size |
| `run_morphology.py` | CLI |
| `test_morphology.py` | pytest (skips if no cg fixture) |

## Key validated facts

* A fresh `unit_cell_dimers` has **unset** `interaction_id`.  Driving occ's own
  `InteractionMapper` from Python (with `solvated` energies wrapped as `DimerResult`s)
  reproduces every reported facet energy **exactly** (verified 80/80 on acetic acid).
  `solvated[asym]` is ordered as `symmetry_unique_dimers(outer_radius).molecule_neighbors[asym]`;
  the outer radius (occ default `max_radius=30`) is recovered by matching that count.
* The scipy Wulff polyhedron agrees with occ's `WulffConstruction` (same vertex/corner count;
  V-E+F = 2).
* After registry minimisation the exact cluster surface density converges to within ~10-20%
  of the analytic optimal-cut `sigma_f` - the residual is the finite-size edge/corner term.

## Caveats / next steps

* Separating **edge** from **corner** by least-squares over a limited size range is
  ill-conditioned (collinear `s^2/s^1/s^0`).  The CLI reports the surface term from the exact
  analytic `gamma_f` and the **edge+corner** as the residual; per-edge/per-corner named
  energies (via local wedge/corner clusters at optimal registry) are a planned refinement.
* `mu_bulk` defaults to the cg `crystal_energy` (vacuum lattice energy); `--mu solution` uses
  `interaction_energy + solution_term`.  Use the same choice for both polymorphs.
* `cg_radius` (occ default 3.8 A nearest-atom) sets the bond set used for both the surface
  energies and the cluster; pass `--cg-radius` to match a non-default run.
