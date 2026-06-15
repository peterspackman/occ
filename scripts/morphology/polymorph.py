"""Size-dependent particle free energy and polymorph stability crossover.

The free energy of a finite crystalline particle of ``N`` molecules is

    G(N) = N * mu_bulk + E_excess(N)

with ``mu_bulk`` the per-molecule bulk (lattice/solution) energy from the cg results and
``E_excess(N)`` the surface + edge + corner excess from the broken-bond cluster enumeration.

For two polymorphs the bulk term dominates at large ``N`` (the thermodynamically stable form
wins), while the excess term - cheaper per molecule for the form with lower surface energy -
can invert the ordering for small particles.  The crossover ``N`` (if any) is where
``G_A(N) - G_B(N)`` changes sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energy import analytic_surface, cluster_excess, size_for_n_molecules


# ---- bulk reference ---------------------------------------------------
def mu_bulk(model, mode: str = "crystal") -> float:
    """Per-molecule bulk reference energy (kJ/mol), averaged over asymmetric molecules.

    ``mode`` selects which cg per-molecule total to use (applied identically to both
    polymorphs so only the difference matters):

    * ``crystal``  - vacuum lattice energy = ``0.5 * crystal_energy`` (default).  occ's
      ``crystal_energy`` is the *un-halved* sum over neighbours (each pair shared between two
      molecules), so the per-molecule cohesive energy is half of it - the same 0.5 occ uses
      for the sublimation enthalpy.  This matters: the crossover size scales as 1/Delta_mu^3.
    * ``solution`` - ``0.5 * crystal_energy + solution_term`` (lattice + per-molecule solvation)
    """
    totals = model.cg.totals_per_molecule
    if mode == "crystal":
        vals = [0.5 * t["crystal_energy"] for t in totals]
    elif mode == "solution":
        vals = [0.5 * t["crystal_energy"] + t["solution_term"] for t in totals]
    else:
        raise ValueError(f"unknown mu_bulk mode {mode!r}")
    return float(np.mean(vals))


@dataclass
class GCurve:
    """G(N) for one polymorph/shape."""

    label: str
    n: np.ndarray
    g: np.ndarray  # total free energy (kJ/mol)
    g_per_molecule: np.ndarray
    e_excess: np.ndarray
    mu_bulk: float


def g_of_size(model, poly, n_targets, mu_mode: str = "crystal", n_registry: int = 3,
              label: str = "") -> GCurve:
    """Free-energy curve ``G(N)`` over the requested molecule counts."""
    mu = mu_bulk(model, mu_mode)
    ns, gs, exc = [], [], []
    for n in n_targets:
        s = size_for_n_molecules(model, poly, n)
        e, n_mol = cluster_excess(model, poly, s, n_registry=n_registry)
        ns.append(n_mol)
        exc.append(e)
        gs.append(n_mol * mu + e)
    ns = np.array(ns, dtype=float)
    gs = np.array(gs)
    return GCurve(label or model.cg.title, ns, gs, gs / ns, np.array(exc), mu)


def crossover_size(curve_a: GCurve, curve_b: GCurve):
    """Estimate the molecule count where ``G_per_molecule`` of two polymorphs cross.

    Returns ``(n_cross, stable_small, stable_large)`` or ``(None, ...)`` if the per-molecule
    free energies do not cross over the sampled range (one form is always more stable).
    """
    # interpolate both onto a common N grid (per-molecule free energy)
    n_grid = np.geomspace(
        max(curve_a.n.min(), curve_b.n.min()),
        min(curve_a.n.max(), curve_b.n.max()),
        400,
    )
    ga = np.interp(n_grid, curve_a.n, curve_a.g_per_molecule)
    gb = np.interp(n_grid, curve_b.n, curve_b.g_per_molecule)
    diff = ga - gb
    small = curve_a.label if diff[0] < 0 else curve_b.label
    large = curve_a.label if diff[-1] < 0 else curve_b.label
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) == 0:
        return None, small, large
    i = sign_change[0]
    # linear interpolation of the crossing
    n_cross = n_grid[i] - diff[i] * (n_grid[i + 1] - n_grid[i]) / (diff[i + 1] - diff[i])
    return float(n_cross), small, large
