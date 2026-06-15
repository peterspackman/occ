"""Energy-stamped crystal model built from an ``occ cg`` results JSON + its CIF.

This wraps occpy together with the cg results so that every unit-cell dimer carries the
correct solvated interaction energy.  It is the single source of truth used by both the
analytic facet/edge/corner decomposition and the exact finite-cluster enumeration.

The key step reproduces occ's own pipeline (validated to reproduce every reported facet
energy exactly): a fresh ``unit_cell_dimers`` has no ``interaction_id`` set, so we run
``occ``'s ``InteractionMapper`` to stamp ``interaction_id`` and ``interaction_energy`` onto
each unit-cell dimer from the ``solvated`` table in the JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import occpy

from cg_data import CGData, KJ_PER_MOL_TO_J_PER_M2


def _match_outer_radius(crystal, n_target: int, asym: int = 0) -> float:
    """Find the neighbour radius whose asym-molecule neighbour count matches the JSON.

    ``solvated[asym]`` is ordered exactly as ``symmetry_unique_dimers(outer_radius)
    .molecule_neighbors[asym]``; we must rebuild that list with the same length/order so the
    energies align.  The count is a step function of radius, so any radius inside the shell
    window gives the same list - we scan upward and return the first exact match.
    """
    last = -1
    for r in np.arange(4.0, 60.0, 0.5):
        n = len(crystal.symmetry_unique_dimers(float(r)).molecule_neighbors[asym])
        if n == n_target:
            return float(r)
        if n > n_target:
            raise ValueError(
                f"could not match outer radius: target {n_target} fell between "
                f"counts {last} and {n} near r={r:.1f}"
            )
        last = n
    raise ValueError(f"could not match outer radius for target neighbour count {n_target}")


@dataclass
class NeighborBond:
    """One directed unit-cell dimer: molecule ``a`` (inside cell) to neighbour ``b``.

    ``frac_disp`` (fractional ``b.centroid - a.centroid``) is identical for every translated
    copy of this bond, so a molecule at fractional centroid ``f`` bonds to ``f + frac_disp`` -
    an exact identity that avoids both Cartesian-rounding noise and the unit-cell-index/
    cell-shift convention pitfalls.
    """

    displacement: np.ndarray  # b.centroid - a.centroid (Angstrom, cartesian)
    frac_disp: np.ndarray  # the same displacement in fractional coordinates
    energy: float  # solvated interaction energy "Total" (kJ/mol)
    interaction_id: int


class MorphologyModel:
    """occpy crystal + cg energies, with unit-cell dimers stamped with solvated energies."""

    def __init__(self, cg: CGData, crystal, cg_radius: float = 3.8):
        self.cg = cg
        self.crystal = crystal
        self.cg_radius = cg_radius
        self.outer_radius = _match_outer_radius(crystal, len(cg.solvated[0]))

        self._sd = crystal.symmetry_unique_dimers(self.outer_radius)
        self.uc_dimers = crystal.unit_cell_dimers(cg_radius)
        self._stamp_energies()

        self.uc_molecules = crystal.unit_cell_molecules()
        self.uc_asym = [m.asymmetric_molecule_idx() for m in self.uc_molecules]
        self.uc_centroids = np.array([m.centroid().ravel() for m in self.uc_molecules])

    @classmethod
    def from_cg_json(cls, json_path: str, cif_path: str, cg_radius: float = 3.8):
        cg = CGData.from_json(json_path)
        crystal = occpy.Crystal.from_cif_file(cif_path)
        return cls(cg, crystal, cg_radius=cg_radius)

    # ---- energy stamping (occ InteractionMapper) ----------------------
    def _stamp_energies(self) -> None:
        cg = self.cg
        vec = []
        for asym, neighbors in enumerate(self._sd.molecule_neighbors):
            row = []
            for k, srd in enumerate(neighbors):
                dr = occpy.DimerResult(srd.dimer, True, srd.unique_index)
                for comp, val in cg.solvated[asym][k].items():
                    dr.set_energy_component(comp, float(val))
                row.append(dr)
            vec.append(row)
        solution_terms = [t["solution_term"] for t in cg.totals_per_molecule]
        # inversion = not asymmetric_partition; has_permutation_symmetry == inversion-symmetric
        inversion = bool(cg.raw.get("has_permutation_symmetry", True))
        mapper = occpy.InteractionMapper(self.crystal, self._sd, self.uc_dimers, inversion)
        mapper.map_interactions(solution_terms, vec)

    # ---- per-molecule neighbour bonds (for cluster enumeration) -------
    @cached_property
    def neighbor_bonds(self) -> list[list[NeighborBond]]:
        """For each unit-cell molecule, its stamped neighbour bonds (displacement + energy)."""
        out: list[list[NeighborBond]] = []
        for neighbors in self.uc_dimers.molecule_neighbors:
            bonds = []
            for srd in neighbors:
                d = srd.dimer
                disp = d.b.centroid().ravel() - d.a.centroid().ravel()
                bonds.append(
                    NeighborBond(
                        displacement=disp,
                        frac_disp=self.cg.to_fractional(disp),
                        energy=d.interaction_energy(),
                        interaction_id=d.interaction_id,
                    )
                )
            out.append(bonds)
        return out

    # integer encoding of a molecule identity from its (rounded) fractional centroid;
    # exact and vectorisable. Cells span a few tens, so frac*100 fits comfortably.
    _B = 1 << 21
    _OFF = 1 << 20

    def mol_ikeys(self, fracs) -> np.ndarray:
        """Vectorised integer identities for fractional centroids ``fracs`` (N,3)."""
        q = np.round(np.atleast_2d(fracs) * 100.0).astype(np.int64) + self._OFF
        return (q[:, 0] * self._B + q[:, 1]) * self._B + q[:, 2]

    # ---- surface energies (reproduce / extend the JSON) ---------------
    def facet_energy(self, hkl, offset: float) -> tuple[float, float]:
        """Return (gamma in J/m^2, area in Angstrom^2) for a surface cut.

        Uses the stamped uc_dimers, so it reproduces occ's reported facet energies exactly
        and works for arbitrary (hkl, offset).
        """
        surf = occpy.Surface(occpy.HKL(int(hkl[0]), int(hkl[1]), int(hkl[2])), self.crystal)
        res = surf.count_crystal_dimers_cut_by_surface(self.uc_dimers, float(offset))
        gamma = 0.5 * res.total_above(self.uc_dimers) / surf.area() * KJ_PER_MOL_TO_J_PER_M2
        return gamma, surf.area()

    def surface_normal(self, hkl) -> np.ndarray:
        surf = occpy.Surface(occpy.HKL(int(hkl[0]), int(hkl[1]), int(hkl[2])), self.crystal)
        n = np.asarray(surf.normal_vector()).ravel()
        return n / np.linalg.norm(n)
