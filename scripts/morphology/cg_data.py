"""Parse ``occ cg`` result JSON for the particle morphology prototype.

The ``<basename>_<solvent>_cg_results.json`` file written by ``occ cg
... --surface-energies N`` contains everything we need:

* ``crystal``            - unit cell matrices, space group operations, unit cell atoms
* ``totals_per_molecule``- per (asymmetric) molecule crystal/interaction/solution energies
* ``pairs``              - per asymmetric molecule, the nearest-neighbour dimers with their
                           full (solvated) energy breakdown and atom offsets
* ``surface_cuts``       - ``solvated``/``vacuum`` per-interaction energy tables (indexed by
                           interaction id) and ``surface_energies.facets`` with the broken-bond
                           ``interaction_energy_counts`` for every surface cut.

The key verified relationship (reproduces ``facet.energy`` exactly) is

    gamma_f = sign * 0.5 * KJ_PER_MOL_TO_J_PER_M2
              * sum_id counts[asym][id] * solvated[asym][id]["Total"] / area_f

with ``sign = +1`` for solvated facet energies and ``-1`` for vacuum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import gcd
from typing import Any

import numpy as np

# kJ/mol per Angstrom^2  ->  J/m^2  (matches src/driver/crystal_surface_energy.cpp)
KJ_PER_MOL_TO_J_PER_M2 = 0.16604390671


@dataclass
class Facet:
    """A single surface cut from the cg results."""

    hkl: tuple[int, int, int]
    offset: float
    area: float  # Angstrom^2 (2D repeat cell area)
    energy: float  # J/m^2 as reported by occ
    counts: list[list[int]]  # [asym_mol][interaction_id] broken-bond counts

    def reduced_hkl(self) -> tuple[int, int, int]:
        h, k, l = self.hkl
        g = gcd(gcd(abs(h), abs(k)), abs(l)) or 1
        return (h // g, k // g, l // g)


@dataclass
class CGData:
    """In-memory view of an ``occ cg`` results JSON."""

    raw: dict[str, Any]
    title: str
    solvent: str
    direct: np.ndarray  # 3x3, columns are lattice vectors a,b,c (cart = direct @ frac)
    reciprocal: np.ndarray  # 3x3 reciprocal matrix as stored by occ
    rotations: list[np.ndarray]  # fractional rotation parts of the space group ops
    totals_per_molecule: list[dict[str, float]]
    facets: list[Facet]
    solvated: list[list[dict[str, float]]]  # [asym][interaction_id] -> components
    vacuum: list[list[dict[str, float]]]

    # ---- construction -------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "CGData":
        with open(path) as fh:
            raw = json.load(fh)
        if raw.get("result_type") != "cg":
            raise ValueError(f"{path}: not an occ cg results file")
        if "surface_cuts" not in raw:
            raise ValueError(
                f"{path}: no surface_cuts block - re-run occ cg with --surface-energies N"
            )
        crystal = raw["crystal"]
        uc = crystal["unit cell"]
        direct = np.array(uc["direct_matrix"], dtype=float)
        reciprocal = np.array(uc["reciprocal_matrix"], dtype=float)
        rotations = [
            np.array(op["seitz"], dtype=float)[:3, :3]
            for op in crystal["space group"]["symmetry_operations"]
        ]
        sc = raw["surface_cuts"]
        facets = [
            Facet(
                hkl=tuple(int(x) for x in f["hkl"]),
                offset=float(f["offset"]),
                area=float(f["area"]),
                energy=float(f["energy"]),
                counts=[[int(c) for c in row] for row in f["interaction_energy_counts"]],
            )
            for f in sc["surface_energies"]["facets"]
        ]
        return cls(
            raw=raw,
            title=raw.get("title", ""),
            solvent=raw.get("solvent", ""),
            direct=direct,
            reciprocal=reciprocal,
            rotations=rotations,
            totals_per_molecule=raw.get("totals_per_molecule", []),
            facets=facets,
            solvated=sc["solvated"],
            vacuum=sc["vacuum"],
        )

    # ---- energy tables ------------------------------------------------
    def interaction_energy(
        self, asym: int, interaction_id: int, *, solvated: bool = True, key: str = "Total"
    ) -> float:
        table = self.solvated if solvated else self.vacuum
        return float(table[asym][interaction_id][key])

    def facet_gamma(self, facet: Facet, *, solvated: bool = True) -> float:
        """Reconstruct the facet surface energy (J/m^2) from broken-bond counts.

        Matches ``occ``'s reported ``facet.energy`` (sign handled implicitly: the
        solvated interaction energies are already positive 'cost to break' terms).
        """
        table = self.solvated if solvated else self.vacuum
        sign = 1.0 if solvated else -1.0
        total = 0.0
        for asym, row in enumerate(facet.counts):
            for iid, count in enumerate(row):
                if count:
                    total += count * table[asym][iid]["Total"]
        return sign * 0.5 * total / facet.area * KJ_PER_MOL_TO_J_PER_M2

    # ---- facet selection for the Wulff/user shape ---------------------
    def unique_facets(self, *, min_energy: float = 1e-6) -> list[Facet]:
        """One facet per (reduced hkl) direction: the lowest positive-energy cut.

        This is the morphologically relevant set - the most stable cut for each
        crystallographic direction - and avoids the invalid/negative cuts that the
        cg driver also records.
        """
        best: dict[tuple[int, int, int], Facet] = {}
        for f in self.facets:
            if f.energy <= min_energy:
                continue
            key = f.reduced_hkl()
            if key not in best or f.energy < best[key].energy:
                best[key] = f
        return list(best.values())

    # ---- geometry helpers ---------------------------------------------
    def to_cartesian(self, frac: np.ndarray) -> np.ndarray:
        return self.direct @ frac

    def to_fractional(self, cart: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.direct, cart)

    @property
    def cell_volume(self) -> float:
        return abs(float(np.linalg.det(self.direct)))

    @property
    def n_unit_cell_molecules(self) -> int:
        # number of (asymmetric) molecules * number of space group operations
        return len(self.totals_per_molecule) * len(self.rotations)

    @property
    def molecular_volume(self) -> float:
        """Cell volume per molecule (Angstrom^3) - the bulk volume of one molecule."""
        n = self.n_unit_cell_molecules
        return self.cell_volume / n if n else float("nan")
