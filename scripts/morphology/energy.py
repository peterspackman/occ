"""Particle excess energy: exact finite-cluster enumeration and analytic decomposition.

Two independent routes to the size-dependent excess (surface + edge + corner) energy of a
finite crystalline particle, in kJ/mol:

* **Exact cluster** - tile the crystal inside the scaled polyhedron, then sum the solvated
  interaction energy of every nearest-neighbour bond that crosses the particle boundary
  (one endpoint inside, one outside).  ``E_excess = 1/2 * sum_broken E_ij``.

* **Analytic surface term** - ``sum_f sigma_f * A_f(s)`` with ``sigma_f = gamma_f / KJ2J`` the
  surface-energy density (kJ/mol/Angstrom^2) and ``A_f(s)`` the face area at scale ``s``.

Fitting the exact ``E_excess(s)`` against the scaled geometry ``a * (area) + b * (edge length)
+ c * (corners)`` separates the surface / edge / corner contributions; the surface coefficient
must reproduce the analytic ``sum_f sigma_f A_f`` (cross-validation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cg_data import KJ_PER_MOL_TO_J_PER_M2


def size_for_n_molecules(model, poly, n_target: int) -> float:
    """Scale ``s`` whose polyhedron volume holds ~``n_target`` molecules."""
    return (n_target * model.cg.molecular_volume / poly.volume) ** (1.0 / 3.0)


def _lattice_cells(model, poly, s: float) -> np.ndarray:
    """Integer cell translations (h,k,l) whose cell could overlap the scaled shape."""
    direct = model.cg.direct
    verts = poly.vertices * s  # scaled corner coordinates (Angstrom)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    pad = np.abs(direct).sum(axis=1)  # pad by the cell diagonal
    corners = np.array(
        np.meshgrid(*[[lo[i] - pad[i], hi[i] + pad[i]] for i in range(3)])
    ).reshape(3, -1)
    frac = np.linalg.solve(direct, corners)
    ns = [range(int(np.floor(frac[i].min())), int(np.ceil(frac[i].max())) + 1) for i in range(3)]
    return np.array(np.meshgrid(*ns)).reshape(3, -1).T  # (M,3) integer cells


@dataclass
class ParticleEnergy:
    s: float
    n_molecules: int
    e_exact: float  # kJ/mol, exact broken-bond excess
    e_surface_analytic: float  # kJ/mol, sum_f sigma_f A_f(s)
    area: float  # Angstrom^2, total surface area at this size
    edge_length: float  # Angstrom
    n_corners: int


def _registry_offsets(n: int) -> np.ndarray:
    """Fractional sub-cell translations used to relax the lattice/face registry."""
    g = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.array(np.meshgrid(g, g, g)).reshape(3, -1).T


def tile_inside(model, poly, s: float, frac_offset=None):
    """Molecules inside the scaled shape (optionally shifting the lattice by ``frac_offset``).

    Returns ``(positions, fracs, uc_idx, inside_keys)`` (``inside_keys`` is a sorted int array).
    """
    cells = _lattice_cells(model, poly, s)
    trans = cells @ model.cg.direct.T  # cartesian translations
    centroids = model.uc_centroids  # (n_uc, 3)
    if frac_offset is not None:
        centroids = centroids + model.cg.direct @ np.asarray(frac_offset)
    pos = (centroids[:, None, :] + trans[None, :, :]).reshape(-1, 3)
    uc_idx = np.repeat(np.arange(len(centroids)), len(cells))
    mask = poly.inside_mask(pos, s)
    pos, uc_idx = pos[mask], uc_idx[mask]
    fracs = np.linalg.solve(model.cg.direct, pos.T).T
    inside_keys = np.sort(model.mol_ikeys(fracs))
    return pos, fracs, uc_idx, inside_keys


def _broken_per_molecule(model, fracs, uc_idx, inside_keys):
    """Per-molecule total broken-bond energy (kJ/mol), vectorised by uc-molecule group."""
    broken = np.zeros(len(fracs))
    for u in range(len(model.uc_centroids)):
        sel = np.where(uc_idx == u)[0]
        if len(sel) == 0:
            continue
        bonds = model.neighbor_bonds[u]
        fdisp = np.array([b.frac_disp for b in bonds])  # (nb,3)
        en = np.array([b.energy for b in bonds])  # (nb,)
        nbr = fracs[sel][:, None, :] + fdisp[None, :, :]  # (M,nb,3)
        keys = model.mol_ikeys(nbr.reshape(-1, 3)).reshape(len(sel), len(bonds))
        is_in = np.searchsorted(inside_keys, keys)
        is_in = (is_in < len(inside_keys)) & (
            inside_keys[np.clip(is_in, 0, len(inside_keys) - 1)] == keys
        )
        broken[sel] = ((~is_in) * en).sum(axis=1)
    return broken


def _excess_at(model, poly, s, frac_offset):
    pos, fracs, uc_idx, inside_keys = tile_inside(model, poly, s, frac_offset)
    broken = _broken_per_molecule(model, fracs, uc_idx, inside_keys)
    return 0.5 * broken.sum(), len(fracs)


def cluster_excess(model, poly, s: float, n_registry: int = 3) -> tuple[float, int]:
    """Exact excess energy (kJ/mol) and count, minimised over lattice/face registry.

    A finite crystal adopts the lowest-energy surface termination; scanning sub-cell lattice
    offsets and keeping the minimum realises those optimal cuts and removes the size-dependent
    registry noise of an arbitrary convex slice.
    """
    best_e, best_n = float("inf"), 0
    for off in _registry_offsets(n_registry):
        e, n = _excess_at(model, poly, s, off)
        if e < best_e:
            best_e, best_n = e, n
    return best_e, best_n


def _bond_range(model) -> float:
    """Maximum nearest-neighbour displacement magnitude (the broken-bond range)."""
    return max(
        (float(np.linalg.norm(b.displacement)) for bonds in model.neighbor_bonds for b in bonds),
        default=0.0,
    )


@dataclass
class FeatureDecomposition:
    s: float
    n_molecules: int
    e_total: float  # kJ/mol (== e_surface + e_edge + e_corner)
    e_surface: float
    e_edge: float
    e_corner: float
    area: float  # Angstrom^2
    edge_length: float  # Angstrom
    n_corners: int


def decompose_excess(
    model, poly, s: float, range_factor: float = 1.0, n_registry: int = 3
) -> FeatureDecomposition:
    """Split the exact broken-bond excess into surface/edge/corner by feature proximity.

    The lattice/face registry is relaxed (minimum-energy termination), then each boundary
    molecule is attributed by how many facet planes it sits next to (within the bond range):
    ``1`` -> surface, ``2`` -> edge, ``>=3`` -> corner.  All of its broken bonds go to that
    class, so the buckets sum to the exact total with no double counting.
    """
    normals = np.array([f.normal for f in poly.faces])
    dists0 = np.array([f.distance for f in poly.faces])
    rcut = _bond_range(model) * range_factor

    best = None
    for off in _registry_offsets(n_registry):
        pos, fracs, uc_idx, inside = tile_inside(model, poly, s, off)
        broken = _broken_per_molecule(model, fracs, uc_idx, inside)
        if best is None or broken.sum() < best[0]:
            best = (broken.sum(), pos, broken)
    _, pos, broken = best

    near = np.count_nonzero(dists0 * s - pos @ normals.T < rcut, axis=1)
    e_surface = 0.5 * broken[near <= 1].sum()
    e_edge = 0.5 * broken[near == 2].sum()
    e_corner = 0.5 * broken[near >= 3].sum()
    return FeatureDecomposition(
        s=s,
        n_molecules=len(pos),
        e_total=e_surface + e_edge + e_corner,
        e_surface=e_surface,
        e_edge=e_edge,
        e_corner=e_corner,
        area=poly.total_area(s),
        edge_length=poly.total_edge_length(s),
        n_corners=poly.n_corners(),
    )


def analytic_surface(model, poly, s: float) -> float:
    """Analytic surface excess (kJ/mol) = sum_f sigma_f A_f(s)."""
    total = 0.0
    for f in poly.faces:
        sigma = f.distance / KJ_PER_MOL_TO_J_PER_M2  # kJ/mol/Angstrom^2
        total += sigma * f.area * s * s
    return total


def energy_vs_size(model, poly, n_targets, n_registry: int = 3) -> list[ParticleEnergy]:
    """Exact + analytic excess energy across a range of target molecule counts."""
    out = []
    for n in n_targets:
        s = size_for_n_molecules(model, poly, n)
        e_exact, n_mol = cluster_excess(model, poly, s, n_registry=n_registry)
        out.append(
            ParticleEnergy(
                s=s,
                n_molecules=n_mol,
                e_exact=e_exact,
                e_surface_analytic=analytic_surface(model, poly, s),
                area=poly.total_area(s),
                edge_length=poly.total_edge_length(s),
                n_corners=poly.n_corners(),
            )
        )
    return out


def fit_surface_edge_corner(points: list[ParticleEnergy]):
    """Least-squares fit ``E_exact = a*area + b*edge_length + c*n_corners``.

    Returns ``(a, b, c)`` with ``a`` ~ surface energy density (kJ/mol/Angstrom^2), ``b`` the
    edge line energy (kJ/mol/Angstrom), ``c`` the per-corner energy (kJ/mol).
    """
    A = np.array([[p.area, p.edge_length, p.n_corners] for p in points])
    y = np.array([p.e_exact for p in points])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coef)
