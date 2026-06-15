"""Particle shapes: Wulff equilibrium polyhedron and user-specified morphology.

A shape is a convex polyhedron defined by facet half-spaces ``n_f . x <= h_f``.  For the
Wulff (equilibrium) shape the support distance ``h_f`` is proportional to the surface energy
``gamma_f`` (Wulff's theorem); for a user/growth morphology the user supplies the distances.

Scaling the particle by a linear factor ``s`` multiplies every ``h_f`` by ``s``: face areas
grow as ``s^2``, edge lengths as ``s``, the corner count is fixed, and the molecule count as
``s^3``.  The :class:`Polyhedron` is built once at unit support and exposes scaled geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

_DIR_TOL = 1e-4  # tolerance for de-duplicating facet directions
_ON_TOL = 1e-6  # tolerance for "vertex lies on facet plane"


def facet_halfspaces(cg, facets, distances=None):
    """Symmetry-expand facets into unit-support half-spaces ``(normal, distance, hkl)``.

    Matches ``occ``'s ``write_wulff`` normal construction: the cartesian normal of
    plane (hkl) is ``reciprocal @ hkl``; rotations are applied in direct fractional
    coordinates.  ``distances`` overrides the per-facet support distance (for a user
    morphology); by default it is the surface energy ``gamma``.
    """
    out = []  # (unit_normal, distance, reduced_hkl)
    for idx, f in enumerate(facets):
        hkl = np.array(f.hkl, dtype=float)
        h = float(f.energy if distances is None else distances[idx])
        hkl_frac = np.linalg.solve(cg.direct, cg.reciprocal @ hkl)
        for rot in cg.rotations:
            n = cg.direct @ (rot @ hkl_frac)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            out.append((n / norm, h, f.reduced_hkl()))
    return _dedup_directions(out)


def _dedup_directions(halfspaces):
    """Keep one half-space per direction (smallest support distance)."""
    kept: list[tuple[np.ndarray, float, tuple]] = []
    for n, h, hkl in halfspaces:
        for i, (n2, h2, _) in enumerate(kept):
            if np.dot(n, n2) > 1.0 - _DIR_TOL:
                if h < h2:
                    kept[i] = (n2, h, hkl)
                break
        else:
            kept.append((n, h, hkl))
    return kept


@dataclass
class Face:
    index: int
    normal: np.ndarray
    distance: float  # unit support distance
    hkl: tuple
    vertices: np.ndarray  # ordered polygon vertices at unit scale (V,3)
    area: float  # unit-scale area


@dataclass
class Edge:
    faces: tuple[int, int]
    length: float  # unit-scale length


@dataclass
class Corner:
    point: np.ndarray  # unit scale
    faces: tuple[int, ...]


@dataclass
class Polyhedron:
    halfspaces: list  # (unit_normal, distance, hkl)
    vertices: np.ndarray  # unit-scale corner coords (V,3)
    faces: list[Face]
    edges: list[Edge]
    corners: list[Corner]
    volume: float  # unit-scale volume

    # ---- construction -------------------------------------------------
    @classmethod
    def from_halfspaces(cls, halfspaces) -> "Polyhedron":
        normals = np.array([n for n, _, _ in halfspaces])
        dists = np.array([h for _, h, _ in halfspaces])
        # scipy form: A x + b <= 0  ->  [n, -h]
        A = np.hstack([normals, -dists[:, None]])
        interior = _interior_point(normals, dists)
        hs = HalfspaceIntersection(A, interior)
        V = hs.intersections
        if not np.all(np.isfinite(V)) or len(V) < 4:
            raise ValueError(
                "shape is unbounded - the facet set does not enclose a volume "
                "(need more surfaces / facets spanning all directions)"
            )
        hull = ConvexHull(V)
        V = V[hull.vertices]  # unique hull vertices
        # which half-spaces is each vertex on
        on = np.abs(V @ normals.T - dists) < _ON_TOL * max(1.0, dists.max())
        faces = []
        for fi in range(len(halfspaces)):
            vidx = np.where(on[:, fi])[0]
            if len(vidx) < 3:
                continue
            verts = _order_polygon(V[vidx], normals[fi])
            faces.append(
                Face(fi, normals[fi], dists[fi], halfspaces[fi][2], verts,
                     _polygon_area(verts, normals[fi]))
            )
        active = {f.index for f in faces}
        edges = _edges_from_vertices(V, on, active)
        corners = [
            Corner(V[v], tuple(i for i in np.where(on[v])[0] if i in active))
            for v in range(len(V))
        ]
        return cls(halfspaces, V, faces, edges, corners, hull.volume)

    # ---- scaled geometry ----------------------------------------------
    def total_area(self, s: float = 1.0) -> float:
        return s * s * sum(f.area for f in self.faces)

    def total_edge_length(self, s: float = 1.0) -> float:
        return s * sum(e.length for e in self.edges)

    def n_corners(self) -> int:
        return len(self.corners)

    def scaled_volume(self, s: float = 1.0) -> float:
        return self.volume * s ** 3

    def inside_mask(self, points: np.ndarray, s: float = 1.0) -> np.ndarray:
        """Boolean mask: which points lie inside the polyhedron scaled by ``s``."""
        normals = np.array([n for n, _, _ in self.halfspaces])
        dists = np.array([h for _, h, _ in self.halfspaces]) * s
        return np.all(points @ normals.T <= dists + 1e-9, axis=1)


# ---- public shape constructors ---------------------------------------
def wulff_shape(cg, facets=None) -> Polyhedron:
    """Equilibrium (Wulff) shape: support distance proportional to surface energy."""
    facets = facets if facets is not None else cg.unique_facets()
    return Polyhedron.from_halfspaces(facet_halfspaces(cg, facets))


def user_shape(cg, hkl_shifts: dict) -> Polyhedron:
    """User/growth morphology from ``{(h,k,l): support_distance}``.

    The supplied distances replace the equilibrium gammas; any (h,k,l) not listed is omitted.
    """
    from cg_data import Facet

    facets, distances = [], []
    for hkl, dist in hkl_shifts.items():
        facets.append(Facet(tuple(int(x) for x in hkl), 0.0, 0.0, float(dist), []))
        distances.append(float(dist))
    return Polyhedron.from_halfspaces(facet_halfspaces(cg, facets, distances))


# ---- geometry helpers -------------------------------------------------
def _interior_point(normals, dists):
    """A strictly interior point via Chebyshev centre (largest inscribed sphere)."""
    from scipy.optimize import linprog

    norm_rows = np.linalg.norm(normals, axis=1)
    A = np.hstack([normals, norm_rows[:, None]])
    res = linprog(
        c=np.array([0.0, 0.0, 0.0, -1.0]),
        A_ub=A,
        b_ub=dists,
        bounds=[(None, None)] * 3 + [(0, None)],
    )
    if not res.success or res.x[3] <= 1e-9:
        raise ValueError("no interior point: facets do not enclose a volume")
    return res.x[:3]


def _order_polygon(verts, normal):
    """Order coplanar vertices counter-clockwise about their centroid."""
    c = verts.mean(axis=0)
    u = verts[0] - c
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    ang = np.arctan2((verts - c) @ v, (verts - c) @ u)
    return verts[np.argsort(ang)]


def _polygon_area(verts, normal):
    c = verts.mean(axis=0)
    area = 0.0
    n = len(verts)
    for i in range(n):
        area += np.dot(normal, np.cross(verts[i] - c, verts[(i + 1) % n] - c))
    return abs(area) / 2.0


def _edges_from_vertices(V, on, active):
    """Edges = vertex pairs shared by two active faces."""
    edges = []
    seen = set()
    nfaces = on.shape[1]
    # map each face -> its vertices
    face_verts = {fi: set(np.where(on[:, fi])[0]) for fi in active}
    face_list = sorted(active)
    for ai in range(len(face_list)):
        for bi in range(ai + 1, len(face_list)):
            fa, fb = face_list[ai], face_list[bi]
            shared = face_verts[fa] & face_verts[fb]
            if len(shared) == 2:
                key = tuple(sorted(shared))
                if key in seen:
                    continue
                seen.add(key)
                v1, v2 = (V[i] for i in key)
                edges.append(Edge((fa, fb), float(np.linalg.norm(v1 - v2))))
    return edges
