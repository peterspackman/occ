"""Prototype: snap each Wulff facet to its optimal cleavage cut, check cluster==analytic.

For each facet (unit normal n_f, d-spacing d_f) the broken-bond energy of a flat cut at phase
phi is

    E(phi) = sum_{j,b: dq_b>0} e_b * ( floor((phi-phi_j)/d_f) - floor((phi-phi_j-dq_b)/d_f) )

with phi_j = n_f . centroid_j and dq_b = n_f . displacement_b.  The optimal offset delta_f =
argmin_phi E(phi).  We then place each facet plane at the optimal cleavage layer nearest gamma_f*s
(a sub-d_f shift; the shape at scale s >> d_f is unchanged) and re-run the cluster.
"""
import sys

sys.path.insert(0, "scripts/morphology")
import numpy as np
import occpy

from cg_data import KJ_PER_MOL_TO_J_PER_M2 as K
from energy import _lattice_cells, size_for_n_molecules
from model import MorphologyModel
from shape import wulff_shape


def optimal_offsets(model, poly):
    """Per active facet: (d_f, delta_f) where delta_f = occ's optimal offset * d_f.

    occ's surface-energy code already finds the lowest-energy cut offset per facet; the cut
    plane sits at n_f.x mod d_f == offset * d_f (verified against the direct count).
    """
    c = model.crystal
    # optimal offset per reduced hkl from the cg surface energies (min-energy cut)
    off_by_hkl = {tuple(fc.reduced_hkl()): fc.offset for fc in model.cg.unique_facets()}
    out = []
    for f in poly.faces:
        d = 1.0 / occpy.Surface(occpy.HKL(*f.hkl), c).d()  # interplanar spacing (A)
        o = off_by_hkl.get(tuple(f.hkl), 0.5)
        out.append((d, o * d, None))
    return out


def cluster_snapped(model, poly, s, offs, snap=True):
    normals = np.array([f.normal for f in poly.faces])
    base = np.array([f.distance for f in poly.faces]) * s  # gamma_f * s
    if snap:
        D = np.array([od[1] + od[0] * round((base[i] - od[1]) / od[0])
                      for i, od in enumerate(offs)])
    else:
        D = base
    cells = _lattice_cells(model, poly, s)
    trans = cells @ model.cg.direct.T
    pos = (model.uc_centroids[:, None, :] + trans[None, :, :]).reshape(-1, 3)
    uc = np.repeat(np.arange(len(model.uc_centroids)), len(cells))
    mask = np.all(pos @ normals.T <= D + 1e-9, axis=1)
    pos, uc = pos[mask], uc[mask]
    fr = np.linalg.solve(model.cg.direct, pos.T).T
    keys = set(model.mol_ikeys(fr))
    total = 0.0
    for f, u in zip(fr, uc):
        for b in model.neighbor_bonds[u]:
            if int(model.mol_ikeys(np.atleast_2d(f + b.frac_disp))[0]) not in keys:
                total += b.energy
    return 0.5 * total, len(pos), normals, D


def main():
    m = MorphologyModel.from_cg_json("tmp/morph/paracetamol_I_water_cg_results.json",
                                     "tmp/morph/paracetamol_I.cif")
    poly = wulff_shape(m.cg)
    sig_an = sum((f.distance / K) * f.area for f in poly.faces) / sum(f.area for f in poly.faces)
    print(f"analytic area-weighted sigma = {sig_an:.4f} kJ/mol/A^2")
    offs = optimal_offsets(m, poly)
    print(f"{'N':>7} {'unsnapped E/A':>14} {'snapped E/A':>12} {'snap/analytic':>14}")
    for Nt in [2000, 4000, 8000, 16000, 32000]:
        s = size_for_n_molecules(m, poly, Nt)
        e0, n0, _, _ = cluster_snapped(m, poly, s, offs, snap=False)
        e1, n1, _, _ = cluster_snapped(m, poly, s, offs, snap=True)
        print(f"{n1:7d} {e0/poly.total_area(s):14.4f} {e1/poly.total_area(s):12.4f} "
              f"{(e1/poly.total_area(s))/sig_an:14.3f}")


if __name__ == "__main__":
    main()
