"""Generate illustrative plots for the morphology prototype."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cg_data import KJ_PER_MOL_TO_J_PER_M2 as K
from energy import energy_vs_size, size_for_n_molecules
from model import MorphologyModel
from polymorph import crossover_size, g_of_size
from shape import wulff_shape

AA = ("tmp/morph/acetic_acid_water_cg_results.json", "tmp/morph/acetic_acid.cif")
UR = ("tmp/morph/urea_water_cg_results.json", "tmp/morph/urea.cif")
SIZES = [1000, 2000, 4000, 8000, 16000, 32000]


def load(j, c):
    m = MorphologyModel.from_cg_json(j, c)
    return m, wulff_shape(m.cg)


def wulff_poly3d(ax, poly, title):
    polys = [f.vertices for f in poly.faces]
    gam = np.array([f.distance for f in poly.faces])
    norm = (gam - gam.min()) / (np.ptp(gam) + 1e-9)
    colors = plt.cm.viridis(norm)
    pc = Poly3DCollection(polys, facecolors=colors, edgecolors="k", linewidths=0.4, alpha=0.95)
    ax.add_collection3d(pc)
    v = poly.vertices
    r = np.abs(v).max()
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    ax.set_axis_off()


def main():
    m_aa, p_aa = load(*AA)
    m_ur, p_ur = load(*UR)
    pts_aa = energy_vs_size(m_aa, p_aa, SIZES)
    pts_ur = energy_vs_size(m_ur, p_ur, SIZES)

    fig = plt.figure(figsize=(15, 9))

    # (a) excess energy vs N, total vs analytic surface
    ax = fig.add_subplot(2, 3, 1)
    N = np.array([p.n_molecules for p in pts_aa])
    ax.loglog(N, [p.e_exact for p in pts_aa], "o-", label="E_excess (exact)")
    ax.loglog(N, [p.e_surface_analytic for p in pts_aa], "s--", label="surface (Σγ·A)")
    ax.loglog(N, [p.e_exact - p.e_surface_analytic for p in pts_aa], "^:", label="edge+corner")
    ax.loglog(N, 2.0 * N ** (2 / 3), "k-", lw=0.6, alpha=0.5, label="∝ N^2/3")
    ax.set_xlabel("N molecules"); ax.set_ylabel("energy (kJ/mol)")
    ax.set_title("acetic acid: excess energy vs size"); ax.legend(fontsize=8)

    # (b) effective surface density vs N (convergence to analytic γ)
    ax = fig.add_subplot(2, 3, 2)
    for pts, p, lab in [(pts_aa, p_aa, "acetic acid"), (pts_ur, p_ur, "urea")]:
        Ns = [x.n_molecules for x in pts]
        sig = [x.e_exact / x.area * K for x in pts]  # J/m^2
        ax.semilogx(Ns, sig, "o-", label=lab)
        an = sum(f.distance * f.area for f in p.faces) / sum(f.area for f in p.faces)
        ax.axhline(an, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("N molecules"); ax.set_ylabel("E_excess / area  (J/m²)")
    ax.set_title("effective surface energy density\n(dashed = analytic optimal-cut)")
    ax.legend(fontsize=8)

    # (c) per-facet surface energies (Wulff area-weighted)
    ax = fig.add_subplot(2, 3, 3)
    seen = {}
    atot = sum(f.area for f in p_aa.faces)
    for f in p_aa.faces:
        key = round(f.distance, 5)
        seen.setdefault(key, [str(f.hkl), 0.0])
        seen[key][1] += f.area / atot
    items = sorted(seen.items())
    ax.bar(range(len(items)), [g for g, _ in items], color=plt.cm.viridis(
        np.linspace(0, 1, len(items))))
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([v[0] for _, v in items], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("γ (J/m²)"); ax.set_title("acetic acid: per-facet surface energy")

    # (d) G/N vs N for both
    ax = fig.add_subplot(2, 3, 4)
    ca = g_of_size(m_aa, p_aa, SIZES, label="acetic acid")
    cu = g_of_size(m_ur, p_ur, SIZES, label="urea")
    ax.semilogx(ca.n, ca.g_per_molecule, "o-", label="acetic acid")
    ax.semilogx(cu.n, cu.g_per_molecule, "s-", label="urea")
    ax.set_xlabel("N molecules"); ax.set_ylabel("G / N (kJ/mol)")
    ax.set_title("free energy per molecule"); ax.legend(fontsize=8)

    # (e,f) Wulff shapes
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    wulff_poly3d(ax, p_aa, "acetic acid Wulff shape")
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    wulff_poly3d(ax, p_ur, "urea Wulff shape")

    fig.tight_layout()
    out = "tmp/morph/morphology_overview.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
