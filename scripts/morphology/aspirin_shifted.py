"""Aspirin I/II: does a size crossover exist, and what would it take?

Uses the EXACT analytic surface energy (E_surf = a * N^(2/3), a = area-weighted gamma x Wulff
shape factor) for each form - clean, no cluster noise - plus an imposed bulk gap Delta_mu
(xTB's own bulk gap is unreliable for near-degenerate polymorphs).

Result: Form I has the lower surface energy AND the lower bulk energy, so it wins at every
size.  A crossover requires the orderings to be OPPOSITE (the metastable form having the
lower surface energy); that is the condition to look for in a candidate system.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cg_data import KJ_PER_MOL_TO_J_PER_M2 as K
from model import MorphologyModel
from polymorph import mu_bulk
from shape import wulff_shape


def surface_coeff(tag):
    """Exact surface coefficient a (E_surf = a*N^(2/3), kJ/mol) and bulk mu (kJ/mol)."""
    m = MorphologyModel.from_cg_json(
        f"tmp/morph/aspirin_{tag}_water_cg_results.json", f"tmp/morph/aspirin_{tag}.cif")
    p = wulff_shape(m.cg)
    aw = sum(f.distance * f.area for f in p.faces) / sum(f.area for f in p.faces)  # J/m^2
    A = sum(f.area for f in p.faces)
    shapefac = A / p.volume ** (2 / 3)
    v = m.cg.molecular_volume
    a = (aw / K) * shapefac * v ** (2 / 3)
    return a, mu_bulk(m), v, aw


def main():
    aI, muI, vI, gI = surface_coeff("I")
    aII, muII, vII, gII = surface_coeff("II")
    v = 0.5 * (vI + vII)
    print(f"Form I : gamma={gI:.4f} J/m^2  surface a={aI:.1f} kJ/mol  mu(xtb)={muI:.1f}")
    print(f"Form II: gamma={gII:.4f} J/m^2  surface a={aII:.1f} kJ/mol  mu(xtb)={muII:.1f}")
    print(f"Form I has the LOWER surface energy (Δa = {aII-aI:+.1f} kJ/mol) -> "
          f"no size can stabilise Form II while Form I is also bulk-stable.\n")

    Ng = np.geomspace(50, 3e5, 400)
    R = (3 * Ng * v / (4 * np.pi)) ** (1 / 3) / 10  # nm

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # dG/N = (mu_II - mu_I) + (a_II - a_I) N^(-1/3); sweep the imposed bulk gap (both signs)
    da = aII - aI
    for dmu, c in [(2.0, "C3"), (1.0, "C1"), (0.0, "k"), (-1.0, "C0"), (-2.0, "C2")]:
        dG = dmu + da * Ng ** (-1 / 3)
        ax[0].plot(Ng, dG, color=c, label=f"Δμ={dmu:+.0f}")
        s = np.where(np.diff(np.sign(dG)))[0]
        if len(s):
            i = s[0]; ncr = Ng[i] - dG[i] * (Ng[i+1]-Ng[i])/(dG[i+1]-dG[i])
            ax[0].plot([ncr], [0], "o", color=c, ms=6)
    ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("N molecules")
    ax[0].set_ylabel("G/N(II) − G/N(I)  (kJ/mol)")
    ax[0].set_title("Aspirin I/II: imposed bulk gap Δμ = μ(II)−μ(I)\n"
                    "crossover only when Δμ<0 (Form II bulk-stable) — not the real ordering")
    ax[0].legend(fontsize=8, title="bulk gap")
    ax[0].text(60, ax[0].get_ylim()[1]*0.85, "Form I favoured", color="gray", fontsize=8)
    ax[0].text(60, ax[0].get_ylim()[0]*0.85, "Form II favoured", color="gray", fontsize=8)

    # (b) what a real crossover needs: metastable form must have lower surface energy.
    # illustrate a hypothetical "good" pair: stable form mu=0, higher surface; metastable
    # mu=+1 kJ/mol, lower surface (a_meta < a_stable by various amounts)
    a_stable = 100.0
    for da_adv, c in [(8, "C0"), (15, "C1"), (25, "C2")]:
        a_meta = a_stable - da_adv  # metastable has LOWER surface
        dmu = 1.0  # metastable less bulk-stable by 1 kJ/mol
        # dG = G_meta - G_stable = dmu + (a_meta - a_stable) N^(-1/3)
        dG = dmu + (a_meta - a_stable) * Ng ** (-1 / 3)
        ax[1].plot(Ng, dG, color=c, label=f"Δγ-surface = {da_adv} kJ/mol")
        s = np.where(np.diff(np.sign(dG)))[0]
        if len(s):
            i = s[0]; ncr = Ng[i] - dG[i]*(Ng[i+1]-Ng[i])/(dG[i+1]-dG[i])
            rcr = (3*ncr*v/(4*np.pi))**(1/3)/10
            ax[1].plot([ncr], [0], "o", color=c, ms=6)
            ax[1].annotate(f"{rcr:.1f} nm", (ncr, 0), textcoords="offset points",
                           xytext=(4, 6), fontsize=8, color=c)
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("N molecules")
    ax[1].set_ylabel("G/N(metastable) − G/N(stable)  (kJ/mol)")
    ax[1].set_title("What a findable crossover needs (Δμ=1 kJ/mol):\n"
                    "metastable form with LOWER surface energy")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = "tmp/morph/aspirin_shifted_crossover.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
