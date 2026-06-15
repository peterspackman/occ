"""Aspirin Form I vs II: surface-driven stability and the crossover-vs-Delta_mu analysis.

GFN2-xTB is fast but its bulk lattice-energy difference between the forms is unreliable
(~25 kJ/mol vs the experimental sub-kJ/mol).  The *surface* energies are dominated by the
nearest-neighbour shell and are far more trustworthy, so we treat the bulk Delta_mu as the
uncertain input and ask: for what Delta_mu does a size crossover appear, and where?
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy import energy_vs_size
from model import MorphologyModel
from polymorph import mu_bulk
from shape import wulff_shape

SIZES = [1000, 2000, 4000, 8000, 16000, 32000]


def load(tag):
    m = MorphologyModel.from_cg_json(
        f"tmp/morph/aspirin_{tag}_water_cg_results.json", f"tmp/morph/aspirin_{tag}.cif"
    )
    return m, wulff_shape(m.cg)


def main():
    mI, pI = load("I")
    mII, pII = load("II")
    ptsI = energy_vs_size(mI, pI, SIZES)
    ptsII = energy_vs_size(mII, pII, SIZES)

    muI, muII = mu_bulk(mI), mu_bulk(mII)
    dmu_xtb = muII - muI  # >0 if Form I is the more bulk-stable form
    print(f"lattice mu: I={muI:.2f}  II={muII:.2f}  ->  Delta_mu(xtb)={dmu_xtb:.2f} kJ/mol")

    # surface advantage of the lower-surface form: dE(N) = E_excess_I - E_excess_II
    NI = np.array([p.n_molecules for p in ptsI]); EI = np.array([p.e_exact for p in ptsI])
    NII = np.array([p.n_molecules for p in ptsII]); EII = np.array([p.e_exact for p in ptsII])
    Ngrid = np.geomspace(800, 30000, 50)
    dE = np.interp(Ngrid, NI, EI) - np.interp(Ngrid, NII, EII)  # >0 => II cheaper surface
    # fit dE ~ c * N^(2/3)
    c = np.linalg.lstsq((Ngrid ** (2 / 3))[:, None], dE, rcond=None)[0][0]
    print(f"surface advantage of Form II:  dE(N) ~ {c:.2f} * N^(2/3) kJ/mol")

    # crossover N*(Delta_mu): solve c*N^(2/3) = N*Delta_mu  ->  N* = (c/Delta_mu)^3
    dmu = np.geomspace(0.05, 30, 200)
    Ncross = (c / dmu) ** 3
    molvol = mI.cg.molecular_volume
    Rcross = (3 * Ncross * molvol / (4 * np.pi)) ** (1 / 3) / 10  # nm

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) per-molecule free energy with the xtb bulk gap (Form I dominates everywhere)
    ax[0].semilogx(NI, EI / NI + muI, "o-", label="Form I")
    ax[0].semilogx(NII, EII / NII + muII, "s-", label="Form II")
    ax[0].set_xlabel("N molecules"); ax[0].set_ylabel("G/N (kJ/mol)")
    ax[0].set_title(f"G/N with xTB bulk gap (Δμ={dmu_xtb:.0f} kJ/mol)\nForm I wins at all sizes")
    ax[0].legend()

    # (b) crossover size vs assumed bulk Delta_mu (the actionable plot)
    ax[1].loglog(dmu, Ncross, "-", color="C3")
    ax[1].axvspan(0.05, 1.0, color="green", alpha=0.12)
    ax[1].text(0.18, 3e5, "experimental\nI/II regime\n(<1 kJ/mol)", fontsize=8, color="green")
    ax[1].axvline(dmu_xtb, color="gray", ls="--")
    ax[1].text(dmu_xtb * 0.5, 1e4, f"xTB\nΔμ={dmu_xtb:.0f}", fontsize=8, color="gray", ha="right")
    for n, r in [(100, ""), (1000, ""), (10000, "")]:
        ax[1].axhline(n, color="k", lw=0.3, alpha=0.3)
    ax[1].set_xlabel("assumed bulk Δμ = μ(II) − μ(I)  (kJ/mol)")
    ax[1].set_ylabel("crossover N (Form II favoured below)")
    ax[1].set_title("crossover size vs bulk energy gap\n(surface advantage of Form II is fixed)")
    ax[1].set_ylim(10, 1e7)

    # secondary axis: particle radius in nm at the crossover
    secax = ax[1].secondary_yaxis(
        "right",
        functions=(lambda N: (3 * np.maximum(N, 1) * molvol / (4 * np.pi)) ** (1 / 3) / 10,
                   lambda R: 4 / 3 * np.pi * (R * 10) ** 3 / molvol),
    )
    secax.set_ylabel("particle radius (nm)")

    fig.tight_layout()
    out = "tmp/morph/aspirin_crossover_analysis.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print()
    for d in [0.25, 0.5, 1.0, 2.0]:
        n = (c / d) ** 3
        r = (3 * n * molvol / (4 * np.pi)) ** (1 / 3) / 10
        print(f"  if true Δμ = {d:>4} kJ/mol  ->  Form II favoured below N ~ {n:>8.0f}  (R ~ {r:.1f} nm)")


if __name__ == "__main__":
    main()
