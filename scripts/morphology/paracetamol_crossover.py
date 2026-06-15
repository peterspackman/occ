"""Paracetamol Form I / II: a genuine size-dependent polymorph crossover.

Unlike aspirin I/II (same layers, one form won both bulk and surface), paracetamol's forms
have OPPOSITE orderings: Form I (corrugated H-bonded layers) has the lower surface energy,
Form II (flat parallel layers) has the lower bulk energy (per xTB; the real gap is ~1 kJ/mol
and its sign is debated).  Lower surface -> favoured small; lower bulk -> favoured large ->
crossover.

Surface energies (NN-dominated) are taken from the exact analytic Wulff sum E_surf = a*N^(2/3);
the bulk gap Delta_mu is swept because that is the uncertain input that sets where (and whether)
the crossover lands.
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


def form(tag):
    m = MorphologyModel.from_cg_json(
        f"tmp/morph/paracetamol_{tag}_water_cg_results.json", f"tmp/morph/paracetamol_{tag}.cif")
    p = wulff_shape(m.cg)
    aw = sum(f.distance * f.area for f in p.faces) / sum(f.area for f in p.faces)
    a = (aw / K) * (sum(f.area for f in p.faces) / p.volume ** (2 / 3)) * m.cg.molecular_volume ** (2 / 3)
    return dict(a=a, gamma=aw, mu=mu_bulk(m), v=m.cg.molecular_volume)


def main():
    I, II = form("I"), form("II")
    v = 0.5 * (I["v"] + II["v"])
    dmu_xtb = II["mu"] - I["mu"]  # <0: Form II more bulk-stable (xtb)
    da = II["a"] - I["a"]  # >0: Form I lower surface energy
    print(f"Form I : gamma={I['gamma']:.4f} J/m^2  a={I['a']:.1f}  mu={I['mu']:.1f}")
    print(f"Form II: gamma={II['gamma']:.4f} J/m^2  a={II['a']:.1f}  mu={II['mu']:.1f}")
    print(f"xtb Delta_mu(II-I) = {dmu_xtb:+.2f} kJ/mol ; surface Delta_a = {da:+.1f}")

    Ng = np.geomspace(50, 1e6, 500)
    Rnm = (3 * Ng * v / (4 * np.pi)) ** (1 / 3) / 10

    def crossover(dmu):
        dG = dmu * Ng + da * Ng ** (2 / 3)  # G_II - G_I ; surface part = (a_II-a_I)N^2/3
        s = np.where(np.diff(np.sign(dG)))[0]
        if not len(s):
            return None
        i = s[0]
        return Ng[i] - dG[i] * (Ng[i + 1] - Ng[i]) / (dG[i + 1] - dG[i])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) per-molecule free energy with xtb bulk gap -> the actual crossover
    gI = I["mu"] + I["a"] * Ng ** (-1 / 3)
    gII = II["mu"] + II["a"] * Ng ** (-1 / 3)
    ax[0].semilogx(Ng, gI, label="Form I (lower surface)")
    ax[0].semilogx(Ng, gII, label="Form II (lower bulk, xTB)")
    ncr = crossover(dmu_xtb)
    if ncr:
        rcr = (3 * ncr * v / (4 * np.pi)) ** (1 / 3) / 10
        ax[0].axvline(ncr, color="k", ls=":", lw=0.8)
        ax[0].annotate(f"crossover\nN≈{ncr:.0f}  ({rcr:.1f} nm)", (ncr, np.interp(ncr, Ng, gI)),
                       textcoords="offset points", xytext=(8, 20), fontsize=8)
    ax[0].set_xlabel("N molecules"); ax[0].set_ylabel("G/N (kJ/mol)")
    ax[0].set_title(f"Paracetamol I/II with xTB bulk gap (Δμ={dmu_xtb:.1f} kJ/mol)\n"
                    "Form I small particles → Form II large particles")
    ax[0].legend(fontsize=8)
    ax[0].text(70, gI.min(), "Form I favoured", color="gray", fontsize=8)

    # (b) crossover size vs the (uncertain) bulk gap; mark xtb and the sign-flip
    dmus = np.linspace(-6, 6, 400)
    Ncr = []
    for d in dmus:
        n = crossover(d)
        Ncr.append(n if n else np.nan)
    Ncr = np.array(Ncr)
    ax[1].plot(dmus, Ncr, color="C3")
    ax[1].axvline(dmu_xtb, color="gray", ls="--")
    ax[1].text(dmu_xtb, 5e5, f"xTB\nΔμ={dmu_xtb:.1f}", fontsize=8, color="gray", ha="center")
    ax[1].axvspan(0, 6, color="red", alpha=0.07)
    ax[1].text(3, 3e5, "Δμ>0: Form I bulk-stable\n→ no crossover\n(Form I wins all sizes)",
               fontsize=8, color="darkred", ha="center")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("bulk gap Δμ = μ(II) − μ(I)  (kJ/mol)")
    ax[1].set_ylabel("crossover N (Form I favoured below)")
    ax[1].set_title("Where the crossover lands vs the bulk gap\n"
                    "(the input xTB can't be trusted on — needs ce-b3lyp)")
    secax = ax[1].secondary_yaxis(
        "right",
        functions=(lambda N: (3 * np.maximum(N, 1) * v / (4 * np.pi)) ** (1 / 3) / 10,
                   lambda R: 4 / 3 * np.pi * (R * 10) ** 3 / v))
    secax.set_ylabel("radius (nm)")
    ax[1].set_ylim(50, 1e6)

    fig.tight_layout()
    out = "tmp/morph/paracetamol_crossover.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    for d in [-3.3, -2, -1, -0.5]:
        n = crossover(d)
        if n:
            r = (3 * n * v / (4 * np.pi)) ** (1 / 3) / 10
            print(f"  Δμ(II-I)={d:>5} kJ/mol -> Form I below N≈{n:>8.0f} (R≈{r:.1f} nm), Form II above")


if __name__ == "__main__":
    main()
