#!/usr/bin/env python
"""Consume the C++ `occ cg --morphology` block: report shape/energies and polymorph crossover.

This is the thin Python "compare" layer: the heavy per-structure work (Wulff shape, surface
energies, registry-minimised E_excess(N), surface/edge/corner decomposition) is done in occ
(C++) and written to the `morphology` block of `<name>_<solvent>_cg_results.json`.  Here we
just read one or two of those, print the summary, and find the size crossover.

    .venv/bin/python scripts/morphology/occ_morphology.py A_..._cg_results.json [B_..._cg_results.json] [--plot out.png]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KJ_PER_MOL_TO_J_PER_M2 = 0.16604390671


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    if "morphology" not in data:
        raise SystemExit(f"{path}: no morphology block (run occ cg with --morphology)")
    m = data["morphology"]
    m["title"] = data.get("title", path)
    m["solvent"] = data.get("solvent", "")
    m["N"] = np.array([s["n_molecules"] for s in m["samples"]], dtype=float)
    m["E_excess"] = np.array([s["e_excess"] for s in m["samples"]])
    m["E_surface_analytic"] = np.array([s["e_surface_analytic"] for s in m["samples"]])
    return m


def surface_coeff(m):
    """Exact analytic surface coefficient a (E_surf = a*N^(2/3))."""
    return np.linalg.lstsq((m["N"] ** (2 / 3))[:, None], m["E_surface_analytic"],
                           rcond=None)[0][0]


def excess_interp(m):
    lN, lE = np.log(m["N"]), np.log(m["E_excess"])
    return lambda n: np.exp(np.interp(np.log(n), lN, lE))


def report(m):
    aw = sum(f["gamma"] * f["area"] for f in m["facets"]) / sum(f["area"] for f in m["facets"])
    print(f"\n{m['title']}  ({m['shape']} shape, {m['solvent']})")
    print(f"  bulk mu = {m['mu_bulk']:.2f} kJ/mol/molecule | molecular volume "
          f"{m['molecular_volume']:.1f} A^3")
    print(f"  facets {len(m['facets'])}, edges {len(m['edges'])}, corners {len(m['corners'])}")
    print(f"  area-weighted gamma = {aw:.4f} J/m^2 | surface coeff a = {surface_coeff(m):.1f} kJ/mol")
    print(f"  {'N':>7} {'E_excess':>9} {'E_surf(anal)':>12} {'E_edge+cnr':>10}")
    for s in m["samples"]:
        print(f"  {s['n_molecules']:7d} {s['e_excess']:9.0f} {s['e_surface_analytic']:12.0f} "
              f"{s['e_excess'] - s['e_surface_analytic']:10.0f}")


def crossover(a, b):
    """Crossover from the exact optimal-cut surface energy (E_surf = a*N^(2/3)).

    The analytic Wulff surface energy is the trustworthy size-dependent term (the convex
    cluster's E_excess overestimates the surface for complex shapes); it also extrapolates
    cleanly to small N where the crossover usually sits.  G/N = mu + a*N^(-1/3).
    """
    v = 0.5 * (a["molecular_volume"] + b["molecular_volume"])
    aa, ab = surface_coeff(a), surface_coeff(b)
    ng = np.geomspace(20, 5e6, 600)
    dG = (a["mu_bulk"] - b["mu_bulk"]) + (aa - ab) * ng ** (-1 / 3)  # G_a/N - G_b/N
    small = a["title"] if dG[0] < 0 else b["title"]
    large = a["title"] if dG[-1] < 0 else b["title"]
    s = np.where(np.diff(np.sign(dG)))[0]
    if not len(s):
        return None, small, large, v
    i = s[0]
    ncr = ng[i] - dG[i] * (ng[i + 1] - ng[i]) / (dG[i + 1] - dG[i])
    return float(ncr), small, large, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="one or two *_cg_results.json with a morphology block")
    ap.add_argument("--plot")
    args = ap.parse_args()
    ms = [load(j) for j in args.json]
    for m in ms:
        report(m)
    if len(ms) == 2:
        ncr, small, large, v = crossover(*ms)
        print(f"\nPolymorph stability ({ms[0]['solvent']}):")
        if ncr is None:
            print(f"  no crossover - '{large}' more stable at all sampled sizes")
        else:
            r = (3 * ncr * v / (4 * np.pi)) ** (1 / 3) / 10
            print(f"  crossover at N ~ {ncr:.0f} molecules (R ~ {r:.1f} nm): "
                  f"'{small}' favoured below, '{large}' above")
        if args.plot:
            _plot(ms, args.plot, ncr)


def _plot(ms, path, ncr=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # lines: analytic G/N = mu + a*N^(-1/3) (what sets the crossover), extended
    # below the sampled sizes; markers: the exact broken-bond samples
    lo = min(m["N"].min() for m in ms) / 4
    if ncr:
        lo = min(lo, ncr / 4)
    ng = np.geomspace(max(lo, 20), max(m["N"].max() for m in ms), 400)
    for m in ms:
        a = surface_coeff(m)
        (line,) = ax.semilogx(ng, m["mu_bulk"] + a * ng ** (-1 / 3), "-", label=m["title"])
        ax.semilogx(m["N"], m["mu_bulk"] + m["E_excess"] / m["N"], "o", ms=4,
                    mfc="none", color=line.get_color())
    if ncr:
        ax.axvline(ncr, color="0.6", ls=":", lw=1)
        ax.text(ncr, ax.get_ylim()[1], f" N ~ {ncr:.0f}", va="top", fontsize=9, color="0.4")
    ax.set_xlabel("N molecules")
    ax.set_ylabel("G / N (kJ/mol)")
    ax.set_title("Free energy per molecule (from occ --morphology)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
