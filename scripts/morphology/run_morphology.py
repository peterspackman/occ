#!/usr/bin/env python
"""Particle size/shape-dependent energy from occ cg results.

Usage (run with the project venv, which has occpy + numpy/scipy):

    .venv/bin/python scripts/morphology/run_morphology.py A_water_cg_results.json [B_..json] \
        [--cif a.cif,b.cif] [--mu crystal|solution] [--plot out.png]

Given one cg results JSON it reports the Wulff shape, per-facet surface energies, and the
size-dependent excess (surface + edge + corner) energy.  Given two it also reports the
polymorph stability crossover size in the chosen solvent.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cg_data import CGData, KJ_PER_MOL_TO_J_PER_M2
from energy import energy_vs_size
from model import MorphologyModel
from polymorph import crossover_size, g_of_size, mu_bulk
from shape import wulff_shape


def infer_cif(json_path: str) -> str:
    cg = CGData.from_json(json_path)
    return os.path.join(os.path.dirname(os.path.abspath(json_path)), f"{cg.title}.cif")


def report_shape(model, poly):
    print(f"\nWulff shape: {len(poly.faces)} faces, {len(poly.edges)} edges, "
          f"{len(poly.corners)} corners  (V-E+F = "
          f"{len(poly.corners) - len(poly.edges) + len(poly.faces)})")
    print(f"{'hkl':>12} {'gamma (J/m^2)':>14} {'area fraction':>14}")
    atot = sum(f.area for f in poly.faces)
    # one row per symmetry-distinct facet energy
    seen = {}
    for f in poly.faces:
        seen.setdefault(round(f.distance, 5), [f.hkl, 0.0])
        seen[round(f.distance, 5)][1] += f.area / atot
    for gamma, (hkl, frac) in sorted(seen.items()):
        print(f"{str(hkl):>12} {gamma:14.4f} {frac:14.3f}")


def report_size_dependence(model, poly, n_targets, n_registry):
    print(f"\nSize-dependent excess energy ({model.cg.solvent}):")
    print(f"{'N':>7} {'R_eq(A)':>8} {'E_excess':>10} {'E_surf':>9} {'E_edge+cnr':>10} "
          f"{'(kJ/mol)':>9}")
    pts = energy_vs_size(model, poly, n_targets, n_registry=n_registry)
    for p in pts:
        r_eq = (3 * poly.scaled_volume(p.s) / (4 * np.pi)) ** (1 / 3)
        resid = p.e_exact - p.e_surface_analytic
        print(f"{p.n_molecules:7d} {r_eq:8.1f} {p.e_exact:10.0f} "
              f"{p.e_surface_analytic:9.0f} {resid:10.0f}")
    # effective surface energy density and edge line tension (residual fit)
    area = np.array([p.area for p in pts])
    edge = np.array([p.edge_length for p in pts])
    e = np.array([p.e_exact for p in pts])
    # surface density from the large-size limit of E/area
    sigma_eff = e[-1] / area[-1]
    resid = e - np.array([p.e_surface_analytic for p in pts])
    lam = np.linalg.lstsq(edge[:, None], resid, rcond=None)[0][0]
    print(f"  effective surface density ~ {sigma_eff:.3f} kJ/mol/A^2 "
          f"({sigma_eff * KJ_PER_MOL_TO_J_PER_M2:.3f} J/m^2)")
    print(f"  effective edge line tension ~ {lam:.3f} kJ/mol/A")
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="one or two *_cg_results.json files")
    ap.add_argument("--cif", help="comma-separated CIF path(s); inferred from title if omitted")
    ap.add_argument("--cg-radius", type=float, default=3.8)
    ap.add_argument("--mu", choices=["crystal", "solution"], default="crystal",
                    help="bulk reference for polymorph comparison")
    ap.add_argument("--registry", type=int, default=3,
                    help="sub-cell registry grid (NxNxN) for optimal-cut minimisation")
    ap.add_argument("--sizes", default="1000,2000,4000,8000,16000,32000",
                    help="comma-separated target molecule counts")
    ap.add_argument("--plot", help="write a G(N) / dG(N) plot to this PNG")
    args = ap.parse_args()

    if len(args.json) > 2:
        ap.error("at most two structures (polymorphs) are supported")
    cifs = args.cif.split(",") if args.cif else [infer_cif(j) for j in args.json]
    n_targets = [int(x) for x in args.sizes.split(",")]

    curves = []
    models = []
    for jpath, cif in zip(args.json, cifs):
        print(f"\n{'=' * 70}\n{jpath}\n{'=' * 70}")
        model = MorphologyModel.from_cg_json(jpath, cif, cg_radius=args.cg_radius)
        poly = wulff_shape(model.cg)
        print(f"bulk mu ({args.mu}) = {mu_bulk(model, args.mu):.2f} kJ/mol/molecule | "
              f"molecular volume {model.cg.molecular_volume:.1f} A^3")
        report_shape(model, poly)
        report_size_dependence(model, poly, n_targets, args.registry)
        curves.append(g_of_size(model, poly, n_targets, args.mu, args.registry,
                                label=model.cg.title))
        models.append(model)

    if len(curves) == 2:
        n_cross, small, large = crossover_size(*curves)
        print(f"\n{'=' * 70}\nPolymorph stability ({models[0].cg.solvent}):")
        if n_cross is None:
            print(f"  no crossover - '{large}' is more stable at all sampled sizes")
        else:
            print(f"  crossover at N ~ {n_cross:.0f} molecules")
            print(f"  '{small}' favoured below, '{large}' favoured above")

    if args.plot:
        _plot(curves, args.plot)


def _plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2 if len(curves) == 2 else 1, figsize=(11, 4.5), squeeze=False)
    for c in curves:
        ax[0][0].plot(c.n, c.g_per_molecule, "o-", label=c.label)
    ax[0][0].set_xlabel("N molecules")
    ax[0][0].set_ylabel("G / N (kJ/mol)")
    ax[0][0].set_xscale("log")
    ax[0][0].legend()
    ax[0][0].set_title("Free energy per molecule")
    if len(curves) == 2:
        ng = np.geomspace(max(curves[0].n.min(), curves[1].n.min()),
                          min(curves[0].n.max(), curves[1].n.max()), 200)
        da = np.interp(ng, curves[0].n, curves[0].g_per_molecule)
        db = np.interp(ng, curves[1].n, curves[1].g_per_molecule)
        ax[0][1].axhline(0, color="k", lw=0.5)
        ax[0][1].plot(ng, da - db)
        ax[0][1].set_xlabel("N molecules")
        ax[0][1].set_ylabel(f"G/N({curves[0].label}) - G/N({curves[1].label})")
        ax[0][1].set_xscale("log")
        ax[0][1].set_title("Relative stability")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nwrote plot to {path}")


if __name__ == "__main__":
    main()
