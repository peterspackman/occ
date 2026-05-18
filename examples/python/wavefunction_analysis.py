#!/usr/bin/env python3
"""Run an HF/STO-3G calculation, then analyze the wavefunction.

Demonstrates accessing molecular orbitals, computing Mulliken and CHELPG
charges, evaluating the electron density on a small grid, and saving the
wavefunction in JSON / FCHK formats.

Run:
    python examples/python/wavefunction_analysis.py [water.xyz]
"""

import sys
import numpy as np
import occpy


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/scf/water.xyz"
    mol = occpy.Molecule.from_xyz_file(path)
    print(f"Loaded {path}: {mol}\n")

    basis = occpy.AOBasis.load(mol.atoms(), "sto-3g")
    print(basis)

    hf = occpy.HartreeFock(basis)
    scf = hf.scf()
    scf.set_charge_multiplicity(0, 1)
    energy = scf.run()
    print(f"SCF energy = {energy:.10f} Ha\n")

    wfn = scf.wavefunction()

    # Orbital energies and MO occupations
    energies = wfn.molecular_orbitals.orbital_energies
    print("Orbital energies (Hartree):")
    for i, e in enumerate(energies):
        print(f"  φ_{i + 1:2d}: {e:+.6f}")

    # Mulliken (population-based) and CHELPG (potential-fitted) charges.
    print(f"\nMulliken charges: {np.round(wfn.mulliken_charges(), 4)}")
    print(f"CHELPG   charges: {np.round(wfn.chelpg_charges(), 4)}")

    # Electron density on a few sample points (1-electron probe positions).
    pts = np.array([[0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])  # 3×3 (x, y, z) for 3 points
    density = wfn.electron_density(pts)
    print(f"\nDensity sample (3 points): {np.round(density, 6)}")

    # Persist the wavefunction — JSON is portable; FCHK is for Gaussian compat.
    out_base = "/tmp/wfn_analysis"
    wfn.save(f"{out_base}.owf.json")
    wfn.to_fchk(f"{out_base}.fchk")
    print(f"\nWrote {out_base}.owf.json and {out_base}.fchk")


if __name__ == "__main__":
    main()
