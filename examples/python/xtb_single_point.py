#!/usr/bin/env python3
"""GFN2-xTB single-point energy + gradient on a molecule, with property output.

Run:
    python examples/python/xtb_single_point.py [water.xyz]
"""

import sys
import occpy


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/scf/water.xyz"
    mol = occpy.Molecule.from_xyz_file(path)
    print(f"Loaded {path}: {mol}")

    calc = occpy.XtbCalculator(mol)
    calc.charge = 0
    calc.include_multipoles = True

    result = calc.single_point()
    if not result.converged:
        print("SCC did not converge!")
        sys.exit(1)

    print(f"\nConverged in {result.n_iterations} iterations.")
    print(f"  Total      = {result.total_energy:.10f} Ha")
    print(f"  SCC        = {result.scc_energy:.10f} Ha")
    print(f"  Repulsion  = {result.repulsion_energy:.10f} Ha")
    print(f"  Dispersion = {result.dispersion_energy:.10f} Ha")

    print("\nAtomic charges (Mulliken):")
    for i, q in enumerate(result.atomic_charges):
        print(f"  atom {i + 1}: {q:+.4f}")

    # Analytical gradient (Hartree/Bohr).
    energy, gradient = calc.energy_and_gradient()
    print(f"\nGradient (Hartree/Bohr) at E = {energy:.10f} Ha:")
    # gradient is a 3×N numpy array.
    for j in range(gradient.shape[1]):
        print(f"  atom {j + 1}: {gradient[0, j]:+ .6f} {gradient[1, j]:+ .6f} {gradient[2, j]:+ .6f}")


if __name__ == "__main__":
    main()
