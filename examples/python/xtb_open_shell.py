#!/usr/bin/env python3
"""Spin-unrestricted GFN2-xTB on a radical.

Setting a multiplicity above 1 — either on the Molecule or via
``num_unpaired_electrons`` — switches the SCC to separate alpha/beta densities
coupled by the on-site spin-polarization term.

Run:
    python examples/python/xtb_open_shell.py
"""

import numpy as np
import occpy

BOHR = 1.8897261246257702


def methyl_radical():
    r = 1.078 * BOHR
    c = 0.8660254037844386 * r
    positions = np.array(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0], [-0.5 * r, c, 0.0], [-0.5 * r, -c, 0.0]]
    ).T
    mol = occpy.Molecule(np.array([6, 1, 1, 1]), positions / BOHR)
    mol.multiplicity = 2  # doublet
    return mol


def main():
    mol = methyl_radical()

    calc = occpy.XtbCalculator(mol)
    print(f"unpaired electrons: {calc.num_unpaired_electrons}")

    result = calc.single_point()
    assert result.converged, "SCC did not converge"

    print(f"\nTotal energy       = {result.total_energy:.10f} Ha")
    print(f"  spin polarization = {result.spin_energy:.10f} Ha")

    print("\n  atom      charge        spin")
    for i, (q, s) in enumerate(
        zip(result.atomic_charges, result.atomic_magnetization)
    ):
        print(f"  {i + 1:>4d}  {q:+10.6f}  {s:+10.6f}")
    print(f"  total magnetization: {result.atomic_magnetization.sum():.6f}")

    # Frontier orbitals differ between the two channels.
    for label, energies, occupations in (
        ("alpha", result.orbital_energies, result.orbital_occupations),
        ("beta", result.orbital_energies_beta, result.orbital_occupations_beta),
    ):
        n_occ = int((occupations > 1e-6).sum())
        homo, lumo = energies[n_occ - 1], energies[n_occ]
        print(f"  {label:>5s}  HOMO {homo:+.4f}  LUMO {lumo:+.4f}  gap {lumo - homo:.4f} Ha")

    # Setting the scale to zero drops the spin coupling, leaving alpha and beta
    # sharing one Hamiltonian — what plain `xtb --uhf` computes.
    unpolarized = occpy.XtbCalculator(mol)
    unpolarized.spin_polarization = 0.0
    print(
        f"\nwithout spin polarization: {unpolarized.single_point_energy():.10f} Ha"
    )

    # The wavefunction carries both spin channels through to the rest of occ
    # (.owf.json round-trip, cube / isosurface, population analysis).
    wfn = calc.to_wavefunction()
    mo = wfn.molecular_orbitals
    print(f"\nwavefunction: {mo.kind}, "
          f"n_alpha={mo.num_alpha} n_beta={mo.num_beta}, "
          f"multiplicity={wfn.multiplicity()}")


if __name__ == "__main__":
    main()
