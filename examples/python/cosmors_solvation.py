#!/usr/bin/env python3
"""openCOSMO-RS 24a solvation free energy, term by term.

The model needs two SCFs — one in the gas phase, one in an ideal conductor —
and then contracts the solute's screening-charge segments against a segment
ensemble for the solvent.

occ ships no solvent ensembles, so this computes the solvent's too, from its
geometry. That is one extra pair of SCFs. If you have a cached ensemble on
the search path ($OCC_DATA_PATH/solvent/cosmors, or the working directory),
pass its name instead and the solvent SCFs are skipped.

Run:
    python examples/python/cosmors_solvation.py [solute.xyz] [solvent]

where [solvent] is either a .xyz geometry or the name of a cached ensemble.
"""

import sys

import occpy

# Hartree to kJ/mol.
AU_TO_KJ = 2625.499639479


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/scf/water.xyz"
    solvent = sys.argv[2] if len(sys.argv) > 2 else "examples/scf/water.xyz"

    mol = occpy.Molecule.from_xyz_file(path)
    print(f"Loaded {path}: {mol}")

    # A cached ensemble is used if the name resolves; otherwise the solvent is
    # treated as a geometry and its cavity computed alongside the solute's.
    cached = occpy.available_cosmo_rs_solvents()
    use_cached = solvent in cached
    if use_cached:
        print(f"Using the cached ensemble for '{solvent}'")
    else:
        solvent_mol = occpy.Molecule.from_xyz_file(solvent)
        print(f"Computing the solvent cavity from {solvent}: {solvent_mol}")

    settings = occpy.CosmoRSSettings()
    settings.basis = "6-31g**"
    settings.temperature = 298.15
    # Liquid-phase volume per solute molecule, Angstrom^3. Leaving it at zero
    # drops the reference-state term, so the total is no longer on an
    # absolute scale.
    settings.liquid_volume = 30.01

    result = occpy.cosmo_rs_solvation_free_energy(
        mol, solvent if use_cached else solvent_mol, settings)
    e = result.energy

    print(f"\ncavity: {result.cavity_area:.2f} A^2, {result.cavity_volume:.2f} A^3")
    print(f"rings:  {result.num_rings}")
    label = solvent if use_cached else f"the geometry in {solvent}"
    print(f"\nSolvation free energy in {label} (kJ/mol):")
    for label, value, formula in [
        ("dielectric", e.dielectric, "gas -> ideal conductor"),
        ("residual", e.residual, "RT ln(gamma_res)"),
        ("combinatorial", e.combinatorial, "RT ln(gamma_comb)"),
        ("van der Waals", e.vdw, "-sum_a tau_a A_a"),
        ("ring", e.ring, "-omega_ring n_ring"),
        ("reference state", e.reference_state, "-RT ln(v_gas/v_liquid)"),
        ("eta", e.eta, "fitted intercept"),
    ]:
        print(f"  {label:<16} {value * AU_TO_KJ:9.3f}   {formula}")
    print(f"  {'total':<16} {e.total() * AU_TO_KJ:9.3f}")


if __name__ == "__main__":
    main()
