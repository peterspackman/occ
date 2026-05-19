#!/usr/bin/env python3
"""Parallel to examples/lua/cube.lua: generate Gaussian-cube files
(electron density + electrostatic potential) from a Hartree–Fock
wavefunction. Pair with the isosurface example to render surfaces
colored by ESP in your favorite viewer.

Run:
    python examples/python/cube.py [molecule.xyz]
"""

import sys
import occpy


def main():
    xyz_path = sys.argv[1] if len(sys.argv) > 1 else "examples/scf/water.xyz"
    n = 60

    print(f"Loading: {xyz_path}")
    mol = occpy.Molecule.from_xyz_file(xyz_path)

    print("Running HF/STO-3G ...")
    basis = occpy.AOBasis.load(mol.atoms(), "sto-3g")
    hf = occpy.HartreeFock(basis)
    scf = hf.scf()
    scf.set_charge_multiplicity(0, 1)
    energy = scf.run()
    print(f"SCF energy = {energy:.10f} Ha")

    wfn = scf.wavefunction()

    print(f"\nGenerating {n}x{n}x{n} cubes ...")

    density_cube = occpy.generate_electron_density_cube(wfn, n, n, n)
    with open("density.cube", "w") as f:
        f.write(density_cube)
    print(f"  density.cube       ({len(density_cube)} bytes)")

    esp_cube = occpy.generate_esp_cube(wfn, n, n, n)
    with open("esp.cube", "w") as f:
        f.write(esp_cube)
    print(f"  esp.cube           ({len(esp_cube)} bytes)")

    # Full control via VolumeCalculator + VolumeGenerationParameters.
    calc = occpy.VolumeCalculator()
    calc.set_wavefunction(wfn)

    params = occpy.VolumeGenerationParameters()
    params.property = occpy.VolumePropertyKind.ElectronDensity
    params.adaptive_bounds = True

    vol = calc.compute_volume(params)
    print(f"\nVolumeData: {vol.nx()}x{vol.ny()}x{vol.nz()} = {vol.total_points()} total points")

    cube = calc.volume_as_cube_string(vol)
    with open("density_adaptive.cube", "w") as f:
        f.write(cube)
    print(f"  density_adaptive.cube ({len(cube)} bytes)")


if __name__ == "__main__":
    main()
