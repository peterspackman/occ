#!/usr/bin/env python3
"""Nudged elastic band from HCN to HNC with GFN2-xTB: a climbing image
finds the barrier, then a saddle point search from it along the band's
tangent refines the transition state.

The command line equivalent:
    occ scf hcn.xyz gfn2 --driver neb --product hnc.xyz --neb-ts

Run:
    python examples/python/neb.py
"""

import numpy as np
import occpy


def gfn2(molecule):
    return occpy.XtbCalculator(molecule).energy_and_gradient()


# GFN2-xTB minima, the same atoms in the same order (Angstrom, 3 x N). Not
# exactly linear: on a straight line the hydrogen could not leave the axis.
numbers = np.array([6, 7, 1])
hcn = occpy.Molecule(numbers, np.array([[0.0036, 1.1412, -1.0548],
                                        [0.0069, -0.0032, 0.0162],
                                        [0.0, 0.0, 0.0]]))
hnc = occpy.Molecule(numbers, np.array([[0.0086, 1.1670, 2.1645],
                                        [-0.0033, 0.0072, 0.0160],
                                        [0.0, 0.0, 0.0]]))

settings = occpy.opt.NebSettings()
settings.images = 8
# A rough band is enough when the saddle point search refines it (as --neb-ts)
settings.force_max = 2e-2
settings.climbing_force_max = 2e-3
band = occpy.opt.NudgedElasticBand(hcn, hnc, settings)
band.set_endpoint_energies(gfn2(hcn)[0], gfn2(hnc)[0])

for iteration in range(200):
    results = [gfn2(image) for image in band.images()]
    band.update([e for e, _ in results], [g for _, g in results])
    if band.step():
        break

energies = band.energies()
top = band.highest_image()
print(f"Band converged after {iteration} iterations; profile (kJ/mol):")
for k, e in enumerate(energies):
    print(f"  {k:2d} {(e - energies[0]) * 2625.4996:9.2f}" + ("  <- climbing" if k == top else ""))

# Refine the climbing image to the saddle point, starting along the band
guess = band.path()[top]
optimizer = occpy.opt.GeometryOptimizer(
    guess, occpy.opt.ConvergenceCriteria(), occpy.opt.OptimizerSettings.saddle_point()
)
optimizer.set_initial_hessian(occpy.XtbCalculator(guess).hessian())
optimizer.set_followed_mode(band.tangent(top))
for _ in range(50):
    optimizer.update(*gfn2(optimizer.get_next_geometry()))
    if optimizer.step():
        break
print(f"\nTransition state: E = {optimizer.current_energy():.8f} Ha")
print(optimizer.get_next_geometry())
