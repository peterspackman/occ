#!/usr/bin/env python3
"""Transition state of HCN -> HNC with GFN2-xTB: a saddle point search from a
rough guess and a computed Hessian, checked by its single imaginary mode.

The command line equivalent: occ scf guess.xyz gfn2 --driver ts

Run:
    python examples/python/transition_state.py
"""

import numpy as np
import occpy

# A rough guess, H bridging C and N (Angstrom, rows x, y, z; columns atoms)
guess = occpy.Molecule(
    np.array([6, 7, 1]),
    np.array([[0.0, 1.18, 0.45], [0.0, 0.0, 1.12], [0.0, 0.0, 0.0]]),
)

optimizer = occpy.opt.GeometryOptimizer(
    guess,
    occpy.opt.ConvergenceCriteria(),
    occpy.opt.OptimizerSettings.saddle_point(),
)
# A saddle point search needs the negative curvature a model cannot give
optimizer.set_initial_hessian(occpy.XtbCalculator(guess).hessian())

for step in range(50):
    calc = occpy.XtbCalculator(optimizer.get_next_geometry())
    energy, gradient = calc.energy_and_gradient()
    print(f"step {step:2d}: E = {energy:.8f} Ha")
    optimizer.update(energy, gradient)
    if optimizer.step():
        break

ts = optimizer.get_next_geometry()
print(f"\nConverged: {optimizer.is_converged()}, E = {optimizer.current_energy():.8f} Ha")
print(ts)

frequencies = occpy.XtbCalculator(ts).vibrational_modes(project_tr_rot=True)
for f in frequencies.get_all_frequencies():
    if f < -50:  # one, about 1426i
        print(f"Imaginary mode: {-f:.1f}i cm-1")
