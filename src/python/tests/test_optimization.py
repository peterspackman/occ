"""Geometry optimization through the Python bindings."""

import numpy as np
import occpy as occ


def water():
    """A distorted water molecule (Angstrom)."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.8, -0.8], [0.6, 0.6, 0.6]])
    return occ.Molecule(np.array([8, 1, 1]), positions)


def energy_and_gradient(molecule):
    hf = occ.HartreeFock(occ.AOBasis.load(molecule.atoms(), "STO-3G"))
    scf = hf.scf()
    energy = scf.compute_scf_energy()
    gradient = hf.compute_gradient(scf.wavefunction().molecular_orbitals)
    return energy, gradient


def test_convergence_criteria_are_settable():
    criteria = occ.opt.ConvergenceCriteria()
    criteria.gradient_max = 1e-4
    criteria.use_energy_criterion = True
    assert criteria.gradient_max == 1e-4
    assert criteria.use_energy_criterion


def test_geometry_optimizer_converges_water():
    optimizer = occ.opt.GeometryOptimizer(water())
    assert optimizer.current_step() == 0
    assert not optimizer.is_converged()

    energies = []
    for _ in range(30):
        energy, gradient = energy_and_gradient(optimizer.get_next_geometry())
        energies.append(energy)
        optimizer.update(energy, gradient)
        if optimizer.step():
            break

    assert optimizer.is_converged()
    assert optimizer.current_energy() == min(energies)
    _, gradient = energy_and_gradient(optimizer.get_next_geometry())
    assert np.linalg.norm(gradient, axis=0).max() < 3e-4
