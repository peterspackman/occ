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


BOHR = 1.8897261246257702  # per Angstrom


def double_well(molecule):
    """A bond 0-1 with minima at 1.6 and 2.4 Bohr and a barrier of 0.0256
    Hartree at 2.0, a spring on 0-2 and one on 1-2 for the bend."""
    x = molecule.positions * BOHR
    energy, gradient = 0.0, np.zeros_like(x)
    for i, j, kind, a, b, c in [(0, 1, "well", 1.0, 1.6, 2.4),
                                (0, 2, "spring", 0.5, 1.81, None),
                                (1, 2, "spring", 0.1, 2.86, None)]:
        d = x[:, i] - x[:, j]
        r = np.linalg.norm(d)
        if kind == "well":
            u, v = r - b, r - c
            energy += a * u * u * v * v
            de = 2 * a * u * v * (u + v)
        else:
            energy += 0.5 * a * (r - b) ** 2
            de = a * (r - b)
        gradient[:, i] += de * d / r
        gradient[:, j] -= de * d / r
    return energy, gradient


def triatomic(r01, r02):
    half = np.radians(104.5 / 2)
    positions = np.array([[0.0, 0.0, 0.0],
                          [r01 * np.sin(half), 0.0, r01 * np.cos(half)],
                          [-r02 * np.sin(half), 0.0, r02 * np.cos(half)]])
    return occ.Molecule(np.array([8, 1, 1]), positions.T / BOHR)


def finite_difference_hessian(molecule, h=1e-4):
    x = molecule.positions
    n = x.shape[1]
    hessian = np.zeros((3 * n, 3 * n))
    for c in range(3 * n):
        columns = []
        for sign in (1, -1):
            y = x.copy()
            y[c % 3, c // 3] += sign * h / BOHR
            columns.append(double_well(occ.Molecule(np.array([8, 1, 1]), y))[1])
        hessian[:, c] = (columns[0] - columns[1]).flatten(order="F") / (2 * h)
    return 0.5 * (hessian + hessian.T)


def bond(molecule, i, j):
    x = molecule.positions * BOHR
    return np.linalg.norm(x[:, i] - x[:, j])


def test_saddle_point_search():
    start = triatomic(1.85, 1.75)
    optimizer = occ.opt.GeometryOptimizer(
        start, occ.opt.ConvergenceCriteria(),
        occ.opt.OptimizerSettings.saddle_point())
    optimizer.set_initial_hessian(finite_difference_hessian(start))
    for _ in range(50):
        optimizer.update(*double_well(optimizer.get_next_geometry()))
        if optimizer.step():
            break
    assert optimizer.is_converged()
    assert optimizer.negative_modes() == 1
    assert abs(bond(optimizer.get_next_geometry(), 0, 1) - 2.0) < 2e-3


def relaxed(r01):
    optimizer = occ.opt.GeometryOptimizer(triatomic(r01, 1.81))
    for _ in range(50):
        optimizer.update(*double_well(optimizer.get_next_geometry()))
        if optimizer.step():
            break
    return optimizer.get_next_geometry()


def test_nudged_elastic_band_climbs_to_the_saddle():
    reactant, product = relaxed(1.6), relaxed(2.4)
    settings = occ.opt.NebSettings()
    settings.images = 5
    band = occ.opt.NudgedElasticBand(reactant, product, settings)
    band.set_endpoint_energies(double_well(reactant)[0],
                               double_well(product)[0])
    for _ in range(200):
        results = [double_well(image) for image in band.images()]
        band.update([e for e, _ in results], [g for _, g in results])
        if band.step():
            break
    assert band.is_converged()
    assert band.climbing()
    top = band.highest_image()
    assert abs(bond(band.path()[top], 0, 1) - 2.0) < 5e-3
    assert len(band.tangent(top)) == 9
