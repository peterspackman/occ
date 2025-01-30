import pytest
import numpy as np
from occpy import Atom, AOBasis, HartreeFock, SpinorbitalKind, MolecularOrbitals


@pytest.fixture
def water_atoms():
    """Create water atoms vector"""

    return [
        Atom(8, -1.32695761, -0.10593856, 0.01878821),
        Atom(1, -1.93166418, 1.60017351, -0.02171049),
        Atom(1, 0.48664409, 0.07959806, 0.00986248),
    ]


def test_aobasis_pure_spherical(water_atoms):
    """Test AOBasis with pure spherical functions"""
    basis = AOBasis.load(water_atoms, "6-31G")
    basis.set_pure(True)

    # Original test just prints shells, we can add some basic checks
    assert basis is not None
    assert basis.size() > 0


def test_water_rhf_scf_energy(water_atoms):
    """Test RHF SCF energy for water with different basis sets"""

    # Test with STO-3G
    basis_sto3g = AOBasis.load(water_atoms, "STO-3G")
    hf = HartreeFock(basis_sto3g)
    scf = hf.scf()
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -74.963706080054

    # Test with 3-21G
    basis_321g = AOBasis.load(water_atoms, "3-21G")
    hf = HartreeFock(basis_321g)
    scf = hf.scf()
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -75.585325673488


def test_water_uhf_scf_energy(water_atoms):
    """Test UHF SCF energy for water with different basis sets"""
    # Test with STO-3G
    basis_sto3g = AOBasis.load(water_atoms, "STO-3G")
    hf = HartreeFock(basis_sto3g)
    scf = hf.scf(SpinorbitalKind.Unrestricted)
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -74.963706080054

    # Test with 3-21G
    basis_321g = AOBasis.load(water_atoms, "3-21G")
    hf = HartreeFock(basis_321g)
    scf = hf.scf(SpinorbitalKind.Unrestricted)
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -75.585325673488


def test_water_ghf_scf_energy(water_atoms):
    """Test GHF SCF energy for water with different basis sets"""

    # Test with STO-3G
    basis_sto3g = AOBasis.load(water_atoms, "STO-3G")
    hf = HartreeFock(basis_sto3g)
    scf = hf.scf(SpinorbitalKind.General)
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -74.963706080054

    # Test with 3-21G
    basis_321g = AOBasis.load(water_atoms, "3-21G")
    hf = HartreeFock(basis_321g)
    scf = hf.scf(SpinorbitalKind.General)
    scf.convergence_settings.energy_threshold = 1e-8
    e = scf.compute_scf_energy()
    assert pytest.approx(e, abs=1e-6) == -75.585325673488


def test_h2_electric_field():
    """Test electric field evaluation for H2 with STO-3G basis"""
    atoms = [Atom(1, 0.0, 0.0, 0.0), Atom(1, 0.0, 0.0, 1.398397)]
    basis = AOBasis.load(atoms, "sto-3g")

    # Setup density matrix
    D = np.full((2, 2), 0.301228, dtype=np.float64)

    grid_pts = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
    )

    hf = HartreeFock(basis)

    # Setup MO object
    mo = MolecularOrbitals()
    mo.density_matrix = D
    mo.kind = SpinorbitalKind.Restricted

    # Expected ESP values
    expected_esp = np.array([-1.37628, -1.37628, -1.95486, -1.45387])

    # Get nuclear field contribution
    field_values = hf.nuclear_electric_field_contribution(grid_pts)

    # Get ESP contribution
    esp = hf.electronic_electric_potential_contribution(mo, grid_pts)
    np.testing.assert_allclose(esp, expected_esp, rtol=1e-5, atol=1e-5)
