"""
Tests for volume generation Python bindings
"""
import numpy as np
from occpy import (
    Molecule, Atom, AOBasis, HartreeFock,
    VolumeCalculator, VolumeGenerationParameters, VolumePropertyKind, SpinConstraint,
    generate_electron_density_cube, generate_mo_cube, generate_esp_cube
)


def test_volume_generation_basic():
    """Test basic volume generation functionality."""
    # Create a simple H2 molecule using atoms list
    atoms = [Atom(1, 0.0, 0.0, 0.0), Atom(1, 0.0, 0.0, 1.4)]
    
    # Create a basis set and run HF calculation
    basis = AOBasis.load(atoms, "sto-3g")
    hf = HartreeFock(basis)
    scf = hf.scf()
    scf.compute_scf_energy()
    wfn = scf.wavefunction()
    
    # Test VolumeCalculator creation
    volume_calc = VolumeCalculator()
    volume_calc.set_wavefunction(wfn)
    
    # Test VolumeGenerationParameters
    params = VolumeGenerationParameters()
    params.property = VolumePropertyKind.ElectronDensity
    params.steps = [5, 5, 5]
    
    # Compute volume
    volume = volume_calc.compute_volume(params)
    
    # Test volume properties
    assert volume.nx() == 5
    assert volume.ny() == 5
    assert volume.nz() == 5
    assert volume.total_points() == 125
    
    # Test data access
    data = volume.get_data()
    data_array = np.array(data).reshape((5, 5, 5))
    assert data_array.shape == (5, 5, 5)
    assert np.any(data_array > 0)  # Should have non-zero density values
    
    # Test cube file generation
    cube_string = volume_calc.volume_as_cube_string(volume)
    assert isinstance(cube_string, str)
    assert len(cube_string) > 0
    assert "electron_density" in cube_string.lower()


def test_convenience_functions():
    """Test convenience functions for cube generation."""
    # Create a simple H2 molecule using atoms list
    atoms = [Atom(1, 0.0, 0.0, 0.0), Atom(1, 0.0, 0.0, 1.4)]
    
    # Create a basis set and run HF calculation
    basis = AOBasis.load(atoms, "sto-3g")
    hf = HartreeFock(basis)
    scf = hf.scf()
    scf.compute_scf_energy()
    wfn = scf.wavefunction()
    
    # Test electron density cube generation
    density_cube = generate_electron_density_cube(wfn, 5, 5, 5)
    assert isinstance(density_cube, str)
    assert len(density_cube) > 0
    
    # Test MO cube generation (HOMO should be index 0 for H2)
    mo_cube = generate_mo_cube(wfn, 0, 5, 5, 5)
    assert isinstance(mo_cube, str)
    assert len(mo_cube) > 0
    
    # Test ESP cube generation
    esp_cube = generate_esp_cube(wfn, 5, 5, 5)
    assert isinstance(esp_cube, str) 
    assert len(esp_cube) > 0


def test_static_methods():
    """Test static methods on VolumeCalculator."""
    # Create a simple H2 molecule using atoms list
    atoms = [Atom(1, 0.0, 0.0, 0.0), Atom(1, 0.0, 0.0, 1.4)]
    
    # Create a basis set and run HF calculation
    basis = AOBasis.load(atoms, "sto-3g")
    hf = HartreeFock(basis)
    scf = hf.scf()
    scf.compute_scf_energy()
    wfn = scf.wavefunction()
    
    # Test static density volume computation
    params = VolumeGenerationParameters()
    params.property = VolumePropertyKind.ElectronDensity
    params.steps = [3, 3, 3]
    
    volume = VolumeCalculator.compute_density_volume(wfn, params)
    assert volume.total_points() == 27
    
    # Test static MO volume computation
    mo_volume = VolumeCalculator.compute_mo_volume(wfn, 0, params)
    assert mo_volume.total_points() == 27


def test_volume_properties():
    """Test volume property enums and data access."""
    # Test VolumePropertyKind enum
    assert hasattr(VolumePropertyKind, 'ElectronDensity')
    assert hasattr(VolumePropertyKind, 'ElectricPotential')
    assert hasattr(VolumePropertyKind, 'CrystalVoid')
    
    # Test SpinConstraint enum
    assert hasattr(SpinConstraint, 'Total')
    assert hasattr(SpinConstraint, 'Alpha')
    assert hasattr(SpinConstraint, 'Beta')


if __name__ == "__main__":
    test_volume_properties()
    test_volume_generation_basic()
    test_convenience_functions() 
    test_static_methods()
    print("All volume generation tests passed!")