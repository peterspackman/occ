#!/usr/bin/env python3
"""
Test and example for geometry optimization and vibrational analysis in OCC Python bindings.

This script demonstrates:
1. Setting up a molecule for optimization
2. Running geometry optimization with the Berny optimizer
3. Computing vibrational frequencies with the Hessian evaluator
4. Analyzing the results

Example usage:
    python test_optimization.py
"""

import numpy as np
import occpy as occ


def create_water_molecule():
    """Create a water molecule with a slightly distorted geometry for optimization."""
    # Water molecule with slightly elongated O-H bonds and compressed angle
    atoms = [
        occ.Atom(8, 0.0, 0.0, 0.0),      # Oxygen
        occ.Atom(1, 0.0, 0.8, 0.6),      # Hydrogen - slightly long bond
        occ.Atom(1, 0.0, -0.8, 0.6),     # Hydrogen - slightly long bond
    ]
    return occ.Molecule(atoms)


def create_hf_calculator(molecule, basis_name="3-21G"):
    """Create a Hartree-Fock calculator for the given molecule."""
    # Load basis set
    basis = occ.AOBasis.load(molecule.atoms, basis_name)
    
    # Create HF object
    hf = occ.HartreeFock(basis)
    
    return hf


def run_geometry_optimization():
    """Example of geometry optimization using the Berny optimizer."""
    print("=== Geometry Optimization Example ===")
    
    # Create initial molecule
    molecule = create_water_molecule()
    print(f"Initial geometry:\n{molecule}")
    
    # Set up HF calculator
    hf = create_hf_calculator(molecule)
    
    # Set up convergence criteria (tighter than default for demonstration)
    criteria = occ.opt.ConvergenceCriteria()
    criteria.gradient_max = 1e-4
    criteria.gradient_rms = 1e-5
    criteria.step_max = 1e-3
    criteria.step_rms = 1e-4
    
    # Create optimizer
    optimizer = occ.opt.BernyOptimizer(molecule, criteria)
    
    print(f"Convergence criteria: max_grad={criteria.gradient_max}, rms_grad={criteria.gradient_rms}")
    print(f"Starting optimization...")
    
    # Energy and gradient function
    def compute_energy_gradient(mol):
        # Update basis for new geometry
        basis = occ.AOBasis.load(mol.atoms, "3-21G")
        hf_calc = occ.HartreeFock(basis)
        
        # Run SCF
        scf = occ.HF(hf_calc)
        scf.set_charge_multiplicity(0, 1)  # Neutral singlet
        energy = scf.run()
        
        # Compute gradient
        wfn = scf.wavefunction()
        gradient = hf_calc.compute_gradient(wfn.molecular_orbitals)
        
        return energy, gradient
    
    # Run optimization step by step
    converged = False
    max_steps = 20
    
    for step in range(max_steps):
        # Get current geometry
        current_mol = optimizer.get_next_geometry() if step > 0 else molecule
        
        # Compute energy and gradient
        energy, gradient = compute_energy_gradient(current_mol)
        
        # Update optimizer
        optimizer.update(energy, gradient)
        
        print(f"Step {step+1}: E = {energy:.8f} Ha, |grad| = {np.linalg.norm(gradient):.6f}")
        
        # Check for convergence
        if optimizer.step():
            print(f"Optimization converged in {step+1} steps!")
            converged = True
            break
    
    if not converged:
        print(f"Optimization did not converge in {max_steps} steps")
    
    # Get final geometry
    final_molecule = optimizer.current_molecule()
    final_energy = optimizer.current_energy()
    
    print(f"\nFinal energy: {final_energy:.8f} Ha")
    print(f"Final geometry:\n{final_molecule}")
    
    return final_molecule, hf


def run_vibrational_analysis(molecule, hf):
    """Example of vibrational frequency analysis."""
    print("\n=== Vibrational Analysis Example ===")
    
    # Create basis for final geometry
    basis = occ.AOBasis.load(molecule.atoms, "3-21G")
    hf_calc = occ.HartreeFock(basis)
    
    # Run SCF to get converged wavefunction
    scf = occ.HF(hf_calc)
    scf.set_charge_multiplicity(0, 1)
    energy = scf.run()
    wfn = scf.wavefunction()
    
    print(f"Computing Hessian at optimized geometry...")
    
    # Create Hessian evaluator using convenience method
    hess_evaluator = hf_calc.hessian_evaluator()
    hess_evaluator.set_step_size(0.005)  # 0.005 Bohr step size
    hess_evaluator.set_use_acoustic_sum_rule(True)  # Use optimization
    
    print(f"Hessian settings: step_size={hess_evaluator.step_size():.3f} Bohr, "
          f"acoustic_sum_rule={hess_evaluator.use_acoustic_sum_rule()}")
    
    # Compute Hessian
    hessian = hess_evaluator(wfn.molecular_orbitals)
    print(f"Hessian computed: {hessian.shape}")
    
    # Compute vibrational modes
    print("Computing vibrational modes...")
    vibrational_modes = occ.compute_vibrational_modes(hessian, molecule, project_tr_rot=False)
    
    print(f"Found {vibrational_modes.n_modes()} modes for {vibrational_modes.n_atoms()} atoms")
    
    # Print frequency summary
    print("\nVibrational Analysis Summary:")
    print(vibrational_modes.summary_string())
    
    print("\nFrequencies:")
    print(vibrational_modes.frequencies_string())
    
    # Get frequencies in cm^-1
    frequencies = vibrational_modes.get_all_frequencies()
    print(f"\nAll frequencies (cm⁻¹): {frequencies}")
    
    # Identify imaginary frequencies (negative values indicate imaginary)
    imaginary_freqs = frequencies[frequencies < 0]
    real_freqs = frequencies[frequencies > 0]
    
    if len(imaginary_freqs) > 0:
        print(f"\nFound {len(imaginary_freqs)} imaginary frequencies:")
        for i, freq in enumerate(imaginary_freqs):
            print(f"  {i+1}: {freq:.2f}i cm⁻¹")
    
    print(f"\nReal frequencies ({len(real_freqs)} modes):")
    for i, freq in enumerate(real_freqs):
        print(f"  {i+1}: {freq:.2f} cm⁻¹")
    
    return vibrational_modes


def run_high_level_optimization():
    """Example using the high-level optimize_geometry function."""
    print("\n=== High-Level Optimization Example ===")
    
    molecule = create_water_molecule()
    print(f"Initial geometry:\n{molecule}")
    
    # Define energy/gradient function
    def energy_gradient_func(mol):
        basis = occ.AOBasis.load(mol.atoms, "3-21G")
        hf_calc = occ.HartreeFock(basis)
        scf = occ.HF(hf_calc)
        scf.set_charge_multiplicity(0, 1)
        energy = scf.run()
        wfn = scf.wavefunction()
        gradient = hf_calc.compute_gradient(wfn.molecular_orbitals)
        return energy, gradient
    
    # Set up criteria
    criteria = occ.opt.ConvergenceCriteria()
    criteria.gradient_max = 1e-4
    criteria.gradient_rms = 1e-5
    
    # Run optimization
    final_mol, final_energy, converged, steps = occ.opt.optimize_geometry(
        molecule, energy_gradient_func, criteria, max_steps=20
    )
    
    print(f"Optimization {'converged' if converged else 'did not converge'} in {steps} steps")
    print(f"Final energy: {final_energy:.8f} Ha")
    print(f"Final geometry:\n{final_mol}")
    
    return final_mol


def main():
    """Run all examples."""
    print("OCC Python Optimization and Vibrational Analysis Examples")
    print("=" * 60)
    
    try:
        # Run step-by-step optimization example
        optimized_molecule, hf = run_geometry_optimization()
        
        # Run vibrational analysis
        vibrational_modes = run_vibrational_analysis(optimized_molecule, hf)
        
        # Run high-level optimization example
        final_mol = run_high_level_optimization()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()