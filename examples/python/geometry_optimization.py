#!/usr/bin/env python3
"""
Simple example of geometry optimization and frequency calculation using OCC Python bindings.

This demonstrates the basic workflow:
1. Create a molecule
2. Set up quantum chemistry method (HF)
3. Run geometry optimization
4. Calculate vibrational frequencies
"""

import occpy as occ
import numpy as np


def main():
    print("OCC Python Optimization Example")
    print("================================")
    
    # 1. Create a water molecule with distorted geometry
    atomic_numbers = np.array([8, 1, 1], dtype=np.int32)  # O, H, H
    positions = np.array([
        [0.0, 0.0, 0.0],      # Oxygen
        [0.0, 0.8, 0.6],      # Hydrogen - elongated bonds  
        [0.0, -0.8, 0.6],     # Hydrogen
    ], dtype=np.float64).T  # Transpose to get 3xN shape
    molecule = occ.Molecule(atomic_numbers, positions)
    print(f"Initial geometry:\n{molecule}")
    
    # 2. Define function to compute energy and gradient
    def compute_energy_gradient(mol):
        # Load basis set for current geometry
        basis = occ.AOBasis.load(mol.atoms(), "3-21G")
        hf = occ.HartreeFock(basis)
        
        # Run SCF calculation
        scf = occ.HF(hf)
        scf.set_charge_multiplicity(0, 1)  # Neutral singlet
        energy = scf.run()
        
        # Compute gradient
        wfn = scf.wavefunction()
        gradient = hf.compute_gradient(wfn.molecular_orbitals)
        
        return energy, gradient
    
    # 3. Set up optimization parameters
    criteria = occ.opt.ConvergenceCriteria()
    criteria.gradient_max = 1e-4    # Maximum gradient component
    criteria.gradient_rms = 1e-5    # RMS gradient
    
    # 4. Run geometry optimization step-by-step
    print("\nRunning geometry optimization...")
    optimizer = occ.opt.BernyOptimizer(molecule, criteria)
    
    converged = False
    max_steps = 25
    
    for step in range(max_steps):
        # Get current geometry for evaluation
        current_mol = optimizer.get_next_geometry()
        
        # Compute energy and gradient
        energy, gradient = compute_energy_gradient(current_mol)
        
        # Update optimizer
        optimizer.update(energy, gradient)
        
        print(f"Step {step + 1}: E = {energy:.8f} Ha, |grad| = {np.linalg.norm(gradient):.6f}")
        
        # Check for convergence
        if optimizer.step():
            print(f"Optimization converged in {step + 1} steps!")
            converged = True
            break
    
    if not converged:
        print(f"Optimization did not converge in {max_steps} steps")
    
    # Get final results
    final_mol = optimizer.get_next_geometry()
    final_energy = optimizer.current_energy()
    
    print(f"Final energy: {final_energy:.8f} Ha")
    print(f"Optimized geometry:\n{final_mol}")
    
    # 5. Calculate vibrational frequencies
    if converged:
        print("\nCalculating vibrational frequencies...")
        
        # Set up final calculation
        basis = occ.AOBasis.load(final_mol.atoms(), "3-21G")
        hf = occ.HartreeFock(basis)
        scf = occ.HF(hf)
        scf.set_charge_multiplicity(0, 1)
        energy = scf.run()
        wfn = scf.wavefunction()
        
        # Compute Hessian matrix using convenience method
        hess_evaluator = hf.hessian_evaluator()
        hess_evaluator.set_step_size(0.005)  # Step size in Bohr
        hessian = hess_evaluator(wfn.molecular_orbitals)
        
        # Compute vibrational modes
        vib_modes = occ.compute_vibrational_modes(hessian, final_mol)
        
        print(f"Computed {vib_modes.n_modes()} vibrational modes")
        print("\nFrequencies (cm⁻¹):")
        frequencies = vib_modes.get_all_frequencies()
        
        for i, freq in enumerate(frequencies):
            if freq < 0:
                print(f"  Mode {i+1:2d}: {abs(freq):8.2f}i cm⁻¹ (imaginary)")
            else:
                print(f"  Mode {i+1:2d}: {freq:8.2f} cm⁻¹")
        
        print(f"\nVibrational analysis summary:")
        print(vib_modes.summary_string())


if __name__ == "__main__":
    main()
