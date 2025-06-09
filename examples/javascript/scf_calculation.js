/**
 * OCC WASM - SCF Calculation Example
 * 
 * This example demonstrates how to perform a simple Hartree-Fock SCF calculation
 * on a hydrogen molecule using the OCC WebAssembly bindings.
 */

const createOccModule = require('../../wasm/src/occjs.js');

async function runSCFExample() {
    console.log("Loading OCC WASM module...");
    
    try {
        const Module = await createOccModule();
        
        // Configure data directory for basis sets (required for WASM)
        Module.setDataDirectory('./share');
        
        // Configure logging  
        Module.setLogLevel(Module.LogLevel.INFO);
        console.log("✓ OCC module loaded successfully");
        
        // Create H2 molecule geometry (bond length = 1.4 Bohr)
        console.log("\n=== Creating H2 Molecule ===");
        const positions = new Module.Mat3N(2, 3);
        
        // First hydrogen at origin
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        
        // Second hydrogen at 1.4 Bohr along z-axis
        positions.set(0, 1, 0.0);
        positions.set(1, 1, 0.0);
        positions.set(2, 1, 1.4);
        
        const atomicNumbers = new Module.IVec([1, 1]);
        const h2 = new Module.Molecule(atomicNumbers, positions);
        h2.setName("Hydrogen molecule");
        
        console.log(`Molecule: ${h2.name}`);
        console.log(`Number of atoms: ${h2.size()}`);
        console.log(`Molar mass: ${h2.molarMass().toFixed(4)} g/mol`);
        
        const com = h2.centerOfMass();
        console.log(`Center of mass: [${com.x().toFixed(4)}, ${com.y().toFixed(4)}, ${com.z().toFixed(4)}] Bohr`);
        
        // Load basis set
        console.log("\n=== Loading Basis Set ===");
        const atoms = h2.atoms();
        const atomSymbols = [];
        for (let i = 0; i < atoms.size(); i++) {
            const element = new Module.Element(atoms.get(i).atomicNumber);
            atomSymbols.push(element.symbol);
        }
        
        const basis = Module.AOBasis.load(atoms, "sto-3g");
        console.log(`Basis set: ${basis.name()}`);
        console.log(`Number of shells: ${basis.size()}`);
        console.log(`Number of basis functions: ${basis.nbf()}`);
        
        // Create Hartree-Fock object
        console.log("\n=== Setting up Hartree-Fock Calculation ===");
        const hf = new Module.HartreeFock(basis);
        
        // Calculate nuclear repulsion energy
        const nuclearRepulsion = hf.nuclearRepulsion();
        console.log(`Nuclear repulsion energy: ${nuclearRepulsion.toFixed(8)} Hartree`);
        
        // Calculate one-electron integrals
        console.log("\nCalculating one-electron integrals...");
        const overlap = hf.overlapMatrix();
        const kinetic = hf.kineticMatrix();
        const nuclear = hf.nuclearAttractionMatrix();
        
        console.log(`Overlap matrix size: ${overlap.rows()} x ${overlap.cols()}`);
        console.log(`Kinetic energy matrix size: ${kinetic.rows()} x ${kinetic.cols()}`);
        console.log(`Nuclear attraction matrix size: ${nuclear.rows()} x ${nuclear.cols()}`);
        
        // Perform SCF calculation
        console.log("\n=== SCF Calculation ===");
        const scf = new Module.HF(hf, Module.SpinorbitalKind.Restricted);
        
        // Set convergence criteria
        const convergence = scf.convergenceSettings;
        convergence.energyThreshold = 1e-8;
        convergence.commutatorThreshold = 1e-6;
        
        console.log(`Energy convergence threshold: ${convergence.energyThreshold}`);
        console.log(`Commutator convergence threshold: ${convergence.commutatorThreshold}`);
        
        // Set charge and multiplicity (neutral singlet)
        scf.setChargeMultiplicity(0, 1);
        
        console.log("Starting SCF iterations...");
        const startTime = Date.now();
        const totalEnergy = scf.run();
        const endTime = Date.now();
        
        console.log("\n=== Results ===");
        console.log(`SCF converged!`);
        console.log(`Total energy: ${totalEnergy.toFixed(8)} Hartree`);
        console.log(`Electronic energy: ${(totalEnergy - nuclearRepulsion).toFixed(8)} Hartree`);
        console.log(`Nuclear repulsion: ${nuclearRepulsion.toFixed(8)} Hartree`);
        console.log(`Calculation time: ${endTime - startTime} ms`);
        
        // Get final wavefunction
        console.log("\n=== Wavefunction Analysis ===");
        const wavefunction = scf.wavefunction();
        const mo = wavefunction.molecularOrbitals;
        
        console.log(`Spinorbital kind: ${mo.kind === Module.SpinorbitalKind.Restricted ? 'Restricted' : 'Unrestricted'}`);
        console.log(`Number of alpha electrons: ${mo.numAlpha}`);
        console.log(`Number of beta electrons: ${mo.numBeta}`);
        console.log(`Number of basis functions: ${mo.numAo}`);
        
        // Get orbital energies
        const orbitalEnergies = mo.orbitalEnergies;
        console.log("\nOrbital energies (Hartree):");
        for (let i = 0; i < Math.min(5, orbitalEnergies.size()); i++) {
            const energy = orbitalEnergies.get(i);
            const occupation = i < mo.numAlpha ? "occupied" : "virtual";
            console.log(`  Orbital ${i + 1}: ${energy.toFixed(6)} (${occupation})`);
        }
        
        // Calculate Mulliken charges
        const mullikenCharges = wavefunction.mullikenCharges();
        console.log("\nMulliken partial charges:");
        for (let i = 0; i < mullikenCharges.size(); i++) {
            const charge = mullikenCharges.get(i);
            console.log(`  Atom ${i + 1} (H): ${charge.toFixed(6)} e`);
        }
        
        // Verify charge conservation
        let totalCharge = 0;
        for (let i = 0; i < mullikenCharges.size(); i++) {
            totalCharge += mullikenCharges.get(i);
        }
        console.log(`Total charge: ${totalCharge.toFixed(6)} e`);
        
        console.log("\n✓ SCF calculation completed successfully!");
        
        // Clean up (automatic with Emscripten, but good practice)
        // Objects will be garbage collected when they go out of scope
        
    } catch (error) {
        console.error("Error during SCF calculation:", error.message);
        if (error.stack) {
            console.error("Stack trace:", error.stack);
        }
    }
}

// Run the example
if (require.main === module) {
    runSCFExample().catch(console.error);
}

module.exports = { runSCFExample };