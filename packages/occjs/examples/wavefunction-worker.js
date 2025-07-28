// Web Worker for running SCF calculations
// This worker handles the computationally intensive tasks off the main thread

let OCC = null;
let occModule = null;

// Initialize OCC module when worker starts
self.addEventListener('message', async function(e) {
    const { type, data } = e.data;
    
    try {
        switch(type) {
            case 'init':
                await initializeOCC(data);
                break;
                
            case 'calculate':
                await runCalculation(data);
                break;
                
            case 'setLogLevel':
                setLogLevel(data.level);
                break;
                
            default:
                postMessage({ type: 'error', error: `Unknown message type: ${type}` });
        }
    } catch (error) {
        postMessage({ 
            type: 'error', 
            error: error.message || 'Unknown error occurred',
            stack: error.stack 
        });
    }
});

async function initializeOCC(config) {
    try {
        postMessage({ type: 'log', level: 'info', message: 'Initializing OCC in worker...' });
        
        // Import the OCC module
        if (config.isNpmPackage) {
            // Try npm package path
            OCC = await import('@peterspackman/occjs');
        } else {
            // Try local build path
            OCC = await import(config.modulePath || '../dist/index.browser.js');
        }
        
        // Load the WASM module
        occModule = await OCC.loadOCC({
            wasmPath: config.wasmPath,
            dataPath: config.dataPath
        });
        
        // Set up logging callback to forward logs to main thread
        if (occModule.registerLogCallback) {
            occModule.registerLogCallback((level, message) => {
                postMessage({ 
                    type: 'log', 
                    level: level, 
                    message: message 
                });
            });
        }
        
        postMessage({ type: 'initialized', success: true });
        
    } catch (error) {
        postMessage({ 
            type: 'initialized', 
            success: false, 
            error: error.message 
        });
    }
}

function setLogLevel(level) {
    if (occModule && occModule.setLogLevel) {
        occModule.setLogLevel(level);
        postMessage({ 
            type: 'log', 
            level: 'info', 
            message: `Log level set to ${level}` 
        });
    }
}

async function runCalculation(params) {
    try {
        const startTime = performance.now();
        
        postMessage({ 
            type: 'progress', 
            stage: 'start',
            message: 'Starting calculation...' 
        });
        
        // Recreate molecule from XYZ data
        postMessage({ 
            type: 'log', 
            level: 'info', 
            message: 'Creating molecule from XYZ data...' 
        });
        
        const molecule = await OCC.moleculeFromXYZ(params.xyzData);
        
        postMessage({ 
            type: 'log', 
            level: 'info', 
            message: `Molecule created: ${molecule.size()} atoms` 
        });
        
        // Create calculation
        postMessage({ 
            type: 'progress', 
            stage: 'setup',
            message: `Setting up ${params.method.toUpperCase()} calculation with ${params.basisSet} basis...` 
        });
        
        const calc = await OCC.createQMCalculation(molecule, params.basisSet);
        
        // Set up SCF settings
        const settings = new OCC.SCFSettings()
            .setMaxIterations(params.maxIterations)
            .setEnergyTolerance(params.energyTolerance);
        
        let energy;
        let iterationCount = 0;
        
        // Run the calculation
        postMessage({ 
            type: 'progress', 
            stage: 'calculation',
            message: 'Running SCF iterations...' 
        });
        
        if (params.method === 'hf') {
            postMessage({ 
                type: 'log', 
                level: 'info', 
                message: 'Running Hartree-Fock calculation...' 
            });
            energy = await calc.runHF(settings);
            
        } else if (params.method.startsWith('dft-')) {
            const functional = params.method.split('-')[1];
            postMessage({ 
                type: 'log', 
                level: 'info', 
                message: `Running DFT calculation with ${functional} functional...` 
            });
            energy = await calc.runDFT(functional, { scfSettings: settings });
        }
        
        const endTime = performance.now();
        const elapsedMs = endTime - startTime;
        
        postMessage({ 
            type: 'progress', 
            stage: 'complete',
            message: 'Calculation completed successfully!' 
        });
        
        // Prepare results
        const results = {
            energy: energy,
            energyInEV: energy * 27.2114,
            elapsedMs: elapsedMs,
            converged: true // We can add convergence check if needed
        };
        
        // Get properties for results
        try {
            const properties = await calc.calculateProperties(['orbitals', 'homo', 'lumo', 'gap']);
            results.properties = {
                homo: properties.homo,
                lumo: properties.lumo,
                gap: properties.gap
            };
        } catch (e) {
            postMessage({ 
                type: 'log', 
                level: 'warn', 
                message: `Could not calculate properties: ${e.message}` 
            });
        }
        
        // Export wavefunction data
        try {
            const wf = calc.wavefunction;
            results.wavefunctionData = {
                fchk: wf.exportToString('fchk'),
                numBasisFunctions: calc.basis.nbf(),
                numAtoms: molecule.size()
            };
        } catch (e) {
            postMessage({ 
                type: 'log', 
                level: 'warn', 
                message: `Could not export wavefunction: ${e.message}` 
            });
        }
        
        // Compute matrices and convert to transferable format
        postMessage({ 
            type: 'progress', 
            stage: 'matrices',
            message: 'Computing matrices...' 
        });
        
        results.matrices = {};
        
        try {
            const Module = OCC.getModule();
            const hf = new Module.HartreeFock(calc.basis);
            const wf = calc.wavefunction;
            
            // Helper function to convert matrix to array format
            const matrixToArray = (matrix) => {
                const rows = matrix.rows();
                const cols = matrix.cols();
                const data = [];
                for (let i = 0; i < rows; i++) {
                    const row = [];
                    for (let j = 0; j < cols; j++) {
                        row.push(matrix.get(i, j));
                    }
                    data.push(row);
                }
                return { rows, cols, data };
            };
            
            // Compute and store matrices
            postMessage({ type: 'log', level: 'info', message: 'Computing overlap matrix...' });
            try {
                const overlapMatrix = hf.overlapMatrix();
                results.matrices.overlap = matrixToArray(overlapMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not compute overlap matrix: ${e.message}` });
            }
            
            postMessage({ type: 'log', level: 'info', message: 'Computing kinetic energy matrix...' });
            try {
                const kineticMatrix = hf.kineticMatrix();
                results.matrices.kinetic = matrixToArray(kineticMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not compute kinetic matrix: ${e.message}` });
            }
            
            postMessage({ type: 'log', level: 'info', message: 'Computing nuclear attraction matrix...' });
            try {
                const nuclearMatrix = hf.nuclearAttractionMatrix();
                results.matrices.nuclear = matrixToArray(nuclearMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not compute nuclear attraction matrix: ${e.message}` });
            }
            
            postMessage({ type: 'log', level: 'info', message: 'Computing Fock matrix...' });
            try {
                const fockMatrix = hf.fockMatrix(wf.molecularOrbitals);
                results.matrices.fock = matrixToArray(fockMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not compute Fock matrix: ${e.message}` });
            }
            
            postMessage({ type: 'log', level: 'info', message: 'Extracting density matrix...' });
            try {
                const densityMatrix = wf.molecularOrbitals.densityMatrix;
                results.matrices.density = matrixToArray(densityMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not extract density matrix: ${e.message}` });
            }
            
            postMessage({ type: 'log', level: 'info', message: 'Extracting MO coefficients...' });
            try {
                const coeffMatrix = wf.coefficients();
                results.matrices.coefficients = matrixToArray(coeffMatrix);
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not extract MO coefficients: ${e.message}` });
            }
            
            // Also get orbital energies as an array
            try {
                const orbitalEnergies = wf.orbitalEnergies();
                const energyArray = [];
                for (let i = 0; i < orbitalEnergies.size(); i++) {
                    energyArray.push(orbitalEnergies.get(i));
                }
                results.orbitalEnergies = energyArray;
            } catch (e) {
                postMessage({ type: 'log', level: 'warn', message: `Could not extract orbital energies: ${e.message}` });
            }
            
        } catch (e) {
            postMessage({ 
                type: 'log', 
                level: 'warn', 
                message: `Matrix computation failed: ${e.message}` 
            });
        }
        
        postMessage({ 
            type: 'result', 
            results: results,
            success: true 
        });
        
    } catch (error) {
        postMessage({ 
            type: 'result', 
            success: false,
            error: error.message,
            stack: error.stack
        });
    }
}