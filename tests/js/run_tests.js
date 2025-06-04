/**
 * Main test runner for OCC JavaScript/WASM tests
 */

const path = require('path');
const test = require('./test_framework.js');
const { runCoreTests } = require('./test_core.js');
const { runQMTests } = require('./test_qm.js');

// Try to load the OCC module
async function loadOccModule() {
    try {
        // Try different possible paths for the WASM module
        const possiblePaths = [
            path.join(__dirname, './occjs.js'),
            path.join(__dirname, '../../wasm/src/occjs.js'),
            path.join(__dirname, '../../build/src/occjs.js'),
            path.join(__dirname, '../occjs.js')
        ];
        
        let createOccModule = null;
        let modulePath = null;
        
        for (const modPath of possiblePaths) {
            try {
                const moduleExports = require(modPath);
                // Handle both CommonJS and ES6 module exports
                createOccModule = moduleExports.default || moduleExports;
                if (typeof createOccModule === 'function') {
                    modulePath = modPath;
                    break;
                }
            } catch (e) {
                // Try next path
                continue;
            }
        }
        
        if (!createOccModule) {
            throw new Error('Could not find occjs.js module. Please build the WASM bindings first.');
        }
        
        console.log(`Loading OCC module from: ${modulePath}`);
        const Module = await createOccModule();
        
        // Set up logging
        Module.setLogLevel(Module.LogLevel.WARN); // Reduce noise during tests
        
        console.log('✓ OCC WASM module loaded successfully');
        return Module;
        
    } catch (error) {
        console.error('Failed to load OCC module:', error.message);
        console.error('\nTo build the WASM bindings, run:');
        console.error('  ./scripts/build_wasm.sh');
        console.error('or:');
        console.error('  emcmake cmake . -Bwasm -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DENABLE_JS_BINDINGS=ON -GNinja');
        console.error('  cmake --build wasm --target occjs');
        process.exit(1);
    }
}

async function main() {
    console.log('OCC JavaScript/WASM Test Suite');
    console.log('===============================\n');
    
    // Load the module
    const Module = await loadOccModule();
    
    // Create test runner
    const runner = new test.TestRunner();
    
    // Add test suites
    runner.addSuite(async () => await runCoreTests(Module));
    runner.addSuite(async () => await runQMTests(Module));
    
    // Run all tests
    await runner.run();
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

if (require.main === module) {
    main().catch(error => {
        console.error('Test runner failed:', error);
        process.exit(1);
    });
}

module.exports = { main, loadOccModule };