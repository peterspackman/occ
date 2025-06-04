/**
 * Simple JavaScript testing framework for OCC WASM tests
 */

class TestSuite {
    constructor(name) {
        this.name = name;
        this.tests = [];
        this.results = {
            passed: 0,
            failed: 0,
            errors: []
        };
    }
    
    test(description, testFunction) {
        this.tests.push({ description, testFunction });
    }
    
    run() {
        console.log(`\n=== ${this.name} ===`);
        
        for (const { description, testFunction } of this.tests) {
            try {
                testFunction();
                this.results.passed++;
                console.log(`✓ ${description}`);
            } catch (error) {
                this.results.failed++;
                this.results.errors.push({ description, error: error.message });
                console.log(`✗ ${description}: ${error.message}`);
            }
        }
        
        const total = this.results.passed + this.results.failed;
        console.log(`\nResults: ${this.results.passed}/${total} passed`);
        
        if (this.results.failed > 0) {
            console.log('\nFailures:');
            for (const { description, error } of this.results.errors) {
                console.log(`  - ${description}: ${error}`);
            }
        }
        
        return this.results;
    }
}

// Assertion functions
function assertEqual(actual, expected, message = '') {
    if (actual !== expected) {
        throw new Error(`${message}: expected ${expected}, got ${actual}`);
    }
}

function assertNotEqual(actual, expected, message = '') {
    if (actual === expected) {
        throw new Error(`${message}: expected not to equal ${expected}, but got ${actual}`);
    }
}

function assertAlmostEqual(actual, expected, tolerance = 1e-6, message = '') {
    const diff = Math.abs(actual - expected);
    if (diff > tolerance) {
        throw new Error(`${message}: expected ${expected} ± ${tolerance}, got ${actual} (diff: ${diff})`);
    }
}

function assertTrue(condition, message = '') {
    if (!condition) {
        throw new Error(`${message}: expected true, got ${condition}`);
    }
}

function assertFalse(condition, message = '') {
    if (condition) {
        throw new Error(`${message}: expected false, got ${condition}`);
    }
}

function assertThrows(func, expectedError = null, message = '') {
    let threw = false;
    let actualError = null;
    
    try {
        func();
    } catch (error) {
        threw = true;
        actualError = error;
    }
    
    if (!threw) {
        throw new Error(`${message}: expected function to throw an error`);
    }
    
    if (expectedError && !(actualError instanceof expectedError)) {
        throw new Error(`${message}: expected ${expectedError.name}, got ${actualError.constructor.name}`);
    }
}

function createSuite(name) {
    return new TestSuite(name);
}

// Performance timing utilities
function time(func, label = 'Operation') {
    const start = Date.now();
    const result = func();
    const end = Date.now();
    console.log(`${label}: ${end - start} ms`);
    return result;
}

async function timeAsync(func, label = 'Async Operation') {
    const start = Date.now();
    const result = await func();
    const end = Date.now();
    console.log(`${label}: ${end - start} ms`);
    return result;
}

// Memory usage utilities (Node.js only)
function getMemoryUsage() {
    if (typeof process !== 'undefined' && process.memoryUsage) {
        const usage = process.memoryUsage();
        return {
            rss: Math.round(usage.rss / 1024 / 1024), // MB
            heapUsed: Math.round(usage.heapUsed / 1024 / 1024), // MB
            heapTotal: Math.round(usage.heapTotal / 1024 / 1024), // MB
            external: Math.round(usage.external / 1024 / 1024) // MB
        };
    }
    return null;
}

function logMemoryUsage(label = 'Memory usage') {
    const usage = getMemoryUsage();
    if (usage) {
        console.log(`${label}: RSS: ${usage.rss}MB, Heap: ${usage.heapUsed}/${usage.heapTotal}MB, External: ${usage.external}MB`);
    }
}

// Test runner for multiple suites
class TestRunner {
    constructor() {
        this.suites = [];
        this.totalResults = {
            passed: 0,
            failed: 0,
            errors: []
        };
    }
    
    addSuite(suiteRunner) {
        this.suites.push(suiteRunner);
    }
    
    async run() {
        console.log('Starting OCC JavaScript Tests...\n');
        
        const startTime = Date.now();
        logMemoryUsage('Initial memory');
        
        for (const suiteRunner of this.suites) {
            const results = await suiteRunner();
            this.totalResults.passed += results.passed;
            this.totalResults.failed += results.failed;
            this.totalResults.errors.push(...results.errors);
        }
        
        const endTime = Date.now();
        const totalTime = endTime - startTime;
        
        console.log('\n' + '='.repeat(50));
        console.log('FINAL RESULTS');
        console.log('='.repeat(50));
        
        const total = this.totalResults.passed + this.totalResults.failed;
        console.log(`Total tests: ${total}`);
        console.log(`Passed: ${this.totalResults.passed}`);
        console.log(`Failed: ${this.totalResults.failed}`);
        console.log(`Success rate: ${((this.totalResults.passed / total) * 100).toFixed(1)}%`);
        console.log(`Total time: ${totalTime} ms`);
        
        logMemoryUsage('Final memory');
        
        if (this.totalResults.failed > 0) {
            console.log('\nFailed tests:');
            for (const { description, error } of this.totalResults.errors) {
                console.log(`  ✗ ${description}: ${error}`);
            }
            process.exit(1);
        } else {
            console.log('\n🎉 All tests passed!');
            process.exit(0);
        }
    }
}

module.exports = {
    createSuite,
    TestRunner,
    assertEqual,
    assertNotEqual,
    assertAlmostEqual,
    assertTrue,
    assertFalse,
    assertThrows,
    time,
    timeAsync,
    getMemoryUsage,
    logMemoryUsage
};