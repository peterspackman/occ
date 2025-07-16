#!/usr/bin/env node

/**
 * CI Environment Check Script
 * Validates that all required files and dependencies are present for CI builds
 */

import fs from 'fs';
import path from 'path';

const errors = [];
const warnings = [];

// Check required files
const requiredFiles = [
  'package.json',
  'package-lock.json',
  'vitest.config.mjs',
  'src/index.js',
  '../../scripts/build_wasm.sh'
];

console.log('🔍 Checking CI requirements...\n');

for (const file of requiredFiles) {
  if (!fs.existsSync(file)) {
    errors.push(`Missing required file: ${file}`);
  } else {
    console.log(`✅ ${file}`);
  }
}

// Check WASM files (may not exist in CI initially)
const wasmFiles = [
  '../../wasm/src/occjs.wasm',
  '../../wasm/src/occjs.js',
  'src/occjs.wasm',
  'src/occjs.js'
];

let wasmExists = false;
for (const file of wasmFiles) {
  if (fs.existsSync(file)) {
    wasmExists = true;
    console.log(`✅ ${file} (WASM)`);
  }
}

if (!wasmExists) {
  warnings.push('No WASM files found - will need to build during CI');
}

// Check package.json scripts
try {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const requiredScripts = ['test', 'build', 'build:wasm', 'lint', 'typecheck'];
  
  for (const script of requiredScripts) {
    if (!pkg.scripts[script]) {
      errors.push(`Missing required script in package.json: ${script}`);
    } else {
      console.log(`✅ npm script: ${script}`);
    }
  }
} catch (e) {
  errors.push(`Failed to parse package.json: ${e.message}`);
}

// Check test files
const testDir = 'test';
if (fs.existsSync(testDir)) {
  const testFiles = fs.readdirSync(testDir).filter(f => f.endsWith('.test.js'));
  console.log(`✅ Found ${testFiles.length} test files`);
  if (testFiles.length === 0) {
    warnings.push('No test files found in test directory');
  }
} else {
  errors.push('Test directory not found');
}

// Report results
console.log('\n📊 CI Check Results:');

if (warnings.length > 0) {
  console.log('\n⚠️  Warnings:');
  warnings.forEach(w => console.log(`   ${w}`));
}

if (errors.length > 0) {
  console.log('\n❌ Errors:');
  errors.forEach(e => console.log(`   ${e}`));
  console.log('\n💡 Fix these issues before running CI');
  process.exit(1);
} else {
  console.log('\n✅ All CI requirements satisfied!');
  if (warnings.length > 0) {
    console.log('   (Some warnings present but not blocking)');
  }
  process.exit(0);
}