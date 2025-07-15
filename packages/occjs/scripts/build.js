#!/usr/bin/env node

/**
 * Build script for OCC JavaScript package
 * Copies WASM files and prepares the distribution
 */

const fs = require('fs');
const path = require('path');

const WASM_BUILD_DIR = path.join(__dirname, '../../../wasm/src');
const SRC_DIR = path.join(__dirname, '../src');
const DIST_DIR = path.join(__dirname, '../dist');

// Ensure dist directory exists
if (!fs.existsSync(DIST_DIR)) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

// Files to copy
const filesToCopy = [
  { src: path.join(WASM_BUILD_DIR, 'occjs.js'), dest: path.join(SRC_DIR, 'occjs.js') },
  { src: path.join(WASM_BUILD_DIR, 'occjs.wasm'), dest: path.join(SRC_DIR, 'occjs.wasm') }
];

// Copy WASM files to src for development
console.log('Copying WASM files...');
for (const file of filesToCopy) {
  if (fs.existsSync(file.src)) {
    fs.copyFileSync(file.src, file.dest);
    console.log(`✓ Copied ${path.basename(file.src)} to src/`);
  } else {
    console.error(`✗ Error: ${file.src} not found. Run build:wasm first.`);
    process.exit(1);
  }
}

// Copy all files from src to dist
console.log('\nBuilding distribution...');
const srcFiles = fs.readdirSync(SRC_DIR);
for (const file of srcFiles) {
  const srcPath = path.join(SRC_DIR, file);
  const destPath = path.join(DIST_DIR, file);
  
  if (fs.statSync(srcPath).isFile()) {
    fs.copyFileSync(srcPath, destPath);
    console.log(`✓ Copied ${file} to dist/`);
  }
}

// Create ESM wrapper
const esmWrapper = `/**
 * ESM wrapper for OCC
 */
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const occ = require('./index.js');

export const { 
  loadOCC, 
  moleculeFromXYZ, 
  createMolecule, 
  Elements, 
  BasisSets,
  Module 
} = occ;

export default occ;
`;

fs.writeFileSync(path.join(DIST_DIR, 'index.mjs'), esmWrapper);
console.log('✓ Created ESM wrapper');

// Copy package.json fields to dist
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, '../package.json'), 'utf8'));
const distPackageJson = {
  name: packageJson.name,
  version: packageJson.version,
  description: packageJson.description,
  main: 'index.js',
  types: 'index.d.ts',
  module: 'index.mjs',
  exports: {
    '.': {
      types: './index.d.ts',
      require: './index.js',
      import: './index.mjs'
    },
    './wasm': {
      default: './occjs.wasm'
    }
  },
  repository: packageJson.repository,
  keywords: packageJson.keywords,
  author: packageJson.author,
  license: packageJson.license,
  bugs: packageJson.bugs,
  homepage: packageJson.homepage,
  engines: packageJson.engines
};

fs.writeFileSync(
  path.join(DIST_DIR, 'package.json'), 
  JSON.stringify(distPackageJson, null, 2)
);
console.log('✓ Created dist/package.json');

console.log('\n✅ Build complete!');