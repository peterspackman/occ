#!/usr/bin/env node

/**
 * Build script for OCC JavaScript package
 * Copies WASM files and prepares the distribution
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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
  { src: path.join(WASM_BUILD_DIR, 'occjs.wasm'), dest: path.join(SRC_DIR, 'occjs.wasm') },
  { src: path.join(WASM_BUILD_DIR, 'occjs.data'), dest: path.join(SRC_DIR, 'occjs.data') }
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

// Copy directories recursively
function copyRecursive(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest);
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursive(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

const srcItems = fs.readdirSync(SRC_DIR);
for (const item of srcItems) {
  const srcPath = path.join(SRC_DIR, item);
  const destPath = path.join(DIST_DIR, item);
  
  if (fs.statSync(srcPath).isDirectory()) {
    copyRecursive(srcPath, destPath);
    console.log(`✓ Copied directory ${item}/ to dist/`);
  } else {
    fs.copyFileSync(srcPath, destPath);
    console.log(`✓ Copied ${item} to dist/`);
  }
}

// Create ESM wrapper that re-exports from index.js
const esmWrapper = `/**
 * ESM wrapper for OCC
 */
export * from './index.js';
export { default } from './index.js';
`;

fs.writeFileSync(path.join(DIST_DIR, 'index.mjs'), esmWrapper);
console.log('✓ Created ESM wrapper');

// Copy package.json fields to dist
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, '../package.json'), 'utf8'));
const distPackageJson = {
  name: packageJson.name,
  version: packageJson.version,
  description: packageJson.description,
  type: 'module',
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