#!/usr/bin/env node

import { loadOCC } from './src/index.js';

async function testVFS() {
  console.log('Loading OCC module...');
  const module = await loadOCC();
  
  console.log('Module loaded. Available properties:');
  console.log('module.FS:', typeof module.FS);
  console.log('module.setDataDirectory:', typeof module.setDataDirectory);
  console.log('module.getDataDirectory:', typeof module.getDataDirectory);
  console.log('Current data directory:', module.getDataDirectory());
  
  if (module.FS) {
    console.log('FS available! Testing preloaded files...');
    
    try {
      // Check if basis files are preloaded
      const files = module.FS.readdir('/');
      console.log('Root directory contents:', files);
      
      if (files.includes('basis')) {
        const basisFiles = module.FS.readdir('/basis');
        console.log('Basis directory contents:', basisFiles.slice(0, 5), '...'); // Show first 5
        
        // Try to read a specific basis file
        if (basisFiles.includes('sto-3g.json')) {
          const content = module.FS.readFile('/basis/sto-3g.json', { encoding: 'utf8' });
          const parsed = JSON.parse(content);
          console.log('STO-3G basis loaded successfully! Elements:', Object.keys(parsed.elements || {}));
        }
      }
      
      if (files.includes('methods')) {
        const methodFiles = module.FS.readdir('/methods');
        console.log('Methods directory contents:', methodFiles);
      }
      
    } catch (error) {
      console.error('Preloaded file test failed:', error.message);
    }
  } else {
    console.log('FS not available on module');
  }
  
  console.log('Managers:', !!module._vfsManager, !!module._basisManager);
  
  // Test built-in AOBasis.load() now that data directory is set
  try {
    console.log('\nTesting AOBasis.load() with preloaded data...');
    const atoms = new module.VectorAtom();
    atoms.push_back(new module.Atom(1, 0.0, 0.0, 0.0));
    atoms.push_back(new module.Atom(1, 0.0, 0.0, 1.4));
    
    const basis = module.AOBasis.load(atoms, 'sto-3g');
    console.log('AOBasis.load() successful! nbf =', basis.nbf(), 'size =', basis.size());
  } catch (error) {
    console.error('AOBasis.load() test failed:', error.message);
  }
}

testVFS().catch(console.error);