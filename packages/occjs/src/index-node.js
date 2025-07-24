/**
 * Node.js-specific OCC JavaScript bindings
 * This file contains Node.js-specific imports and functionality
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { loadOCC as loadOCCBase, getModule } from './module-loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Node.js-specific loadOCC that handles file paths
 */
export async function loadOCC(options = {}) {
  // If no wasmPath is provided, use Node.js path resolution
  if (!options.wasmPath) {
    options.wasmPath = path.join(__dirname, 'occjs.wasm');
  }
  if (!options.dataPath) {
    options.dataPath = path.join(__dirname, 'occjs.data');
  }
  
  return loadOCCBase(options);
}

// Re-export everything else from the browser implementation
export * from './index-browser-impl.js';
export { getModule };