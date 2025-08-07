/**
 * Browser-compatible OCC JavaScript bindings
 * Re-exports from the main index.js without Node.js dependencies
 */

// Re-export everything from the implementation (which includes all WASM exports)
export * from './index-browser-impl.js';