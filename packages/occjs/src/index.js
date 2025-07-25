/**
 * OCC JavaScript/WebAssembly bindings
 * Main entry point - uses conditional exports in package.json to route to correct implementation
 */

// For Node.js environments, re-export from index-node.js
// For browser environments, this file won't be loaded (index.browser.js will be used instead)
export * from './index-node.js';