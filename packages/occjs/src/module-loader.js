/**
 * Module loader that provides environment-specific implementations
 * This helps avoid circular dependencies and keeps environment logic in one place
 */

// Cache for the loaded module
let moduleInstance = null;
let modulePromise = null;

/**
 * Load and initialize the OCC WASM module
 * @param {Object} options - Configuration options
 * @returns {Promise<Object>} The initialized OCC module
 */
export async function loadOCC(options = {}) {
  // Return cached instance if already loaded
  if (moduleInstance) {
    return moduleInstance;
  }

  // Return existing promise if loading is in progress
  if (modulePromise) {
    return modulePromise;
  }

  modulePromise = (async () => {
    try {
      // Load the module factory
      const createModule = await import('./occjs.js');
      
      // Handle different module formats
      const moduleFactory = typeof createModule === 'function' 
        ? createModule 
        : (createModule.default || createModule);
      
      // Determine paths based on environment
      const isNode = typeof window === 'undefined';
      let wasmPath = options.wasmPath;
      let dataPath = options.dataPath;
      
      if (!wasmPath) {
        if (isNode) {
          // Only use import.meta.url in Node.js
          const fileURL = new URL('./occjs.wasm', import.meta.url);
          wasmPath = fileURL.pathname;
        } else {
          // In browser, use relative path
          wasmPath = './occjs.wasm';
        }
      }
      
      if (!dataPath) {
        if (isNode) {
          // Only use import.meta.url in Node.js
          const fileURL = new URL('./occjs.data', import.meta.url);
          dataPath = fileURL.pathname;
        } else {
          // In browser, use relative path
          dataPath = './occjs.data';
        }
      }
      
      // Initialize the module with options
      const Module = await moduleFactory({
        locateFile: (filename) => {
          if (filename.endsWith('.wasm')) {
            return wasmPath;
          }
          if (filename.endsWith('.data')) {
            return dataPath;
          }
          return filename;
        },
        ...options.env
      });

      // Set default log level to reduce noise
      if (Module.LogLevel && Module.setLogLevel) {
        Module.setLogLevel(Module.LogLevel.WARN || 3);
      }

      // Set to single thread by default to avoid SharedArrayBuffer/CORS requirements in browser
      // Users can call Module.setNumThreads(n) to enable multithreading when needed
      if (Module.setNumThreads) {
        Module.setNumThreads(1);
      }

      // Set data directory to use preloaded files
      Module.setDataDirectory('/');
      console.log('OCC module loaded with data directory:', Module.getDataDirectory());

      moduleInstance = Module;
      return Module;
    } catch (error) {
      modulePromise = null; // Reset on error to allow retry
      throw new Error(`Failed to load OCC module: ${error.message}`);
    }
  })();

  return modulePromise;
}

/**
 * Get the loaded module instance
 * @returns {Object} The OCC module
 * @throws {Error} If module is not loaded
 */
export function getModule() {
  if (!moduleInstance) {
    throw new Error('OCC module not loaded. Call loadOCC() first.');
  }
  return moduleInstance;
}