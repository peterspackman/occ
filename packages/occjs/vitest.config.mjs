import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'test/',
        'scripts/',
        'dist/',
        '**/*.d.ts',
        '**/*.config.js',
        '**/*.config.mjs',
        '**/occjs.js' // Exclude the generated WASM wrapper
      ]
    },
    testTimeout: 30000, // 30 seconds for WASM initialization
    hookTimeout: 30000,
    teardownTimeout: 10000,
    pool: 'forks', // Use separate processes for better isolation with WASM
    poolOptions: {
      forks: {
        singleFork: true // Run tests sequentially in forks to avoid WASM conflicts
      }
    },
    // Allow mixed module formats
    deps: {
      external: [/occjs\.js$/] // Don't transform the WASM wrapper
    }
  }
});