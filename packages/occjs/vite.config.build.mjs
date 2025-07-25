import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/index.js'),
      name: 'OCC',
      fileName: 'occjs-bundle',
      formats: ['es']
    },
    rollupOptions: {
      external: ['./occjs.js'], // Don't bundle the WASM module itself
      output: {
        dir: 'dist',
        format: 'es',
        paths: {
          './occjs.js': './occjs.js'
        }
      }
    },
    target: 'esnext',
    minify: false,
    emptyOutDir: false,
  }
});