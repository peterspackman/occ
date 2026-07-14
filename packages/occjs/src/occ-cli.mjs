#!/usr/bin/env node
// Node CLI driver for the OCC WebAssembly build.
//
// The wasm module ships with basis-set data preloaded in MEMFS at `/`.
// The real working directory is mounted (via NODEFS) at /work so input
// files are read from — and output files written to — the caller's cwd.
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const distDir = path.dirname(fileURLToPath(import.meta.url));
const createOccCliModule = (await import(path.join(distDir, 'occ.js'))).default;

const config = {
  noInitialRun: true,
  locateFile: (p) => path.join(distDir, p),
  // Basis-set data is preloaded into MEMFS at /; point OCC there. A host
  // OCC_DATA_PATH would name a path that doesn't exist inside the wasm
  // filesystem, so it is deliberately not inherited.
  preRun: [() => {
    config.ENV.OCC_DATA_PATH = '/';
  }],
};
const M = await createOccCliModule(config);

M.FS.mkdir('/work');
M.FS.mount(M.FS.filesystems.NODEFS, { root: process.cwd() }, '/work');
M.FS.chdir('/work');

let code = 0;
try {
  code = M.callMain(process.argv.slice(2));
} catch (e) {
  if (e && e.name === 'ExitStatus') {
    code = e.status;
  } else {
    console.error(e);
    code = 1;
  }
}
process.exit(code);
