// Temporary module worker for running OCC commands (fresh state each time).
// Spawn with: new Worker('occ-run-worker.js', { type: 'module' })
import createOccCliModule from '../dist/occ.js';

self.onmessage = async function (e) {
  const { command, cwd, files } = e.data;

  let Module;
  try {
    Module = await createOccCliModule({
      print: (text) => self.postMessage({ type: 'output', text }),
      printErr: (text) => self.postMessage({ type: 'error', text }),
      noInitialRun: true,
      locateFile: (path) => new URL(`../dist/${path}`, import.meta.url).href,
    });
  } catch (error) {
    self.postMessage({ type: 'error', text: `Failed to load occ.js: ${error.message}` });
    self.postMessage({ type: 'exit', code: 1, files: {} });
    return;
  }

  function collectFiles(dir, outputFiles) {
    let contents;
    try {
      contents = Module.FS.readdir(dir);
    } catch (e) {
      return;
    }
    for (const item of contents) {
      if (item === '.' || item === '..') continue;

      const fullPath = dir === '/' ? '/' + item : dir + '/' + item;
      try {
        const stat = Module.FS.stat(fullPath);
        if (Module.FS.isDir(stat.mode)) {
          collectFiles(fullPath, outputFiles);
        } else {
          outputFiles[fullPath] = Module.FS.readFile(fullPath);
        }
      } catch (e) {
        // Skip files we can't read
      }
    }
  }

  try {
    // Sync files from persistent storage
    if (files) {
      for (const [path, content] of Object.entries(files)) {
        // Create parent directories if needed
        const parts = path.split('/').filter((p) => p);
        let currentPath = '/';
        for (let i = 0; i < parts.length - 1; i++) {
          currentPath += parts[i];
          try {
            Module.FS.mkdir(currentPath);
          } catch (e) {
            // Directory might already exist
          }
          currentPath += '/';
        }
        Module.FS.writeFile(path, content);
      }
    }

    // Set working directory
    if (cwd) {
      try {
        Module.FS.chdir(cwd);
      } catch (e) {
        // If directory doesn't exist, stay in root
      }
    }

    self.postMessage({ type: 'ready' });

    // main() runs on its own thread: runMain resolves with its exit status
    // once it has finished (non-zero included), and rejects if occ aborts.
    // This module is then done; the next command gets a fresh one.
    const args = command.split(/\s+/).filter((a) => a.length > 0);
    const exitCode = await Module.runMain(args);

    // Collect all files from filesystem to send back
    const outputFiles = {};
    collectFiles('/', outputFiles);

    self.postMessage({ type: 'exit', code: exitCode, files: outputFiles });
  } catch (error) {
    self.postMessage({ type: 'error', text: `Runtime error: ${error.message}` });
    self.postMessage({ type: 'exit', code: 1, files: {} });
  }
};
