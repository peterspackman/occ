// Temporary worker for running OCC commands (fresh state each time)

let commandData = null;
let moduleReady = false;

self.onmessage = async function(e) {
    commandData = e.data;

    if (moduleReady) {
        executeCommand();
    }
};

function executeCommand() {
    const { command, cwd, files } = commandData;

    try {
        // Sync files from persistent storage
        if (files) {
            for (const [path, content] of Object.entries(files)) {
                // Create parent directories if needed
                const parts = path.split('/').filter(p => p);
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

        // Call main with arguments
        const args = command.split(/\s+/).filter(a => a.length > 0);
        const exitCode = Module.callMain(args);

        // Collect all files from filesystem to send back
        const outputFiles = {};
        function collectFiles(dir) {
            const contents = Module.FS.readdir(dir);
            for (const item of contents) {
                if (item === '.' || item === '..') continue;

                const fullPath = dir === '/' ? '/' + item : dir + '/' + item;
                try {
                    const stat = Module.FS.stat(fullPath);
                    if (Module.FS.isDir(stat.mode)) {
                        collectFiles(fullPath);
                    } else {
                        outputFiles[fullPath] = Module.FS.readFile(fullPath);
                    }
                } catch (e) {
                    // Skip files we can't read
                }
            }
        }
        collectFiles('/');

        self.postMessage({ type: 'exit', code: exitCode, files: outputFiles });
    } catch (error) {
        if (error && error.name === 'ExitStatus') {
            // Still collect files even on non-zero exit
            const outputFiles = {};
            function collectFiles(dir) {
                try {
                    const contents = Module.FS.readdir(dir);
                    for (const item of contents) {
                        if (item === '.' || item === '..') continue;

                        const fullPath = dir === '/' ? '/' + item : dir + '/' + item;
                        try {
                            const stat = Module.FS.stat(fullPath);
                            if (Module.FS.isDir(stat.mode)) {
                                collectFiles(fullPath);
                            } else {
                                outputFiles[fullPath] = Module.FS.readFile(fullPath);
                            }
                        } catch (e) {}
                    }
                } catch (e) {}
            }
            collectFiles('/');

            self.postMessage({ type: 'exit', code: error.status, files: outputFiles });
        } else {
            self.postMessage({ type: 'error', text: `Runtime error: ${error.message}` });
            self.postMessage({ type: 'exit', code: 1, files: {} });
        }
    }
}

// Set up Module configuration BEFORE loading
var Module = {
    print: (text) => {
        self.postMessage({ type: 'output', text });
    },
    printErr: (text) => {
        self.postMessage({ type: 'error', text });
    },
    onAbort: (msg) => {
        self.postMessage({ type: 'error', text: `Module aborted: ${msg}` });
        self.postMessage({ type: 'exit', code: 1, files: {} });
    },
    onRuntimeInitialized: () => {
        moduleReady = true;
        if (commandData) {
            executeCommand();
        }
    },
    locateFile: (path) => {
        if (path.endsWith('.wasm') || path.endsWith('.data')) {
            const base = self.location.href.substring(0, self.location.href.lastIndexOf('/'));
            return base + '/../dist/' + path;
        }
        return path;
    },
    noInitialRun: true
};

// Load the OCC module
try {
    importScripts('../dist/occ.js');
} catch (error) {
    self.postMessage({ type: 'error', text: `Failed to load occ.js: ${error.message}` });
    self.postMessage({ type: 'exit', code: 1, files: {} });
}
