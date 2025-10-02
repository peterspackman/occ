import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const rootDir = dirname(__dirname); // packages/occjs directory
const PORT = 8080;

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.wasm': 'application/wasm',
  '.data': 'application/octet-stream',
  '.json': 'application/json',
  '.css': 'text/css'
};

const server = createServer(async (req, res) => {
  // Set CORS headers for SharedArrayBuffer
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  // Use 'credentialless' to allow cross-origin resources without CORP headers
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');

  let filePath = req.url === '/' ? '/examples/wavefunction_calculator.html' : req.url;
  filePath = join(rootDir, filePath);

  try {
    const content = await readFile(filePath);
    const ext = extname(filePath);
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);
  } catch (err) {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log('CORS headers enabled for SharedArrayBuffer support');
  console.log(`Serving from: ${rootDir}`);
});
