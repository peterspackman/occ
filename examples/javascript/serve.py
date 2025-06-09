#!/usr/bin/env python3
"""
Simple HTTP server for OCC JavaScript/WASM examples.

This server handles CORS headers properly for WebAssembly loading
and serves the examples on a local development server.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

def main():
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    print(f"🧪 OCC JavaScript/WASM Examples Server")
    print(f"=====================================")
    print(f"")
    print(f"Starting server on port {port}...")
    print(f"")
    print(f"📁 Serving files from: {script_dir}")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print(f"")
    print(f"Available examples:")
    print(f"  • http://localhost:{port}/index.html - Interactive demo")
    print(f"  • http://localhost:{port}/simple_molecule.html - Original example")
    print(f"")
    print(f"⚠️  Make sure you have built the WASM bindings first:")
    print(f"   ./scripts/build_wasm.sh")
    print(f"   cp wasm/src/occjs.* examples/javascript/")
    print(f"")
    print(f"Press Ctrl+C to stop the server")
    print(f"")
    
    # Check if WASM files exist
    wasm_js = script_dir / "occjs.js"
    wasm_binary = script_dir / "occjs.wasm"
    
    if not wasm_js.exists() or not wasm_binary.exists():
        print(f"⚠️  WARNING: WASM files not found!")
        print(f"   Missing: {wasm_js if not wasm_js.exists() else wasm_binary}")
        print(f"   Please build the WASM bindings first.")
        print(f"")
    else:
        print(f"✅ WASM files found and ready to serve")
        print(f"")
    
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python3 serve.py {port + 1}")
        else:
            print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()