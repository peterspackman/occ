#!/bin/bash

# OCC JavaScript/WASM Test Runner
# Builds WASM bindings and runs JavaScript test suite

set -e  # Exit on any error

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    echo "Please install Node.js first: https://nodejs.org/"
    exit 1
fi

# Check if emcmake is available
if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake (Emscripten) is not installed or not in PATH"
    echo "Please install and activate Emscripten first: https://emscripten.org/"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "🧪 OCC JavaScript/WASM Test Runner"
echo "==================================="
echo ""

# Check if WASM bindings already exist
WASM_JS_FILE="wasm/src/occjs.js"
WASM_BINARY_FILE="wasm/src/occjs.wasm"

if [ ! -f "$WASM_JS_FILE" ] || [ ! -f "$WASM_BINARY_FILE" ]; then
    echo "WASM bindings not found, building..."
    echo ""
    
    # Build WASM bindings
    echo "Building WASM bindings with Emscripten..."
    emcmake cmake . -Bwasm -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DENABLE_JS_BINDINGS=ON -DUSE_SYSTEM_EIGEN=OFF -GNinja
    
    echo ""
    echo "Compiling WASM target..."
    cmake --build wasm --target occjs
    echo ""
    
    # Check if build was successful
    if [ ! -f "$WASM_JS_FILE" ] || [ ! -f "$WASM_BINARY_FILE" ]; then
        echo "Error: WASM build failed - output files not found"
        echo "Expected files:"
        echo "  - $WASM_JS_FILE"
        echo "  - $WASM_BINARY_FILE"
        exit 1
    fi
    
    echo "✓ WASM bindings built successfully"
else
    echo "✓ WASM bindings found"
fi

echo ""

# Copy WASM files to test directory for easier access
TEST_DIR="tests/js"
echo "Copying WASM files to test directory..."
cp "$WASM_JS_FILE" "$TEST_DIR/"
cp "$WASM_BINARY_FILE" "$TEST_DIR/"

echo ""

# Run JavaScript tests
echo "Running JavaScript tests..."
echo ""

cd "$TEST_DIR"

# Check Node.js version
NODE_VERSION=$(node --version)
echo "Using Node.js version: $NODE_VERSION"
echo ""

# Run the tests
echo "Starting test execution..."
echo ""

if node run_tests.js; then
    echo ""
    echo "🎉 All JavaScript tests passed!"
    exit 0
else
    echo ""
    echo "❌ Some JavaScript tests failed!"
    exit 1
fi