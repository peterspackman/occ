#!/bin/bash

# Script to install the project and run Python tests using uv
# This script assumes uv is available in the PATH

set -e  # Exit on any error

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Please install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo ""

# Install the project
echo "Installing project with uv..."
uv pip install . -v
echo ""

# Install test dependencies
echo "Installing test dependencies..."
uv pip install pytest numpy
echo ""

# Run Python tests
echo "Running Python tests..."
cd src/python/tests

# Run all test files
echo "Running core tests..."
python -m pytest test_core.py -v

echo ""
echo "Running crystal tests..."
python -m pytest test_crystal.py -v

echo ""
echo "Running QM tests..."
python -m pytest test_qm.py -v

echo ""
echo "Running DMA tests..."
python -m pytest test_dma.py -v

echo ""
echo "Success!"
