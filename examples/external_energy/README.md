# External Energy Interface Examples

This directory contains examples for using OCC's external energy interface with machine learning potentials.

## Overview

The external energy interface allows OCC to use external programs for energy calculations by:

1. Sending molecular geometry as JSON to stdin of external programs
2. Receiving energy values as JSON from stdout
3. Using these energies for lattice energy calculations and crystal growth simulations

## Files

- `petmad_host.py` - FastAPI server that loads PET-MAD model once and serves energy calculations
- `petmad_client.py` - Lightweight client that forwards requests to the host server
- `acenaphthene.cif` - Example crystal structure for testing
- `README.md` - This documentation

## Quick Start

### 1. Start the PET-MAD Host Server

```bash
# From this directory, this assumes you're using uv for python
./petmad_host.py
```

This will:
- Download and load the PET-MAD model (first run only)
- Start a FastAPI server on localhost:5001
- Keep the model in memory for fast calculations

### 2. Run OCC with External Energy

```bash
# Calculate lattice energy using external PET-MAD energy
occ elat acenaphthene.cif --external-command "./petmad_client.py"

# The model will automatically be set to "external" when --external-command is provided
```

## Architecture

The system uses a host/client architecture to avoid reloading expensive ML models:

```
OCC → petmad_client.py → HTTP → petmad_host.py → PET-MAD model
```

1. **OCC** calls `petmad_client.py` for each energy calculation
2. **Client** forwards JSON requests to the host server via HTTP
3. **Host** uses the pre-loaded PET-MAD model to calculate energies
4. **Results** flow back through the same chain

## JSON Protocol

### Input (OCC → External Program)

```json
{
  "molecule": {
    "elements": ["C", "O", "N", "N", "H", "H", "H", "H"],
    "positions": [
      [0.0, 0.0, 1.543],
      [0.0, 0.0, 2.797],
      [0.817, 0.817, 0.844],
      [-0.817, -0.817, 0.844],
      [0.790, 0.790, 0.0],
      [1.453, 1.453, 1.219],
      [-0.790, -0.790, 0.0],
      [-1.453, -1.453, 1.219]
    ],
    "name": ""
  },
  "task": "single_point_energy",
  "metadata": {
    "source": "OCC ExternalEnergyModel",
    "timestamp": 1706123456,
    "num_atoms": 8
  }
}
```

### Output (External Program → OCC)

```json
{
  "energy": -12.345678,
  "energy_eV": -335.789,
  "model": "PET-MAD (host server)"
}
```

## Host Server Options

```bash
./petmad_host.py --help
usage: petmad_host.py [-h] [--version VERSION] [--device DEVICE] [--port PORT] [--host HOST]

PET-MAD Energy Host Server

options:
  -h, --help         show this help message and exit
  --version VERSION  PET-MAD model version
  --device DEVICE    Device to use (cpu/cuda)
  --port PORT        Port to run server on
  --host HOST        Host to bind to
```

Examples:
```bash
# Use GPU if available
./petmad_host.py --device cuda --port 5001

# Use specific model version
./petmad_host.py --version v1.0.0 --port 5001
```

## Advanced Usage

### Custom External Programs

You can create your own external energy programs that follow the JSON protocol:

```python
#!/usr/bin/env python3
import json
import sys

# Read input from stdin
input_data = json.load(sys.stdin)
molecule = input_data["molecule"]

# Your energy calculation here
energy = calculate_energy(molecule["elements"], molecule["positions"])

# Output result to stdout
result = {
    "energy": energy,  # Energy in Hartree
    "model": "MyCustomModel"
}
print(json.dumps(result))
```

### Multiple Models

Run different models on different ports:

```bash
# Terminal 1: PET-MAD on port 5001
./petmad_host.py --port 5001

# Terminal 2: Another model on port 5002
./another_model_host.py --port 5002

# Use specific model
occ elat --external-command "./petmad_client.py http://localhost:5001" urea.cif
```

## Implementation Details

### External Energy Model Class

The `ExternalEnergyModel` class in OCC:

- Inherits from `EnergyModelBase`
- Computes monomer energies during initialization
- Calculates interaction energies as: E(dimer) - E(monomer1) - E(monomer2)
- Applies energy thresholding to reduce numerical noise
- Provides error handling and validation

### Configuration Options

```cpp
struct ExternalEnergyOptions {
  std::string command;                    // External program command
  double energy_threshold{1e-8};          // Energy threshold (Hartree)
  int timeout_seconds{30};                // Program timeout
  std::string working_directory{"."};     // Working directory
};
```

## Further Reading

- [PET-MAD Documentation](https://github.com/dxq-git/pet-mad)
- [OCC External Energy Model Source](/src/interaction/external_energy_model.cpp)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
