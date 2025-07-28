#!/usr/bin/env -S uv run 
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pet-mad",
#   "ase",
#   "numpy",
#   "fastapi",
#   "uvicorn"
# ]
# ///

import json
import sys
import numpy as np
from ase import Atoms
from pet_mad.calculator import PETMADCalculator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from typing import List

app = FastAPI(title="PET-MAD Energy Server", version="1.0.0")

# Global calculator - loaded once
calculator = None

# Pydantic models for request/response
class Molecule(BaseModel):
    elements: List[str]
    positions: List[List[float]]
    name: str = ""

class EnergyRequest(BaseModel):
    molecule: Molecule
    task: str = "single_point_energy"

class EnergyResponse(BaseModel):
    energy: float
    energy_eV: float
    model: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

def load_model(version="latest", device="cpu"):
    """Load the PET-MAD model once"""
    global calculator
    print(f"Loading PET-MAD model: version={version} on {device}")
    try:
        calculator = PETMADCalculator(version=version, device=device)
        print(f"PET-MAD model loaded successfully")
    except Exception as e:
        print(f"Failed to load PET-MAD model: {e}")
        sys.exit(1)

@app.post("/energy", response_model=EnergyResponse)
async def calculate_energy(request: EnergyRequest):
    """API endpoint to calculate energy"""
    if calculator is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract molecule data - OCC sends positions in Bohr, convert to Angstroms
        positions = np.array(request.molecule.positions)  # Bohr from OCC
        elements = request.molecule.elements  # Element symbols
        
        # Debug: print XYZ geometry
        print(f"\n--- Geometry (Angstroms) ---")
        print(f"{len(elements)}")
        print(f"Energy calculation")
        for i, (elem, pos) in enumerate(zip(elements, positions)):
            print(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
        print("---")
        
        # Convert to ASE Atoms object
        atoms = Atoms(symbols=elements, positions=positions)
        atoms.calc = calculator
        
        # Calculate energy
        energy = atoms.get_potential_energy()  # Returns energy in eV
        print(f"Energy: {energy:.6f} eV")
        
        # Convert from eV to Hartree (OCC typically uses Hartree)
        energy_hartree = energy / 27.211386245988  # eV to Hartree conversion
        
        return EnergyResponse(
            energy=energy_hartree,
            energy_eV=energy,
            model="PET-MAD (host server)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        model_loaded=calculator is not None
    )

def main():
    parser = argparse.ArgumentParser(description='PET-MAD Energy Host Server')
    parser.add_argument('--version', default='latest', help='PET-MAD model version')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--port', type=int, default=5001, help='Port to run server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Load the model
    load_model(args.version, args.device)
    
    print(f"Starting PET-MAD energy server on {args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
