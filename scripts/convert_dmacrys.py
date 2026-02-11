#!/usr/bin/env python3
"""Convert a DMACRYS regression test directory to a single JSON file.

Reads:
  NEIGHCRYS_input/*.res       - SHELX crystal structure
  NEIGHCRYS_input/dmacrys.dma - Body-frame DMA multipoles (Bohr, Stone convention)
  NEIGHCRYS_input/fit.pots    - Buckingham parameters (eV, Angstrom)
  DMACRYS_output/*.dmaout     - Reference energy breakdown

Outputs:
  A JSON file suitable for loading into OCC via --multipole-json.
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path


# --- .res parser ---

def parse_res_file(path):
    """Parse a SHELX .res file for cell, symops, and atoms.

    Handles LATT directive: LATT N (N>0) adds inversion center to operations.
    |N| encodes centering: 1=P, 2=I, 3=R(obverse), 4=F, 5=A, 6=B, 7=C.
    """
    cell = {}
    atoms = []
    symops = []
    sfac_types = []
    Z = 1
    latt = 1  # default: primitive, centrosymmetric

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            keyword = parts[0].upper()

            if keyword == "CELL":
                cell = {
                    "a": float(parts[2]),
                    "b": float(parts[3]),
                    "c": float(parts[4]),
                    "alpha": float(parts[5]),
                    "beta": float(parts[6]),
                    "gamma": float(parts[7]),
                }

            elif keyword == "ZERR":
                Z = int(parts[1])

            elif keyword == "SYMM":
                symop = line[4:].strip()
                symops.append(symop)

            elif keyword == "SFAC":
                sfac_types = parts[1:]

            elif keyword == "LATT":
                latt = int(parts[1])

            elif keyword == "TITL":
                pass

            elif keyword == "END" or keyword == "HKLF":
                break

            else:
                if len(parts) >= 5 and parts[1].isdigit():
                    type_idx = int(parts[1]) - 1
                    if 0 <= type_idx < len(sfac_types):
                        atoms.append({
                            "label": parts[0],
                            "element": sfac_types[type_idx],
                            "frac_xyz": [float(parts[2]), float(parts[3]), float(parts[4])],
                        })

    # Normalize symops to standard triplet format
    normalized_symops = normalize_shelx_symops(symops, latt)

    return cell, Z, normalized_symops, atoms


def normalize_shelx_symops(symops, latt):
    """Convert SHELX SYMM lines + LATT into normalized symop strings.

    Returns a list of symop strings (NOT including identity x,y,z).
    """
    def normalize(s):
        """Strip spaces and lowercase."""
        return "".join(c for c in s if c != " ").lower()

    # Start with the explicit SYMM operations
    ops = [normalize(s) for s in symops]

    # LATT > 0 means centrosymmetric: add inversion -x,-y,-z
    # and compose with all existing ops
    if latt > 0:
        # Add inversion and its products with existing symops
        base_ops = ["x,y,z"] + list(ops)
        inv_ops = []
        for op in base_ops:
            inv = apply_inversion(op)
            if inv != "x,y,z" and inv not in ops and inv not in inv_ops:
                inv_ops.append(inv)
        ops.extend(inv_ops)

    return ops


def apply_inversion(symop):
    """Apply inversion (-x,-y,-z) to a symop string.

    Transforms each component by negating the variable part.
    e.g., "x+1/2,-y+1/2,-z" -> "-x+1/2,y+1/2,z"
    Wait, that's wrong. Inversion of (x+1/2, -y+1/2, -z) gives
    (-x-1/2, y-1/2, z). We should negate each component fully:
    f(x,y,z) -> -f(-x,-y,-z)... actually for Seitz notation it's simpler.

    For a symop W|t applied to position r: Wr + t
    Inversion of this operation: -Wr - t
    So we negate all components.
    """
    parts = symop.split(",")
    result = []
    for part in parts:
        # Negate the entire expression
        negated = negate_expression(part.strip())
        result.append(negated)
    return ",".join(result)


def negate_expression(expr):
    """Negate a symop expression like 'x+1/2' -> '-x-1/2'."""
    result = []
    # Parse into terms
    terms = []
    current = ""
    for c in expr:
        if c in "+-" and current:
            terms.append(current)
            current = c
        else:
            current += c
    if current:
        terms.append(current)

    for term in terms:
        term = term.strip()
        if not term:
            continue
        # Negate the sign
        if term[0] == "-":
            result.append("+" + term[1:])
        elif term[0] == "+":
            result.append("-" + term[1:])
        else:
            result.append("-" + term)

    negated = "".join(result)
    # Clean up leading +
    if negated.startswith("+"):
        negated = negated[1:]
    return negated


# --- .dma parser ---

def parse_dma_file(path):
    """Parse dmacrys.dma for body-frame multipole sites."""
    sites = []

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("!"):
            i += 1
            continue

        # Try to match site header:
        #   N   LABEL   X Y Z   Next  N2  Limit  RANK
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit() and "Next" in line:
            site_num = int(parts[0])
            label = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            # "Next" at parts[5], next_val at parts[6], "Limit" at parts[7], rank at parts[8]
            rank = int(parts[8])

            # Extract element and atom type from label
            # e.g., "C_F1_1____" -> element="C", atom_type="C_F1"
            # e.g., "H_F1_3____" -> element="H", atom_type="H_F1"
            label_parts = label.rstrip("_").split("_")
            element = label_parts[0]
            atom_type = "_".join(label_parts[:2]) if len(label_parts) >= 2 else element
            atomic_number = element_to_z(element)

            # Read multipole components
            # Rank 0: 1 component (charge)
            # Rank 1: 3 components (dipole)
            # Rank 2: 5 components (quadrupole)
            # Rank 3: 7 components (octupole)
            # Rank 4: 9 components (hexadecapole)
            total_components = sum(2 * l + 1 for l in range(rank + 1))
            components = []
            i += 1
            while len(components) < total_components and i < len(lines):
                line = lines[i].strip()
                if line and not line.startswith("!"):
                    components.extend(float(v) for v in line.split())
                i += 1

            sites.append({
                "label": label,
                "element": element,
                "atomic_number": atomic_number,
                "atom_type": atom_type,
                "position_bohr": [x, y, z],
                "multipoles": {
                    "rank": rank,
                    "components": components,
                },
            })
        else:
            i += 1

    return sites


# --- fit.pots parser ---

def parse_fit_pots(path):
    """Parse fit.pots for Buckingham parameters.

    Preserves DMACRYS atom type names (e.g., "H_F1", "H_F2") for proper filtering.
    """
    pairs = []

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("BUCK"):
            parts = line.split()
            # BUCK  TYPE1  TYPE2
            type1 = parts[1]
            type2 = parts[2]
            el1 = type1.split("_")[0]  # e.g., "C_F1" -> "C"
            el2 = type2.split("_")[0]

            # Next line: A rho C6 [unused] [cutoff]
            i += 1
            params = lines[i].split()
            A = float(params[0])
            rho = float(params[1])
            C6 = float(params[2])

            pairs.append({
                "type1": type1,
                "type2": type2,
                "el1": el1,
                "el2": el2,
                "A_eV": A,
                "rho_ang": rho,
                "C6_eV_ang6": C6,
            })
        i += 1

    return pairs


# --- .dmaout parser ---

def parse_dmaout(path):
    """Parse DMACRYS output for reference energies."""
    initial = {}
    optimized = {}

    with open(path) as f:
        content = f.read()

    # Find all "Contributions to lattice energy" blocks
    blocks = list(re.finditer(
        r"Contributions to lattice energy \(eV per unit cell \[kJ/mol\]\)(.*?)(?=\n\s*\n|\Z)",
        content, re.DOTALL
    ))

    def parse_energy_block(text):
        result = {}

        m = re.search(r"Inter-molecular charge-charge energy.*?=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["charge_charge_inter_eV"] = float(m.group(1).replace("D", "E"))
            result["charge_charge_inter_kJ"] = float(m.group(2).replace("D", "E"))

        m = re.search(r"Total charge-dipole energy\s*=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["charge_dipole_eV"] = float(m.group(1).replace("D", "E"))
            result["charge_dipole_kJ"] = float(m.group(2).replace("D", "E"))

        m = re.search(r"Total dipole-dipole energy\s*=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["dipole_dipole_eV"] = float(m.group(1).replace("D", "E"))
            result["dipole_dipole_kJ"] = float(m.group(2).replace("D", "E"))

        m = re.search(r"Higher multipole interaction energy.*?=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["higher_multipole_eV"] = float(m.group(1).replace("D", "E"))
            result["higher_multipole_kJ"] = float(m.group(2).replace("D", "E"))

        m = re.search(r"Total isotropic repulsion-dispersion.*?=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["repulsion_dispersion_eV"] = float(m.group(1).replace("D", "E"))
            result["repulsion_dispersion_kJ"] = float(m.group(2).replace("D", "E"))

        m = re.search(r"Total lattice energy.*?=\s*([^\[]+)\[([^\]]+)\]", text)
        if m:
            result["total_eV_per_cell"] = float(m.group(1).replace("D", "E"))
            result["total_kJ_per_mol"] = float(m.group(2).replace("D", "E"))

        return result

    if len(blocks) >= 1:
        initial = parse_energy_block(blocks[0].group(1))
    if len(blocks) >= 2:
        optimized = parse_energy_block(blocks[-1].group(1))

    return initial, optimized


def element_to_z(element):
    """Convert element symbol to atomic number."""
    table = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    }
    # Handle multi-character labels like "Cl"
    el = element.capitalize()
    if el in table:
        return table[el]
    if el[0] in table:
        return table[el[0]]
    raise ValueError(f"Unknown element: {element}")


def filter_relevant_pairs(pairs, atom_types_present):
    """Keep only Buckingham pairs for atom types actually in the molecule.

    atom_types_present: set of DMACRYS type prefixes, e.g. {"C_F1", "O_F1", "H_F1"}
    """
    return [
        p for p in pairs
        if p["type1"] in atom_types_present and p["type2"] in atom_types_present
    ]


def convert_dmacrys_directory(directory):
    """Convert a DMACRYS regression test directory to JSON."""
    directory = Path(directory)
    name = directory.name

    # Find input files
    neighcrys = directory / "NEIGHCRYS_input"
    dmacrys_out = directory / "DMACRYS_output"

    res_files = list(neighcrys.glob("*.res"))
    if not res_files:
        raise FileNotFoundError(f"No .res file in {neighcrys}")
    res_file = res_files[0]

    dma_file = neighcrys / "dmacrys.dma"
    if not dma_file.exists():
        raise FileNotFoundError(f"No dmacrys.dma in {neighcrys}")

    pots_file = neighcrys / "fit.pots"
    if not pots_file.exists():
        raise FileNotFoundError(f"No fit.pots in {neighcrys}")

    dmaout_files = list(dmacrys_out.glob("*.dmaout"))
    if not dmaout_files:
        raise FileNotFoundError(f"No .dmaout file in {dmacrys_out}")
    dmaout_file = dmaout_files[0]

    # Parse everything
    cell, Z, symops, atoms = parse_res_file(res_file)
    sites = parse_dma_file(dma_file)
    all_pairs = parse_fit_pots(pots_file)
    initial_ref, optimized_ref = parse_dmaout(dmaout_file)

    # Filter Buckingham pairs to only atom types in this molecule
    atom_types_present = set(s["atom_type"] for s in sites)
    pairs = filter_relevant_pairs(all_pairs, atom_types_present)

    # Strip internal keys from output
    for p in pairs:
        del p["type1"]
        del p["type2"]
    for s in sites:
        del s["atom_type"]

    # Extract title from directory name
    # e.g., "01.lem_fit_AXOSOW" -> "AXOSOW"
    title_parts = name.split("_")
    title = title_parts[-1] if title_parts else name

    # Determine space group from symops
    # P 21 21 21 has 3 symmetry operations (plus identity)
    space_group = infer_space_group(symops)

    result = {
        "title": title,
        "source": name,
        "crystal": {
            "cell": cell,
            "space_group": space_group,
            "Z": Z,
            "atoms": atoms,
            "symmetry_operations": symops,
        },
        "molecule": {
            "comment": f"Body-frame multipoles from dmacrys.dma (one molecule type, MOLX 1)",
            "sites": sites,
        },
        "potentials": {
            "type": "buckingham",
            "comment": "V = A*exp(-r/rho) - C/r^6, units: eV and Angstrom",
            "pairs": pairs,
        },
        "settings": {
            "repulsion_dispersion_cutoff_angstrom": 15.0,
        },
        "reference": {
            "initial": initial_ref,
            "optimized": optimized_ref,
        },
    }

    return result


def infer_space_group(symops):
    """Return empty string; let C++ determine space group from symops.

    The SHELX LATT directive has already been applied to expand the symop
    list, so the C++ side (via gemmi) can identify the space group from
    the complete set of symmetry operations.
    """
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Convert DMACRYS regression test to JSON"
    )
    parser.add_argument("directory", help="DMACRYS regression test directory")
    parser.add_argument("-o", "--output", help="Output JSON file (default: <title>.json)")
    args = parser.parse_args()

    result = convert_dmacrys_directory(args.directory)

    output_path = args.output
    if not output_path:
        output_path = f"{result['title']}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {output_path}")
    print(f"  Title: {result['title']}")
    print(f"  Space group: {result['crystal']['space_group']}")
    print(f"  Z = {result['crystal']['Z']}")
    print(f"  Atoms: {len(result['crystal']['atoms'])}")
    print(f"  Multipole sites: {len(result['molecule']['sites'])}")
    print(f"  Buckingham pairs: {len(result['potentials']['pairs'])}")
    if result["reference"]["initial"]:
        ref = result["reference"]["initial"]
        print(f"  Initial energy: {ref.get('total_kJ_per_mol', 'N/A')} kJ/mol")
        print(f"  Rep-disp: {ref.get('repulsion_dispersion_eV', 'N/A')} eV/cell")


if __name__ == "__main__":
    main()
