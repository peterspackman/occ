#!/usr/bin/env python3
"""Convert a DMACRYS regression test directory to a single JSON file.

Reads:
  NEIGHCRYS_input/*.res       - SHELX crystal structure
  NEIGHCRYS_input/dmacrys.dma - Body-frame DMA multipoles (Bohr, Stone convention)
  NEIGHCRYS_input/{fit.pots,will01.pots,pote.dat,pote_same_as_CP.dat}
                              - Buckingham/DBUC parameters (eV, Angstrom)
  DMACRYS_input/*.dmain       - cutoff/SPLI settings
  DMACRYS_output/*.dmaout     - Reference energy breakdown

Outputs:
  A JSON file suitable for loading into OCC via --multipole-json.
"""

import argparse
import json
import re
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
            # e.g., "ClF1_2____" -> element="Cl", atom_type="ClF1"
            label_parts = label.rstrip("_").split("_")
            element = infer_element_symbol(label_parts[0])
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

def parse_potential_file(path):
    """Parse fit.pots/will01.pots/pote.dat for Buckingham-like parameters.

    Preserves DMACRYS atom type names (e.g., "H_F1", "H_F2") for proper filtering.
    """
    pairs = []

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()
        if upper.startswith("BUCK") or upper.startswith("DBUC"):
            parts = line.split()
            # BUCK/DBUC  TYPE1  TYPE2
            if len(parts) < 3:
                i += 1
                continue
            type1 = parts[1]
            type2 = parts[2]
            el1 = infer_element_symbol(type1.split("_")[0])
            el2 = infer_element_symbol(type2.split("_")[0])

            # Next numeric line: A rho C6 [extra...]
            i += 1
            while i < len(lines):
                params_line = lines[i].strip()
                if not params_line or params_line.startswith("!"):
                    i += 1
                    continue
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", params_line)
                if len(nums) >= 3:
                    A = float(nums[0].replace("D", "E").replace("d", "E"))
                    rho = float(nums[1].replace("D", "E").replace("d", "E"))
                    C6 = float(nums[2].replace("D", "E").replace("d", "E"))
                    break
                i += 1
            else:
                raise ValueError(f"Could not parse parameters for potential pair {type1}-{type2} in {path}")

            pairs.append({
                "type1": type1,
                "type2": type2,
                "el1": el1,
                "el2": el2,
                "A_eV": A,
                "rho_ang": rho,
                "C6_eV_ang6": C6,
                "kind": "DBUC" if upper.startswith("DBUC") else "BUCK",
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
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
    }
    el = infer_element_symbol(element)
    if el in table:
        return table[el]
    raise ValueError(f"Unknown element: {element}")


def infer_element_symbol(token):
    """Infer element symbol from DMACRYS atom type prefixes.

    Examples:
      C_F1 -> C
      ClF1 -> Cl
      BrBR -> Br
      N_Ab -> N
    """
    tok = token.strip()
    if not tok:
        raise ValueError("Empty element token")
    symbols = [
        "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca",
        "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
        "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc", "Ru",
        "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
        "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
        "Mc", "Lv", "Ts", "Og",
        "H", "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "I", "W", "U",
    ]
    low = tok.lower()
    for sym in symbols:
        if low.startswith(sym.lower()):
            return sym
    # Conservative fallback: first alphabetic tokenized symbol.
    m = re.match(r"([A-Za-z]{1,2})", tok)
    if m:
        guess = m.group(1)
        return guess[0].upper() + (guess[1:].lower() if len(guess) > 1 else "")
    raise ValueError(f"Could not infer element from token: {token}")


def filter_relevant_pairs(pairs, atom_types_present):
    """Keep only Buckingham pairs for atom types actually in the molecule.

    atom_types_present: set of DMACRYS type prefixes, e.g. {"C_F1", "O_F1", "H_F1"}
    """
    return [
        p for p in pairs
        if p["type1"] in atom_types_present and p["type2"] in atom_types_present
    ]


def parse_dmain_settings(path):
    """Parse CUTO/SPLI from DMACRYS .dmain file.

    CUTO usually appears as: CUTO <c_mag_ang> <rcut_lattice>.
    Physical cutoff is c_mag_ang * rcut_lattice (Angstrom).
    """
    cutoff_angstrom = None
    spline_min = None
    spline_max = None
    pressure_value = None
    pressure_units = "Pa"

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("CUTO"):
                vals = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", line)
                nums = [float(v.replace("D", "E").replace("d", "E")) for v in vals]
                if len(nums) >= 2:
                    cutoff_angstrom = nums[0] * nums[1]
                elif len(nums) == 1:
                    cutoff_angstrom = nums[0]
            elif upper.startswith("SPLI"):
                vals = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", line)
                nums = [float(v.replace("D", "E").replace("d", "E")) for v in vals]
                if len(nums) >= 2:
                    spline_min, spline_max = nums[0], nums[1]
                elif len(nums) == 1:
                    spline_min = nums[0]
                    spline_max = nums[0]
            elif upper.startswith("PRES"):
                parts = line.split()
                if len(parts) >= 2:
                    pressure_value = float(parts[1].replace("D", "E").replace("d", "E"))
                if len(parts) >= 3:
                    pressure_units = parts[2]

    return cutoff_angstrom, spline_min, spline_max, pressure_value, pressure_units


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
        dma_candidates = sorted(
            p for p in neighcrys.glob("*.dma")
            if not p.name.lower().endswith(".dma.pol")
        )
        if not dma_candidates:
            raise FileNotFoundError(f"No .dma file in {neighcrys}")
        dma_file = dma_candidates[0]

    pot_candidates = [
        neighcrys / "fit.pots",
        neighcrys / "will01.pots",
        neighcrys / "pote.dat",
        neighcrys / "pote_same_as_CP.dat",
    ]
    pots_file = next((p for p in pot_candidates if p.exists()), None)
    if pots_file is None:
        extra = sorted(list(neighcrys.glob("*.pots")) + list(neighcrys.glob("pote*.dat")))
        if extra:
            pots_file = extra[0]
        else:
            raise FileNotFoundError(f"No potential file found in {neighcrys}")

    dmain_candidates = sorted((directory / "DMACRYS_input").glob("*.dmain"))
    dmain_file = dmain_candidates[0] if dmain_candidates else None

    dmaout_files = list(dmacrys_out.glob("*.dmaout"))
    if not dmaout_files:
        raise FileNotFoundError(f"No .dmaout file in {dmacrys_out}")
    dmaout_file = dmaout_files[0]

    # Parse everything
    cell, Z, symops, atoms = parse_res_file(res_file)
    sites = parse_dma_file(dma_file)
    all_pairs = parse_potential_file(pots_file)
    initial_ref, optimized_ref = parse_dmaout(dmaout_file)
    cutoff_angstrom, spline_min, spline_max, pressure_value, pressure_units = (None, None, None, None, "Pa")
    if dmain_file is not None:
        cutoff_angstrom, spline_min, spline_max, pressure_value, pressure_units = parse_dmain_settings(dmain_file)

    # Filter Buckingham pairs to only atom types in this molecule
    atom_types_present = set(s["atom_type"] for s in sites)
    pairs = filter_relevant_pairs(all_pairs, atom_types_present)
    has_dbuc = any(p.get("kind") == "DBUC" for p in pairs)

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
        "settings": {},
        "reference": {
            "initial": initial_ref,
            "optimized": optimized_ref,
        },
    }

    if cutoff_angstrom is not None:
        result["settings"]["repulsion_dispersion_cutoff_angstrom"] = cutoff_angstrom
    else:
        result["settings"]["repulsion_dispersion_cutoff_angstrom"] = 15.0
    result["settings"]["short_range_potential_kind"] = (
        "damped_buckingham" if has_dbuc else "buckingham"
    )
    result["settings"]["potential_source_file"] = pots_file.name
    if pressure_value is not None:
        result["settings"]["pressure"] = {
            "value": pressure_value,
            "units": pressure_units,
        }
    if spline_min is not None and spline_max is not None:
        result["settings"]["spline"] = {"min": spline_min, "max": spline_max}

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
