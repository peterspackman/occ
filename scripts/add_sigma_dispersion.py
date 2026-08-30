"""Add the COSMO-SAC-dsp fields to the `# meta:` header of a .sigma profile.

The dispersion parameter depends only on the geometry, so profiles generated
before the term existed can be updated in place without redoing the SCF: the
psigmaA rows are left byte-identical and only the header is rewritten.

Profiles written by `occ sigma` carry these fields already. The values here
must agree with `occ::solvent::sigma::dispersion_parameters`; the
`[dispersion]` tests in solvent_tests.cpp check that they do.

Usage: python scripts/add_sigma_dispersion.py <dir-with-matching .sigma/.xyz>
"""
import json
import math
import pathlib
import sys

# Hsieh, Lin & Vrabec, Fluid Phase Equilib. 367 (2014) 72, Table 3, in K.
LIB = {
    "C(sp3)": 115.7023, "C(sp2)": 117.4650, "C(sp)": 66.0691,
    "-O-": 95.6184, "=O": -11.0549,
    "N(sp3)": 15.4901, "N(sp2)": 84.6268, "N(sp)": 109.6621,
    "F": 52.9318, "Cl": 104.2534,
    "H(OH)": 19.3477, "H(NH)": 141.1709, "H(water)": 58.3301,
}

# Covalent radii (A) for the elements the term is parameterised for, matching
# occ::core::Element; bonded when d < r_i + r_j + 0.4.
RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Cl": 1.02}
TOLERANCE = 0.4


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + n]:
        f = line.split()
        atoms.append((f[0], tuple(float(x) for x in f[1:4])))
    return atoms


def bonds_of(atoms):
    n = len(atoms)
    out = [[] for _ in range(n)]
    if n == 2:
        return [[1], [0]]
    for i in range(n):
        for j in range(i + 1, n):
            threshold = RCOV[atoms[i][0]] + RCOV[atoms[j][0]] + TOLERANCE
            if math.dist(atoms[i][1], atoms[j][1]) < threshold:
                out[i].append(j)
                out[j].append(i)
    return out


def is_carboxyl_carbon(i, atoms, bonds):
    if len(bonds[i]) != 3:
        return False
    if sum(atoms[j][0] == "O" for j in bonds[i]) != 2:
        return False
    for j in bonds[i]:
        if atoms[j][0] != "O" or len(bonds[j]) != 2:
            continue
        neighbours = sorted(atoms[k][0] for k in bonds[j])
        if neighbours == ["C", "H"]:
            return True
    return False


def dispersion(atoms):
    bonds = bonds_of(atoms)
    elements = [a[0] for a in atoms]
    water = len(atoms) == 3 and elements.count("H") == 2 and elements.count("O") == 1

    total, typed, carboxyl = 0.0, 0, False
    for i, (element, _) in enumerate(atoms):
        degree = len(bonds[i])
        if element not in RCOV:
            raise ValueError(f"no dispersion parameter for element {element}")
        if element == "N" and degree not in (1, 2, 3):
            raise ValueError(f"nitrogen with {degree} bonds has no type")
        if element == "O" and degree not in (1, 2):
            raise ValueError(f"oxygen with {degree} bonds has no type")
        if element == "C" and is_carboxyl_carbon(i, atoms, bonds):
            carboxyl = True

        value = None
        if element == "C":
            value = {4: LIB["C(sp3)"], 3: LIB["C(sp2)"], 2: LIB["C(sp)"]}.get(degree)
        elif element == "N":
            value = {3: LIB["N(sp3)"], 2: LIB["N(sp2)"], 1: LIB["N(sp)"]}[degree]
        elif element == "O":
            value = LIB["-O-"] if degree == 2 else LIB["=O"]
        elif element == "F":
            value = LIB["F"]
        elif element == "Cl":
            value = LIB["Cl"]
        elif element == "H":
            neighbours = [atoms[j][0] for j in bonds[i]]
            if water:
                value = LIB["H(water)"]
            elif "O" in neighbours:
                value = LIB["H(OH)"]
            elif "N" in neighbours:
                value = LIB["H(NH)"]
        if value is not None:
            total += value
            typed += 1

    if typed == 0:
        raise ValueError("no typed atoms")

    if water:
        klass = "H2O"
    elif carboxyl:
        klass = "COOH"
    else:
        heteroatoms = [i for i, (e, _) in enumerate(atoms) if e in ("O", "N", "F")]
        if not heteroatoms:
            klass = "NHB"
        elif any(atoms[j][0] == "H" for i in heteroatoms for j in bonds[i]):
            klass = "HB-DONOR-ACCEPTOR"
        else:
            klass = "HB-ACCEPTOR"
    return total / typed, klass


def patch(sigma_path, xyz_path):
    lines = sigma_path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        marker = line.find("# meta:")
        if marker == -1:
            continue
        meta = json.loads(line[marker + 7:])
        epsilon, klass = dispersion(read_xyz(xyz_path))
        meta["dispersion e/kB [K]"] = epsilon
        meta["dispersion class"] = klass
        lines[index] = "# meta: " + json.dumps(meta, sort_keys=True) + "\n"
        sigma_path.write_text("".join(lines))
        return epsilon, klass
    raise ValueError(f"{sigma_path} has no '# meta:' header")


if __name__ == "__main__":
    directory = pathlib.Path(sys.argv[1])
    for sigma_path in sorted(directory.glob("*.sigma")):
        xyz_path = sigma_path.with_suffix(".xyz")
        if not xyz_path.exists():
            print(f"{sigma_path.name}: no matching .xyz, skipped")
            continue
        epsilon, klass = patch(sigma_path, xyz_path)
        print(f"{sigma_path.name}: e/kB = {epsilon:.4f} K, class {klass}")
