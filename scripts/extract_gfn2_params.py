#!/usr/bin/env python3
"""Extract GFN2-xTB parameters from Grimme's xtb source tree to JSON.

Reads:
  - $XTB_SRC/param_gfn2-xtb.txt
  - $XTB_SRC/src/xtb/gfn2.f90       (referenceOcc table; sanity checks)
  - $XTB_SRC/src/param/paulingen.f90 (Pauling EN, used for H0 EN-shift)

Writes:
  - share/xtb/gfn2.json

Schema:
{
  "method": "GFN2-xTB",
  "doi": "...",
  "version": 1,
  "max_z": 86,
  "globals": { ks, kp, kd, kf, ksd, kpd, kdiff, ipeashift_au,
               enscale, enscale4,
               gam3shell: [[s1,s2],[p1,p2],[d1,d2],[f1,f2]],
               aesshift, aesexp, aesrmax,
               alphaj, a1, a2, s8, s9, aesdmp3, aesdmp5,
               kexp, kexplight },
  "elements": [
     { "z": 1, "ao": "1s", "pauling_en": 2.20,
       "atomic_hardness": ..., "third_order_atom_au": ...,
       "rep_alpha": ..., "rep_zeff": ...,
       "dip_kernel": ..., "quad_kernel": ...,
       "shells": [
         {"n": 1, "l": 0, "n_prim": 3, "is_valence": true,
          "self_energy_ev": -10.707211,
          "slater_exponent": 1.230000,
          "kcn_au": -0.05,
          "shell_poly": -0.953618,
          "ref_occ": 1.0,
          "shell_hardness_au": 0.0}
       ]
     }, ...
  ]
}

Per-key scalings applied here (matches xtb's read_gfn_param.f90):
  KCNS/KCNP/KCND  : x 0.1   (kcn_au)
  GAM3            : x 0.1   (third_order_atom_au)
  DPOL            : x 0.01  (dip_kernel)
  QPOL            : x 0.01  (quad_kernel)
  LPARS/P/D       : x 0.1   (shell_hardness_au)
  ipeashift       : x 0.1   (globals.ipeashift_au)

Run:
    python3 scripts/extract_gfn2_params.py [--xtb-src ~/git/xtb] \
        [--out share/xtb/gfn2.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded supporting tables scraped once from xtb source.
# These are NOT in param_gfn2-xtb.txt; they are Fortran-level constants in
# src/xtb/gfn2.f90 / src/param/paulingen.f90. We bake them in so we don't
# have to re-parse the Fortran modules at every run.
# ---------------------------------------------------------------------------

PAULING_EN = [  # Z = 1 .. 86 (xtb has 1..118 but GFN2 only uses 86)
    2.20, 3.00,
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 4.50,
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 3.50,
    0.82, 1.00,
    1.36, 1.54, 1.63, 1.66, 1.55,
    1.83, 1.88, 1.91, 1.90, 1.65,
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00,
    0.82, 0.95,
    1.22, 1.33, 1.60, 2.16, 1.90,
    2.20, 2.28, 2.20, 1.93, 1.69,
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60,
    0.79, 0.89,
    1.10, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18,
    1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26,
    1.27, 1.30, 1.50, 2.36, 1.90,
    2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20,
]
assert len(PAULING_EN) == 86

# referenceOcc(0:2, 1:86) from gfn2.f90 — atomic occupations for s,p,d shells.
# Columns are [s_occ, p_occ, d_occ]; only valence shells get them.
REFERENCE_OCC = [
    # (s, p, d) per element, Z=1..86
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),  # H, He
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 3.0, 0.0),
    (1.5, 3.5, 0.0), (2.0, 4.0, 0.0), (2.0, 5.0, 0.0), (2.0, 6.0, 0.0),
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.5, 2.5, 0.0),
    (1.5, 3.5, 0.0), (2.0, 4.0, 0.0), (2.0, 5.0, 0.0), (2.0, 6.0, 0.0),
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 1.0, 3.0), (1.0, 1.0, 4.0),
    (1.0, 1.0, 5.0), (1.0, 1.0, 6.0), (1.0, 1.0, 7.0), (1.0, 1.0, 8.0),
    (1.0, 0.0, 10.0), (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0), (1.5, 2.5, 0.0), (1.5, 3.5, 0.0), (2.0, 4.0, 0.0),
    (2.0, 5.0, 0.0), (2.0, 6.0, 0.0),
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 1.0, 3.0), (1.0, 1.0, 4.0),
    (1.0, 1.0, 5.0), (1.0, 1.0, 6.0), (1.0, 1.0, 7.0), (1.0, 1.0, 8.0),
    (1.0, 0.0, 10.0), (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0), (2.0, 2.0, 0.0), (2.0, 3.0, 0.0), (2.0, 4.0, 0.0),
    (2.0, 5.0, 0.0), (2.0, 6.0, 0.0),
    (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),  # La
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),  # Ce-Eu
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),  # Gd-Yb
    (1.0, 1.0, 1.0),  # Lu
    (1.0, 1.0, 2.0), (1.0, 1.0, 3.0), (1.0, 1.0, 4.0), (1.0, 1.0, 5.0),
    (1.0, 1.0, 6.0), (1.0, 1.0, 7.0), (1.0, 1.0, 8.0),
    (1.0, 0.0, 10.0), (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0), (2.0, 2.0, 0.0), (2.0, 3.0, 0.0), (2.0, 4.0, 0.0),
    (2.0, 5.0, 0.0), (2.0, 6.0, 0.0),
]
assert len(REFERENCE_OCC) == 86, f"got {len(REFERENCE_OCC)} entries"

# ---------------------------------------------------------------------------
# Per-key scaling factors as applied by xtb/src/read_gfn_param.f90 when
# loading the .txt file. Multiply the raw text value by this factor to get
# the internal (atomic-units, where applicable) value.
# ---------------------------------------------------------------------------

ELEM_SCALE = {
    "kcns": 0.1, "kcnp": 0.1, "kcnd": 0.1,
    "gam3": 0.1,
    "dpol": 0.01, "qpol": 0.01,
    "lpars": 0.1, "lparp": 0.1, "lpard": 0.1, "lparf": 0.1,
    # everything else: 1.0
}
GLOBAL_SCALE = {
    "ipeashift": 0.1,
    # everything else: 1.0
}

# ---------------------------------------------------------------------------
# Shell-metadata helpers
# ---------------------------------------------------------------------------

L_OF_SYMBOL = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def parse_ao_string(ao: str) -> list[tuple[int, int]]:
    """'2s2p3d' -> [(2,0),(2,1),(3,2)]  (n, l) per shell."""
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < len(ao):
        if ao[i].isdigit():
            n = int(ao[i])
            sym = ao[i + 1].lower()
            pairs.append((n, L_OF_SYMBOL[sym]))
            i += 2
        else:
            raise ValueError(f"Cannot parse ao string {ao!r} at offset {i}")
    return pairs


def number_of_primitives(z: int, n: int, l: int) -> int:
    """xtb's setGFN2NumberOfPrimitives rule (gfn2.f90)."""
    if z <= 2:
        return {0: 3, 1: 4}[l]
    if l == 0 or l == 1:
        return 6 if n > 5 else 4
    if l == 2:
        return 3
    if l == 3:
        return 4
    raise ValueError(f"unsupported (z={z},n={n},l={l})")


def is_valence_shell(z: int, n: int, l: int, all_shells: list[tuple[int, int]]) -> bool:
    """xtb's generateValenceShellData. The rule: a shell is valence if it is
    the lowest-n shell of its angular momentum present for the element.
    Otherwise it is a 'diffuse polarization' shell with zero ref. occupation."""
    same_l = [pair for pair in all_shells if pair[1] == l]
    return min(p[0] for p in same_l) == n


def reference_occupation(z: int, n: int, l: int, all_shells: list[tuple[int, int]]) -> float:
    if not is_valence_shell(z, n, l, all_shells):
        return 0.0
    if l > 2:
        return 0.0
    return REFERENCE_OCC[z - 1][l]


# ---------------------------------------------------------------------------
# .txt parsing
# ---------------------------------------------------------------------------

GLOBAL_KEY_MAP = {
    # raw key -> (json key, list-position-or-None)
    "ks": ("ks", None), "kp": ("kp", None), "kd": ("kd", None), "kf": ("kf", None),
    "ksp": ("ksp", None), "ksd": ("ksd", None), "kpd": ("kpd", None),
    "kdiff": ("kdiff", None),
    "enscale": ("enscale", None), "enscale4": ("enscale4", None),
    "ipeashift": ("ipeashift_au", None),
    "aesshift": ("aesshift", None), "aesexp": ("aesexp", None),
    "aesrmax": ("aesrmax", None),
    "alphaj": ("alphaj", None),
    "a1": ("a1", None), "a2": ("a2", None),
    "s6": ("s6", None), "s8": ("s8", None), "s9": ("s9", None),
    "aesdmp3": ("aesdmp3", None), "aesdmp5": ("aesdmp5", None),
    "kexp": ("kexp", None), "kexplight": ("kexplight", None),
}

# gam3{s,p,d1,d2} populate the gam3shell[l][which] table. xtb's reader puts
# scalar values into arrays of length 2: gam3s -> [s,s], gam3p -> [p,p],
# gam3d1 -> [d,_], gam3d2 -> [_,d], gam3f -> [f,f].
GAM3_KEY = {"gam3s": (0, "both"), "gam3p": (1, "both"),
            "gam3d": (2, "both"), "gam3d1": (2, 0), "gam3d2": (2, 1),
            "gam3f": (3, "both")}


def parse_param_file(path: Path) -> dict:
    text = path.read_text()
    info: dict = {}
    globals_: dict = {}
    gam3shell = [[0.0, 0.0] for _ in range(4)]
    elements: list[dict | None] = [None] * 86

    # split into sections by leading $
    sections = re.split(r"^\$", text, flags=re.M)
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        head, _, body = sec.partition("\n")
        head = head.strip()
        if head == "info" or head.startswith("info"):
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                k, _, v = line.partition(" ")
                info[k.strip()] = v.strip()
        elif head.startswith("globpar"):
            for line in body.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition(" ")
                k = k.strip().lower()
                v = v.strip()
                if not v:
                    continue
                try:
                    val = float(v.split()[0])
                except ValueError:
                    continue
                if k in GLOBAL_SCALE:
                    val *= GLOBAL_SCALE[k]
                if k in GLOBAL_KEY_MAP:
                    name, _ = GLOBAL_KEY_MAP[k]
                    globals_[name] = val
                elif k in GAM3_KEY:
                    li, which = GAM3_KEY[k]
                    if which == "both":
                        gam3shell[li] = [val, val]
                    else:
                        gam3shell[li][which] = val
                # silently drop unknown globals (e.g. xbdamp/xbrad GFN1-only)
        elif head.startswith("pairpar"):
            # GFN2 leaves this empty (kPair defaults to 1.0 everywhere).
            continue
        elif head.startswith("Z="):
            m = re.match(r"Z=\s*(\d+)", head)
            if not m:
                continue
            z = int(m.group(1))
            if z < 1 or z > 86:
                continue
            elements[z - 1] = parse_element_block(z, body)
        elif head == "end":
            continue

    globals_["gam3shell"] = gam3shell

    return {
        "info": info,
        "globals": globals_,
        "elements": [e for e in elements if e is not None],
    }


def parse_element_block(z: int, body: str) -> dict:
    raw: dict[str, list[float] | str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ao="):
            raw["ao"] = line[3:].strip()
            continue
        m = re.match(r"([A-Za-z][A-Za-z0-9]*)=\s*(.*)", line)
        if not m:
            continue
        key = m.group(1).lower()
        rest = m.group(2).strip()
        if not rest:
            continue
        try:
            vals = [float(tok) for tok in rest.split()]
        except ValueError:
            continue
        raw[key] = vals

    ao = raw.get("ao", "")
    if not isinstance(ao, str) or not ao:
        raise ValueError(f"Z={z}: missing ao= line")
    shell_pairs = parse_ao_string(ao)
    n_shells = len(shell_pairs)

    lev = raw.get("lev", [])
    exp = raw.get("exp", [])
    if len(lev) < n_shells or len(exp) < n_shells:
        raise ValueError(
            f"Z={z}: lev/exp length mismatch (have {len(lev)}/{len(exp)}, need {n_shells})"
        )

    # KCN per angular momentum (xtb stores per-l, not per-shell).
    kcn_by_l = [0.0, 0.0, 0.0, 0.0]
    if "kcns" in raw: kcn_by_l[0] = raw["kcns"][0] * ELEM_SCALE["kcns"]
    if "kcnp" in raw: kcn_by_l[1] = raw["kcnp"][0] * ELEM_SCALE["kcnp"]
    if "kcnd" in raw: kcn_by_l[2] = raw["kcnd"][0] * ELEM_SCALE["kcnd"]

    # Shell polynomials per angular momentum.
    poly_by_l = [0.0, 0.0, 0.0, 0.0]
    if "polys" in raw: poly_by_l[0] = raw["polys"][0]
    if "polyp" in raw: poly_by_l[1] = raw["polyp"][0]
    if "polyd" in raw: poly_by_l[2] = raw["polyd"][0]
    if "polyf" in raw: poly_by_l[3] = raw["polyf"][0]

    # Shell hardness per angular momentum.
    lpar_by_l = [0.0, 0.0, 0.0, 0.0]
    if "lpars" in raw: lpar_by_l[0] = raw["lpars"][0] * ELEM_SCALE["lpars"]
    if "lparp" in raw: lpar_by_l[1] = raw["lparp"][0] * ELEM_SCALE["lparp"]
    if "lpard" in raw: lpar_by_l[2] = raw["lpard"][0] * ELEM_SCALE["lpard"]
    if "lparf" in raw: lpar_by_l[3] = raw["lparf"][0] * ELEM_SCALE["lparf"]

    shells = []
    for idx, (n, l) in enumerate(shell_pairs):
        shells.append({
            "n": n,
            "l": l,
            "n_prim": number_of_primitives(z, n, l),
            "is_valence": is_valence_shell(z, n, l, shell_pairs),
            "self_energy_ev": lev[idx],
            "slater_exponent": exp[idx],
            "kcn_au": kcn_by_l[l],
            "shell_poly": poly_by_l[l],
            "ref_occ": reference_occupation(z, n, l, shell_pairs),
            "shell_hardness_au": lpar_by_l[l],
        })

    elem: dict = {
        "z": z,
        "ao": ao,
        "pauling_en": PAULING_EN[z - 1],
        "atomic_hardness": raw.get("gam", [0.0])[0],
        "third_order_atom_au": raw.get("gam3", [0.0])[0] * ELEM_SCALE["gam3"],
        "rep_alpha": raw.get("repa", [0.0])[0],
        "rep_zeff": raw.get("repb", [0.0])[0],
        "dip_kernel": raw.get("dpol", [0.0])[0] * ELEM_SCALE["dpol"],
        "quad_kernel": raw.get("qpol", [0.0])[0] * ELEM_SCALE["qpol"],
        "shells": shells,
    }
    return elem


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xtb-src", default=os.path.expanduser("~/git/xtb"))
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "share" / "xtb" / "gfn2.json"
    p.add_argument("--out", default=str(default_out))
    args = p.parse_args(argv)

    xtb_src = Path(args.xtb_src)
    param = xtb_src / "param_gfn2-xtb.txt"
    if not param.is_file():
        print(f"missing: {param}", file=sys.stderr)
        return 1

    data = parse_param_file(param)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "method": data["info"].get("name", "GFN2-xTB"),
        "doi": data["info"].get("doi", "10.1021/acs.jctc.8b01176"),
        "version": 1,
        "max_z": 86,
        "globals": data["globals"],
        "elements": data["elements"],
    }

    with out.open("w") as fh:
        json.dump(bundle, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out} ({len(data['elements'])} elements)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
