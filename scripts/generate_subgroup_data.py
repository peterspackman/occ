#!/usr/bin/env python3
"""Generate src/crystal/subgroup_data.cpp from the Bilbao maximal-subgroup tables.

The group-subgroup relations of the 230 space groups originate from the
International Tables for Crystallography Vol. A1 and are published by the Bilbao
Crystallographic Server (MAXSUB). We take them via PyXtal, which redistributes
them as JSON under the MIT licence.

    PyXtal            https://github.com/MaterSim/PyXtal        (MIT)
    Bilbao MAXSUB     https://cryst.ehu.es/cryst/subgroups.html

Cite: Aroyo et al., "Bilbao Crystallographic Server I", Z. Kristallogr. 221,
15-27 (2006).

We keep only the fields describing the graph -- the subgroup's space group
number, the index, whether the relation is translationengleiche ("t") or
klassengleiche ("k"), and the 3x4 transformation [P | p] to the subgroup's
standard setting. PyXtal's `cosets` and `relations` (Wyckoff splitting) fields
are dropped: both are recomputable from the parent and subgroup symmetry
operations, and together they account for ~95% of the 11 MB of source JSON.

Two corrections are applied, see PATCHES below.

Usage:
    python scripts/generate_subgroup_data.py [-o src/crystal/subgroup_data.cpp]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

# Pinned so regeneration is reproducible.
PYXTAL_COMMIT = "8c5ff94575edca4f9927e403789b67d4bde55dbe"
BASE_URL = f"https://raw.githubusercontent.com/MaterSim/PyXtal/{PYXTAL_COMMIT}/pyxtal/database"

# Corrections to the upstream tables.
#
# SG 97 (I422) is missing a maximal t-subgroup. Its point group 422 = D4 has
# three pairwise non-conjugate maximal subgroups (order 4): the cyclic C4, the
# axial D2 with 2-folds along [100]/[010], and the *diagonal* D2 with 2-folds
# along [110]/[1-10]. PyXtal lists only the first two (I4 #79 and I222 #23).
#
# The diagonal D2 spans a cell rotated 45 degrees, which in a body-centred
# tetragonal lattice is F-centred, giving F222 (#22) with a' = a - b, b' = a + b,
# c' = c and no origin shift. Verified against gemmi: transforming those symmetry
# operations by that basis change yields exactly the 16 operations of F222.
PATCHES: dict[int, list[dict]] = {
    97: [
        {
            "subgroup": 22,
            "index": 2,
            "type": "t",
            # rows of [P | p]; columns of P are the new basis vectors in terms
            # of the old, i.e. a' = a - b, b' = a + b, c' = c
            "transformation": [
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    ],
}

# The transformation entries are exact rationals; these are all the denominators
# that occur (matching occ's duodecimal symmetry-operation convention).
MAX_DENOMINATOR = 12


def unwrap_mson(x):
    """PyXtal serializes numpy arrays via monty; unwrap to plain nested lists."""
    if isinstance(x, dict) and x.get("@class") == "array":
        return x["data"]
    if isinstance(x, dict):
        return {k: unwrap_mson(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unwrap_mson(v) for v in x]
    return x


def fetch(name: str) -> dict:
    url = f"{BASE_URL}/{name}.json"
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response:
        return unwrap_mson(json.load(response))


def edges_from(table: dict, kind: str) -> dict[int, list[dict]]:
    """Flatten PyXtal's parallel arrays into a list of edges per parent."""
    result: dict[int, list[dict]] = {}
    for parent_str, entry in table.items():
        if not entry:
            continue
        parent = int(parent_str)
        edges = []
        for subgroup, index, type_, transformation in zip(
            entry["subgroup"],
            entry["index"],
            entry["type"],
            entry["transformation"],
        ):
            assert type_ == kind, f"expected {kind!r} relation, got {type_!r}"
            edges.append(
                {
                    "subgroup": int(subgroup),
                    "index": int(index),
                    "type": type_,
                    "transformation": transformation,
                }
            )
        result[parent] = edges
    return result


def encode_transformation(transformation) -> list[int]:
    """Encode the 3x4 [P | p] matrix as 12 (numerator, denominator) byte pairs."""
    out: list[int] = []
    for row in transformation:
        assert len(row) == 4, "expected a 3x4 transformation"
        for value in row:
            f = Fraction(value).limit_denominator(MAX_DENOMINATOR)
            if abs(float(f) - value) > 1e-9:
                raise ValueError(f"{value} is not a rational with denominator <= {MAX_DENOMINATOR}")
            num, den = f.numerator, f.denominator
            if not (-128 <= num <= 127):
                raise ValueError(f"numerator {num} does not fit in an int8")
            if not (0 < den <= 255):
                raise ValueError(f"denominator {den} does not fit in a uint8")
            out.append(num & 0xFF)  # two's complement, decoded as int8
            out.append(den)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src/crystal/subgroup_data.cpp",
    )
    args = parser.parse_args()

    t_edges = edges_from(fetch("t_subgroup"), "t")
    k_edges = edges_from(fetch("k_subgroup"), "k")

    by_parent: dict[int, list[dict]] = {}
    for parent in range(1, 231):
        by_parent[parent] = t_edges.get(parent, []) + k_edges.get(parent, [])

    for parent, patches in PATCHES.items():
        existing = {(e["subgroup"], e["index"], e["type"]) for e in by_parent[parent]}
        for patch in patches:
            key = (patch["subgroup"], patch["index"], patch["type"])
            if key in existing:
                print(
                    f"note: patch for SG {parent} -> {key} already present upstream; "
                    "the table may have been fixed, review PATCHES",
                    file=sys.stderr,
                )
                continue
            # keep t-relations before k-relations
            insert_at = sum(1 for e in by_parent[parent] if e["type"] == "t")
            by_parent[parent].insert(insert_at, patch)
            print(f"patched: SG {parent} -> subgroup {key[0]} (index {key[1]}, {key[2]})", file=sys.stderr)

    # flatten, and build the per-parent [begin, end) offsets
    blob: list[int] = []
    offsets: list[int] = []
    n_edges = 0
    n_t = n_k = 0
    for parent in range(1, 231):
        offsets.append(n_edges)
        for edge in by_parent[parent]:
            blob.append(edge["subgroup"])
            blob.append(edge["index"])
            blob.append(0 if edge["type"] == "t" else 1)
            blob.extend(encode_transformation(edge["transformation"]))
            n_edges += 1
            if edge["type"] == "t":
                n_t += 1
            else:
                n_k += 1
    offsets.append(n_edges)

    stride = 3 + 24
    assert len(blob) == n_edges * stride

    lines = []
    lines.append("// GENERATED FILE -- do not edit by hand.")
    lines.append("// Regenerate with: python scripts/generate_subgroup_data.py")
    lines.append("//")
    lines.append("// Maximal subgroups of the 230 space groups.")
    lines.append("//")
    lines.append("// Data originates from the International Tables for Crystallography Vol. A1")
    lines.append("// and is published by the Bilbao Crystallographic Server (MAXSUB). Obtained")
    lines.append("// via PyXtal (https://github.com/MaterSim/PyXtal), which redistributes it")
    lines.append("// under the MIT licence.")
    lines.append("//")
    lines.append("//   Copyright 2018 Scott Fredericks, Qiang Zhu (PyXtal, MIT)")
    lines.append("//")
    lines.append("// Cite: Aroyo et al., Z. Kristallogr. 221, 15-27 (2006).")
    lines.append("//")
    lines.append(f"// PyXtal commit: {PYXTAL_COMMIT}")
    lines.append(f"// {n_edges} edges ({n_t} translationengleiche, {n_k} klassengleiche).")
    lines.append("//")
    lines.append("// Known limits of the upstream data:")
    lines.append("//  - klassengleiche relations are truncated at index 9. Maximal *isomorphic*")
    lines.append("//    subgroups are infinite in number, so no finite table is complete.")
    lines.append("//  - SG 97 (I422) was missing its F222 maximal t-subgroup upstream; that edge")
    lines.append("//    is patched in during generation (see the generator script).")
    lines.append("")
    lines.append("#include <occ/crystal/subgroup.h>")
    lines.append("")
    lines.append("namespace occ::crystal::impl {")
    lines.append("")
    lines.append(f"const int num_subgroup_edges = {n_edges};")
    lines.append("")
    lines.append("// [begin, end) into subgroup_edge_data for space group n, at index n - 1.")
    lines.append("const int subgroup_offsets[231] = {")
    for i in range(0, len(offsets), 16):
        lines.append("    " + ", ".join(str(v) for v in offsets[i : i + 16]) + ",")
    lines.append("};")
    lines.append("")
    lines.append(f"// {stride} bytes per edge:")
    lines.append("//   [0] subgroup space group number")
    lines.append("//   [1] index")
    lines.append("//   [2] 0 = translationengleiche, 1 = klassengleiche")
    lines.append("//   [3..27] the 3x4 matrix [P | p], row-major, as 12 (int8 numerator,")
    lines.append("//           uint8 denominator) pairs")
    lines.append("const unsigned char subgroup_edge_data[] = {")
    for i in range(0, len(blob), 24):
        lines.append("    " + ", ".join(str(v) for v in blob[i : i + 24]) + ",")
    lines.append("};")
    lines.append("")
    lines.append("} // namespace occ::crystal::impl")
    lines.append("")

    args.output.write_text("\n".join(lines))
    size_kb = args.output.stat().st_size / 1024
    print(
        f"wrote {args.output} -- {n_edges} edges ({n_t} t, {n_k} k), {size_kb:.0f} KB of source",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
