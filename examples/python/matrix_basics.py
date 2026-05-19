#!/usr/bin/env python3
"""Python counterpart to examples/lua/matrix_basics.lua.

Python's occpy bindings return Eigen matrices / vectors as numpy arrays —
the layout (3xN for Mat3N, column-major in Eigen storage) is the same as
the Lua Mat3N userdata. Use whichever feels natural for the language.

Run:
    python examples/python/matrix_basics.py [molecule.xyz]
(defaults to examples/scf/water.xyz)
"""

import sys
import numpy as np
import occpy


def main():
    # 1. Construction — just plain numpy.
    print("---- construction ----")
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(m)
    print("shape:", m.shape, "size:", m.size)

    dynamic = np.zeros((2, 4))
    dynamic[0, 2] = 7.5   # 0-indexed in Python, vs Lua's [1][3]
    print("\ndynamic matrix after [0, 2] = 7.5:")
    print(dynamic)

    v = np.array([1.5, 2.5, 3.5, 4.5])
    print("\nvector:", v)
    print("v.sum() =", v.sum(), "  np.linalg.norm(v) =", np.linalg.norm(v))

    # 2. Indexing — 0-based, m[row, col] (numpy) ↔ m[row+1][col+1] (Lua).
    print("\n---- indexing ----")
    print("m[0, 0] =", m[0, 0], " (top-left)")
    print("m[2, 2] =", m[2, 2], " (bottom-right)")

    m[1, 1] = -99
    print("after m[1, 1] = -99:")
    print(m)

    m[0, :] = [100, 200, 300]   # whole-row assignment
    print("after m[0] = [100, 200, 300]:")
    print(m)

    # 3. Use in a real occ call. Mat3N comes back as a (3, N) numpy array.
    print("\n---- in a real binding ----")
    xyz_path = sys.argv[1] if len(sys.argv) > 1 else "examples/scf/water.xyz"
    print("loading:", xyz_path)
    mol = occpy.Molecule.from_xyz_file(xyz_path)
    calc = occpy.XtbCalculator(mol)

    pos = calc.positions     # numpy array, shape (3, N)
    print("positions shape:", pos.shape)
    print("oxygen x =", pos[0, 0])

    # 4. Returned matrices also come back as numpy.
    result = calc.single_point()
    print("\natomic charges:", np.round(result.atomic_charges, 4))
    print("sum (should be ~0):", result.atomic_charges.sum())

    grad = calc.gradient()
    print("\ngradient (3xN):")
    print(grad)

    # 5. No equivalent "to_table" / "from_table" round trip needed in
    # Python — numpy arrays serialize cleanly to JSON via .tolist().
    print("\nas list:", pos.tolist())


if __name__ == "__main__":
    main()
