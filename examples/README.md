# OCC examples

Paired Python and Lua examples covering common occ workflows. Each example
that exists in both `python/` and `lua/` does the same calculation through
the equivalent binding — handy when switching languages or learning either
API for the first time.

## Running

**Python** (requires `occpy` installed; from a build with `WITH_PYTHON_BINDINGS=ON`):

```
python examples/python/<name>.py
```

**Lua** (uses the `occ lua` subcommand built into the main `occ` binary):

```
build/bin/occ lua examples/lua/<name>.lua
```

Drop into the REPL with no script:

```
build/bin/occ lua
occ> mol = occ.load_molecule("examples/scf/water.xyz")
occ> help(mol)         -- list methods on the userdata
occ> mol.positions     -- bare expression auto-pretty-prints the matrix
```

## Paired examples

| Workflow                       | Python                          | Lua                            |
| ------------------------------ | ------------------------------- | ------------------------------ |
| Matrix / vector basics         | `matrix_basics.py`              | `matrix_basics.lua`            |
| GFN2-xTB single point + grad   | `xtb_single_point.py`           | `xtb_single_point.lua`         |
| Hartree–Fock SCF               | `scf.py`                        | `scf.lua`                      |
| DFT + density fitting          | `df.py`                         | `df.lua`                       |
| Geometry optimization + freqs  | `geometry_optimization.py`      | `geometry_optimization.lua`    |
| Wavefunction analysis          | `wavefunction_analysis.py`      | `wavefunction_analysis.lua`    |
| Isosurface (PLY mesh)          | `isosurface.py`                 | `isosurface.lua`               |
| Cube files (density / ESP)     | `cube.py`                       | `cube.lua`                     |
| Crystal-growth lattice energy  | `cg.py`                         | `cg.lua`                       |
| Elastic / acoustic properties  | `acoustic_velocities.py`        | `acoustic_velocities.lua`      |

## Lua-only

| Workflow                       | Path                          |
| ------------------------------ | ----------------------------- |
| Core type tour (Element/Atom…) | `lua/core_demo.lua`           |
| Crystal namespace tour         | `lua/crystal_demo.lua`        |
| First-touch xTB demo           | `lua/water_xtb.lua`           |

## Convention notes

- **Naming.** Python uses `snake_case` for methods/properties (`mol.atoms()`,
  `wfn.mulliken_charges()`). Lua matches it (`mol:atoms()`,
  `wfn:mulliken_charges()`) — `.` for property access, `:` for method calls.
- **Matrices.** Python returns NumPy arrays (`Mat3N` is a `(3, N)`
  array). Lua returns **typed Eigen userdata** of the matching shape —
  `Mat3N`, `Mat3`, `Mat`, `Vector`, `Vec3`, etc. — same column-major /
  xyz-rows layout as Eigen storage. `pos[1][j]` is *x* of atom *j*,
  exactly like numpy's `pos[0, j]`. The userdata supports in-place
  mutation (`pos[1][j] = 0.0`), `:rows()` / `:cols()` / `:size()`,
  whole-row assignment (`m[1] = {a, b, c}`), and a `pp(m)` /
  `tostring(m)` pretty-print. There's no copy at the binding boundary —
  the same Eigen storage is shared with C++. Use `m:to_table()` to
  serialize to a nested Lua table, `occ.Mat3N(t)` to go the other way.
  See `lua/matrix_basics.lua` for a full tour.
- **Static / factory methods.** Python has `Wavefunction.from_fchk(path)`;
  Lua has `occ.Wavefunction_from_fchk(path)` — sol2 doesn't model Python's
  static-method shape so we expose them as free functions in the `occ`
  namespace with a `<Type>_<func>` naming convention.
- **Submodules.** Only `occ.opt` is nested (mirrors Python's `occpy.opt`);
  the rest of the API lives flat in `occ`.

## Background

The Python bindings use [nanobind](https://github.com/wjakob/nanobind);
the Lua bindings use [sol2](https://github.com/ThePhD/sol2) on top of
embedded Lua 5.4, with [linenoise](https://github.com/antirez/linenoise)
for the REPL. All are built from the same `occ` C++ sources via the
parallel `src/python/`, `src/js/`, `src/lua/` binding trees.
