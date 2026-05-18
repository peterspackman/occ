-- examples/lua/wavefunction_analysis.lua
-- Parallel to examples/python/wavefunction_analysis.py: run an HF/STO-3G
-- calculation and analyze the resulting wavefunction.
--
-- Run:
--     occ lua examples/lua/wavefunction_analysis.lua [water.xyz]

local path = arg[1] or "examples/scf/water.xyz"
local mol = occ.load_molecule(path)
print(string.format("Loaded %s: %s\n", path, tostring(mol)))

local basis = occ.AOBasis_load(mol:atoms(), "sto-3g")
print(basis)

local hf = occ.HartreeFock(basis)
local scf = hf:scf()
scf:set_charge_multiplicity(0, 1)
local energy = scf:run()
print(string.format("SCF energy = %.10f Ha\n", energy))

local wfn = scf:wavefunction()

print("Orbital energies (Hartree):")
for i, e in ipairs(wfn.molecular_orbitals.orbital_energies) do
    print(string.format("  φ_%2d: %+.6f", i, e))
end

-- Mulliken (population-based) and CHELPG (potential-fitted) charges.
io.write("\nMulliken charges:")
for _, q in ipairs(wfn:mulliken_charges()) do
    io.write(string.format(" %+.4f", q))
end
io.write("\n")

io.write("CHELPG   charges:")
for _, q in ipairs(wfn:chelpg_charges()) do
    io.write(string.format(" %+.4f", q))
end
io.write("\n")

-- Electron density on 3 probe points. Columns are atoms in the eigen
-- convention; for points pass as 3xN: rows = x, y, z.
local pts = {{0.0, 1.0, 2.0},
             {0.0, 0.0, 0.0},
             {0.0, 0.0, 0.0}}
local density = wfn:electron_density(pts)
io.write("\nDensity sample (3 points):")
-- electron_density returns a single-row matrix → flatten the first row.
for _, row in ipairs(density) do
    for _, v in ipairs(row) do
        io.write(string.format(" %.6f", v))
    end
end
io.write("\n")

-- Persist the wavefunction.
local out_base = "/tmp/wfn_analysis_lua"
wfn:save(out_base .. ".owf.json")
wfn:to_fchk(out_base .. ".fchk")
print(string.format("\nWrote %s.owf.json and %s.fchk", out_base, out_base))
