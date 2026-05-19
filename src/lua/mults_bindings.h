#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::mults` (CrystalEnergy, CrystalOptimizer, RigidMolecule, ...)
// onto the `occ` namespace. Mirrors the load-bearing types from
// src/python/mults_bindings.cpp; the verbose JSON IO wrappers
// (StructureInput / MoleculeType / Settings etc.) are not bound here —
// callers should use the higher-level `compute_crystal_energy(path)`
// convenience or load via `occ::io::read_structure_json` directly.
void register_mults_bindings(lua_State *L);

} // namespace occ::lua_bindings
