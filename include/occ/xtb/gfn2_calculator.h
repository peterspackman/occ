#pragma once
#include <occ/core/atom.h>
#include <occ/gto/shell.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/multipole_damping.h>
#include <occ/xtb/multipole_ewald.h>
#include <occ/xtb/periodic_integrals.h>
#include <occ/xtb/scc.h>

namespace occ::xtb {

// Owns the AO basis, integrals, parameters, and all geometry-derived
// quantities for GFN2-xTB. A single Gfn2Calculator instance can be reused
// across many `single_point()` calls (e.g. during geometry optimization)
// without re-allocating the integral matrices.
class Gfn2Calculator {
public:
  // Construct from atoms + parameters. All geometry-dependent quantities
  // (basis, integrals, CN, repulsion, gamma, H0) are computed eagerly.
  Gfn2Calculator(std::vector<core::Atom> atoms, Gfn2Parameters params);

  // Replace atomic positions (atomic numbers and shell layout unchanged).
  // Recomputes the geometry-dependent caches.
  void update_positions(const std::vector<core::Atom> &atoms);

  // Run an SCC and return the result. Honors:
  //   opts.include_dispersion (D4)
  //   opts.total_charge / opts.unpaired_electrons
  // and the `include_multipoles` flag below.
  SccResult single_point(const SccOptions &opts = {},
                         bool include_multipoles = true);

  // Convenience wrappers that match the original free-function APIs.
  SccResult run_charge_only(const SccOptions &opts = {}) {
    return single_point(opts, /*include_multipoles=*/false);
  }
  SccResult run_full(const SccOptions &opts = {}) {
    return single_point(opts, /*include_multipoles=*/true);
  }

  // Read-only accessors to internal state (handy for tests / downstream
  // post-processing).
  inline const std::vector<core::Atom> &atoms() const { return m_atoms; }
  inline const Gfn2Parameters &parameters() const { return m_params; }
  inline const gto::AOBasis &basis() const { return m_basis; }
  inline const ShellTable &shell_table() const { return m_shells; }
  inline const Mat &overlap() const { return m_S; }
  inline const Mat &gamma() const { return m_J; }
  inline const Mat &h0() const { return m_H0; }
  inline const Vec &coordination_numbers() const { return m_cn; }
  inline double repulsion_energy() const { return m_e_rep; }
  inline const std::vector<int> &bf_to_atom() const { return m_bf_to_atom; }
  // Mutable access to the integral engine — needed by the analytical gradient
  // path which calls `one_electron_operator_grad(overlap)`.
  inline qm::IntegralEngine &engine() { return m_engine; }

private:
  void recompute_geometry_caches();

  std::vector<core::Atom> m_atoms;
  Gfn2Parameters m_params;
  gto::AOBasis m_basis;
  ShellTable m_shells;
  qm::IntegralEngine m_engine;
  Mat m_S;
  Mat m_J;
  Mat m_H0;
  Vec m_cn;
  Vec m_mp_radii;
  // Lazily built when multipoles are enabled. m_mp_ao stores Bra/Ket atom-
  // centered AO multipole matrices (tblite convention) plus the origin-0 D, Q;
  // m_mp_tensors stores the sd/dd/sq pair tensors used by the clean tensor
  // potential / energy routines (no Ewald split for molecular).
  PeriodicMultipoleAO m_mp_ao;
  MultipolePairTensors m_mp_tensors;
  bool m_have_multipole_ints{false};
  std::vector<int> m_bf_to_atom;
  std::vector<int> m_bf_to_shell;
  Vec m_z_sh;          // reference shell occupations
  int m_n_shells{0};
  int m_nbf{0};
  double m_e_rep{0.0};
  Vec m_n_elec_baseline; // per-shell ref_occ summed (cached)
};

} // namespace occ::xtb
