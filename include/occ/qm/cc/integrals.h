#pragma once
#include <functional>
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/qm/cc/thc.h>
#include <occ/qm/mo.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;

/// Restricted (closed-shell, spatial-MO) integral blocks in chemist notation
/// (pq|rs), plus a vvvv-ladder callable. The CCSD solver is written against this
/// interface, so the exact / DF / THC backends are drop-in interchangeable. Only
/// the <=2-virtual blocks and ovvv are stored; the O(V^4) vvvv is never formed --
/// it is applied through `ladder`.
struct CCIntegrals {
  using Tensor4 = Eigen::Tensor<double, 4>;

  int nocc{0};
  int nvir{0};
  occ::Vec mo_energy; ///< length nocc+nvir
  occ::Mat fock;      ///< (nmo x nmo); diag(mo_energy) for a canonical reference

  Tensor4 oooo; ///< (ij|kl)
  Tensor4 ooov; ///< (ij|ka)
  Tensor4 oovv; ///< (ij|ab)
  Tensor4 ovoo; ///< (ia|jk)
  Tensor4 ovov; ///< (ia|jb)
  Tensor4 ovvo; ///< (ia|bj)
  Tensor4 ovvv; ///< (ia|bc)

  /// vvvv ladder: L(i,j,a,b) = sum_cd (ac|bd) tau(i,j,c,d), never forming vvvv.
  std::function<Tensor4(const Tensor4 &tau)> ladder;
};

/// Number of frozen-core spatial orbitals for `basis` (chemical core: 1s for
/// Li-Ne, [Ne] for Na-Ar, [Ar] for Sc+), matching the standard CCSD(T) default.
int num_frozen_core(const AOBasis &basis);

/// Exact backend: all blocks (incl. ovvv and the dense vvvv ladder) from a
/// semidirect AO->MO transform (no nao^4 tensor). Reference for the others.
/// `n_frozen` lowest occupied orbitals are excluded from the correlation space.
CCIntegrals exact_eris(const AOBasis &basis, const MolecularOrbitals &mo,
                       int n_frozen = 0,
                       size_t memory_budget = (size_t(1) << 30));

/// Density-fitted (RI) backend: every block from the metric-folded DF B-tensor;
/// the ladder is contracted through B so (ac|bd) is never formed.
CCIntegrals df_eris(const AOBasis &basis, const AOBasis &aux_basis,
                    const MolecularOrbitals &mo, int n_frozen = 0,
                    size_t memory_budget = (size_t(1) << 30));

/// THC backend: the <=2-virtual blocks come from DF (scalable); ovvv and the
/// vvvv ladder are applied through the THC factors (X, V) -- the ladder as three
/// GEMMs, never forming vvvv. `aux_basis` is the DF/LS-THC reference basis.
CCIntegrals thc_eris(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo, const ThcOptions &opts = {},
                     int n_frozen = 0,
                     size_t memory_budget = (size_t(1) << 30));

} // namespace occ::qm::cc
