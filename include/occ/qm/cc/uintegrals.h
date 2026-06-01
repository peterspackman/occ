#pragma once
#include <cstddef>
#include <functional>
#include <occ/core/linear_algebra.h>
#include <occ/qm/cc/thc.h>   // ThcOptions
#include <occ/qm/cc/uccsd.h> // AOBasis, MolecularOrbitals
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

/// Spin-blocked spatial MO integrals for spin-adapted unrestricted CCSD, in
/// chemist notation, matching the PySCF UCCSD `_ChemistsERIs` block layout.
/// Lowercase = alpha, UPPERCASE = beta, mixed case = alpha/beta. The O(V^4)
/// vvvv blocks are never stored as tensors; they are applied through the three
/// ladder closures (exact stores them, df/thc contract on the fly).
struct UCCIntegrals {
  using T4 = Eigen::Tensor<double, 4>;

  int nocca{0}, noccb{0}, nvira{0}, nvirb{0};
  occ::Vec mo_energy_a; ///< active alpha orbital energies (occ then vir)
  occ::Vec mo_energy_b; ///< active beta orbital energies (occ then vir)

  // alpha-alpha
  T4 oooo, ovoo, ovov, oovv, ovvo, ovvv;
  // beta-beta
  T4 OOOO, OVOO, OVOV, OOVV, OVVO, OVVV;
  // alpha-beta
  T4 ooOO, ovOO, ovOV, ooVV, ovVO, ovVV;
  // beta-alpha
  T4 OVoo, OOvv, OVvo, OVvv;

  /// vvvv ladders: ladder_aa(tau)(i,j,a,b) = sum_ef tau(i,j,e,f) (ae|bf),
  /// ladder_ab(tau)(i,J,a,B) = sum_eF tau(i,J,e,F) (ae|BF).
  std::function<T4(const T4 &)> ladder_aa, ladder_bb, ladder_ab;
};

/// Exact (full AO->MO) spin-blocked integrals. Builds the three spatial chemist
/// tensors (alpha, beta, alpha-beta) and materialises the vvvv blocks for the
/// ladder closures. Reference backend. `n_frozen` frozen-core spatial orbitals
/// are dropped from each spin's occupied space.
UCCIntegrals u_exact_eris(const AOBasis &basis, const MolecularOrbitals &mo,
                          int n_frozen = 0,
                          std::size_t memory_budget = (std::size_t(1) << 30));

/// Density-fitted spin-blocked integrals. All blocks come from the metric-folded
/// DF B-tensor (one 3-center store, shared aux basis), and the three vvvv ladders
/// contract through the per-spin B-tensors so no O(V^4) block is ever formed.
UCCIntegrals u_df_eris(const AOBasis &basis, const AOBasis &aux_basis,
                       const MolecularOrbitals &mo, int n_frozen = 0,
                       std::size_t memory_budget = (std::size_t(1) << 30));

/// THC spin-blocked integrals: cheap blocks (<= 2 virtuals + ovvv) come from DF,
/// and the three vvvv ladders use cross-spin THC factors (one ISDF point set,
/// per-spin X and three cores Vaa/Vbb/Vab). No O(V^4) block is ever formed.
UCCIntegrals u_thc_eris(const AOBasis &basis, const AOBasis &aux_basis,
                        const MolecularOrbitals &mo, const ThcOptions &opts = {},
                        int n_frozen = 0,
                        std::size_t memory_budget = (std::size_t(1) << 30));

} // namespace occ::qm::cc
