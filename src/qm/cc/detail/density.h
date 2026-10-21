#pragma once
// Internal to occ_cc: what the canonical correlated densities (density.cpp)
// share with the cluster-in-molecule route (local_density.cpp).
#include <functional>
#include <occ/qm/cc/density.h>
#include <occ/qm/cc/einsum.h>
#include <occ/qm/cc/integrals.h>
#include <string>

namespace occ::qm::cc::detail {

using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;

inline void check(const bool ok, const std::string &what) {
  if (!ok)
    throw std::runtime_error(what);
}

// PySCF ccsd_t_lambda_slow / ccsd_t_rdm_slow: the triples terms that make the
// Lambda equations and the density those of CCSD(T), evaluated one occupied
// triple at a time. W[ijk](abc) is PySCF's p6-symmetrised t3-like array w
// (likewise V for v), already divided by D, as a v^3 block; nothing o^3 v^3 is
// ever formed.
struct Triples {
  T2 l1_t; // constant terms added to every Lambda update
  T4 l2_t;
  T2 goo, gvv, dvo;
  T2 gvv_c; // the vv block's share of the occupied directions in proj (local
            // clusters)
  double e_t = 0.0;
  double e_t_c = 0.0; // E(T)'s share of the occupied directions in proj
};

// PySCF ccsd_rdm._gamma1_intermediates: the CCSD blocks before symmetrisation
struct Gamma1 {
  T2 doo, dvv, dvo;
};

Gamma1 gamma1(const T2 &t1, const T4 &t2, const T2 &l1, const T4 &l2);

struct CcSolution {
  T2 t1, l1;
  T4 t2, l2;
  Triples tr;
  double e_corr = 0.0;
  int cc_iterations = 0, lambda_iterations = 0;
};

// CCSD, the triples terms and the Lambda equations on prepared integrals. proj
// (o x n, optional) holds the occupied directions whose share of the triples vv
// block is formed too (a local cluster's central orbitals)
CcSolution solve_cc(const CCIntegrals &cc_eris,
                    const CorrelatedDensityOptions &opts,
                    const std::function<void(const char *)> &mark,
                    const occ::Mat *proj = nullptr);

// "exact" or "df" from opts.backend (auto, conventional and ri accepted)
std::string resolve_backend(const CorrelatedDensityOptions &opts);

// the correlation fitting basis: opts.aux_basis, or the one matched to basis
occ::gto::AOBasis load_aux_basis(const occ::gto::AOBasis &basis,
                                 const CorrelatedDensityOptions &opts,
                                 std::string &aux_name);

// the frozen-core count for a closed-shell reference, as occ's CCSD driver:
// -1 is the chemical core, and at least one occupied orbital stays correlated
int frozen_core_count(const Wavefunction &scf, int n_frozen);

// opts.max_memory_gb in bytes
size_t memory_budget(const CorrelatedDensityOptions &opts);

// occ's integral blocks for mo (exact or DF, as opts.backend), the n_frozen
// lowest occupied left out; the exact vvvv ladder only with_ladder (DF never
// stores it)
CCIntegrals build_eris(const occ::gto::AOBasis &basis,
                       const occ::qm::MolecularOrbitals &mo, int n_frozen,
                       const CorrelatedDensityOptions &opts,
                       std::string &backend, std::string &aux_name,
                       bool with_ladder = true);

// natural orbitals of dm (in scf's MO basis) as an occ Wavefunction: C_NO in
// mo.C, n/2 in mo.occupation
void natural_orbitals(const occ::qm::Wavefunction &scf, const occ::Mat &dm,
                      const std::string &name, double e_total,
                      CorrelatedDensity &result);

} // namespace occ::qm::cc::detail
