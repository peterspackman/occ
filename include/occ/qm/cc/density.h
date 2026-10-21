#pragma once
#include <occ/qm/wavefunction.h>
#include <string>

namespace occ::qm {
class HartreeFock;
}

namespace occ::qm::cc {

/// Options for the correlated one-particle densities of a closed-shell RHF
/// reference (MP2 relaxed, CCSD and CCSD(T) unrelaxed, local CCSD(T)).
struct CorrelatedDensityOptions {
  int n_frozen{-1};             ///< -1 = chemical core, 0 = all electrons
  std::string backend{"exact"}; ///< exact | df (ri is accepted for df)
  std::string aux_basis{};      ///< df and local MP2 screening; empty = auto
  double max_memory_gb{1.0};    ///< integrals and (T) buffers
  /// CCSD and Lambda: max |D (t_new - t_old)| (ORCA STol); MP2: max
  /// |Z-vector residual|
  double tol{1e-5};
  int max_cycle{200};
  bool triples{false}; ///< CCSD(T): (T) terms in Lambda and in the density
  // cluster-in-molecule (local_ccsd_density)
  double cluster_radius{8.0};    ///< Angstrom between LMO centres
  double domain_threshold{0.01}; ///< Mulliken population for PAO domains
  double pair_threshold{1e-5};   ///< |MP2 pair energy| (Eh) to join a cluster
  double lno_threshold{1e-6};    ///< MP2 virtual NO occupation kept
  /// relaxed MP2: the SCF's own Coulomb and exchange (density fitting, COSX)
  /// for the orbital response; null = four-centre. Not owned; it must outlive
  /// the call
  const HartreeFock *scf_jk{nullptr};
};

/// A correlated density as natural orbitals of the SCF basis. The natural
/// orbitals are in natural.mo.C with n/2 in natural.mo.occupation and the alpha
/// density in natural.mo.D (occ's restricted conventions).
struct CorrelatedDensity {
  Wavefunction natural;
  Vec occupations;    ///< natural occupations, descending, sum = N
  double e_corr{0.0}; ///< MPn or CCSD correlation energy
  /// (T), from the arrays entering the density; for local_ccsd_density the sum
  /// of each cluster's share for its central LMOs
  double e_triples{0.0};
  int n_frozen{0};
  int n_occ{0}, n_vir{0};   ///< correlated occupied / virtual orbitals
  int cc_iterations{0};     ///< CCSD amplitude iterations
  int lambda_iterations{0}; ///< Lambda (CC) or Z-vector (MP2) iterations
  double t1_norm{0.0}, t2_norm{0.0}, l1_norm{0.0}, l2_norm{0.0};
  std::string backend, aux_basis;
  // local_ccsd_density only
  int n_lmo{0}, n_clusters{0}, max_cluster_occ{0}, max_cluster_vir{0};
  double mean_candidates{0.0}, mean_domain{0.0};
  double mean_cluster_occ{0.0}, mean_cluster_vir{0.0};
};

/// Unrelaxed CCSD (or, with opts.triples, CCSD(T)) density from the Lambda
/// equations (Stanton & Bartlett, JCP 98, 7029 (1993); the (T) terms in the
/// response form of PySCF ccsd_t_lambda_slow / ccsd_t_rdm_slow).
CorrelatedDensity ccsd_density(const Wavefunction &scf,
                               const CorrelatedDensityOptions &opts = {});

/// The same density from one cluster per Pipek-Mezey LMO (cluster-in-molecule,
/// Li & Piecuch) with MP2 pair selection and local natural orbitals; clusters
/// spanning the molecule give ccsd_density back.
CorrelatedDensity local_ccsd_density(const Wavefunction &scf,
                                     const CorrelatedDensityOptions &opts = {});

/// Relaxed MP2 density: the MP2 Lagrangian with the Z-vector orbital response
/// (as PySCF grad/mp2.py), the frozen core relaxed too. With the df backend the
/// MP2 part is built from the B tensors one occupied orbital at a time (no
/// ovvv or vvvv intermediates; B_vv streamed within max_memory_gb). The orbital
/// response builds Coulomb and exchange with opts.scf_jk, which must be set up
/// as the SCF was (density fitting, COSX) for the response to match it; without
/// it the builds are four-centre, right for an SCF with exact J and K.
CorrelatedDensity
mp2_relaxed_density(const Wavefunction &scf,
                    const CorrelatedDensityOptions &opts = {});

/// Unrelaxed MP3 (order 3) or MP4 (order 4; MP4(SDQ) unless opts.triples)
/// density from the MPn Lagrangian, H0 the Fock operator: the oo and vv blocks
/// are the energy derivative, the ov block (MP4 only) the singles-doubles and
/// doubles-triples couplings in the second-order wavefunction. ORCA's AutoCI
/// unrelaxed densities.
CorrelatedDensity mpn_density(const Wavefunction &scf, int order,
                              const CorrelatedDensityOptions &opts = {});

} // namespace occ::qm::cc
