#pragma once
#include <occ/qm/cc/uccsd.h> // UCCSDResult
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

/// Spin-orbital CCSD(T) for an open- or closed-shell reference. Builds the
/// antisymmetrised spin-orbital MO integrals from the (UHF or RHF) coefficients
/// -- spin orbitals are sorted by energy -- then runs the textbook spin-orbital
/// CCSD (Stanton-Gauss-Watts-Bartlett 1991) and, optionally, the perturbative
/// (T) (Crawford project-6 formulas). Works for any reference but is ~4-8x the
/// work of the spin-adapted unrestricted path; kept as an independent oracle for
/// validating the spin-adapted `uccsd` and as a fallback for references the
/// spin-adapted path does not handle (e.g. ROHF). `n_frozen` is the number of
/// frozen-core *spatial* orbitals (2*n_frozen spin orbitals are frozen). Exact
/// integrals only.
UCCSDResult uccsd_so(const AOBasis &basis, const MolecularOrbitals &mo,
                     int n_frozen = 0, bool with_triples = true,
                     int max_cycle = 100, double tol = 1e-9);

/// Perturbative (T) correction for converged spin-adapted UCCSD amplitudes,
/// evaluated through the validated spin-orbital triples kernel. The spatial
/// spin-block amplitudes (t1a,t1b; t2aa,t2ab,t2bb) are mapped into the
/// energy-sorted spin-orbital basis built from the MOs, then `so_triples` runs.
double uccsd_t_via_so(const AOBasis &basis, const MolecularOrbitals &mo,
                      int n_frozen, const Eigen::Tensor<double, 2> &t1a,
                      const Eigen::Tensor<double, 2> &t1b,
                      const Eigen::Tensor<double, 4> &t2aa,
                      const Eigen::Tensor<double, 4> &t2ab,
                      const Eigen::Tensor<double, 4> &t2bb);

} // namespace occ::qm::cc
