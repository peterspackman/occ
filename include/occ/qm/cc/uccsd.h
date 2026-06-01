#pragma once
#include <cstddef>
#include <occ/gto/shell.h>
#include <occ/qm/cc/thc.h> // ThcOptions
#include <occ/qm/mo.h>
#include <string>

namespace occ::qm::cc {

using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;

struct UCCSDResult {
  double e_corr{0.0};
  double e_triples{0.0};
  int iterations{0};
  bool converged{false};
};

struct UCCSDOptions {
  std::string backend{"exact"}; // "exact" | "df" | "thc"
  int n_frozen{0};
  bool with_triples{true};
  int max_cycle{100};
  double tol{1e-9};
  std::size_t memory_budget{std::size_t(1) << 30};
  ThcOptions thc{}; ///< THC factorization options (thc backend only)
};

/// Spin-adapted unrestricted CCSD(T) for a UHF (or RHF) reference. Amplitudes
/// and integrals are organised by spin into spatial blocks (t1a,t1b;
/// t2aa,t2ab,t2bb) -- the formulation production codes use for open shell.
/// ~2-4x less work/memory than the spin-orbital path (`uccsd_so`) and, for the
/// df/thc backends, never forms the O(V^4) vvvv block. Exact backend only via
/// this overload; df/thc take an auxiliary basis (see the aux overload).
UCCSDResult uccsd(const AOBasis &basis, const MolecularOrbitals &mo,
                  const UCCSDOptions &opts = {});

/// df/thc backends need an auxiliary basis; "exact" is also accepted here.
UCCSDResult uccsd(const AOBasis &basis, const AOBasis &aux_basis,
                  const MolecularOrbitals &mo, const UCCSDOptions &opts = {});

} // namespace occ::qm::cc
