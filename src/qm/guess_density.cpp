#include "Faddeeva.hpp"
#include "detail/three_center_kernels.h"
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/integral_engine.h>
// modified routines from libint2
// include/libint2/chemistry/sto3g_atomic_density.h

namespace occ::qm::guess {

namespace impl {

/* compute orbital occupation numbers for a subshell created
 * by smearing at most num_electrons_remaining
 * (corresponds to spherical averaging)
 */

void update_occupation_subshell(std::vector<double> &destination, int size,
                                int &num_electrons_remaining) {
  const int electrons_allocated =
      (num_electrons_remaining > 2 * size) ? 2 * size : num_electrons_remaining;
  num_electrons_remaining -= electrons_allocated;
  const double electrons_per_orbital =
      static_cast<double>(electrons_allocated) / size;
  for (size_t f = 0; f < size; f++) {
    destination.push_back(electrons_per_orbital);
  }
}

} // namespace impl

int minimal_basis_nao(int Z, bool spherical) {
  int nao = 1;
  if (Z == 1 || Z == 2) // H, He
    nao = 1;
  else if (Z <= 10) // Li - Ne
    nao = 5;
  else if (Z <= 18) // Na - Ar
    nao = 9;
  else if (Z < 20) // K, Ca
    nao = 13;
  else if (Z <= 36) // Sc - Kr
    nao = spherical ? 18 : 19;
  else if (Z <= 54) // Rb - Xe
    nao = spherical ? 27 : 29;
  else if (Z <= 86)
    nao = spherical ? 40 : 46; // 3 D functions, 1 F = 6 extra functions
  else
    throw "minimal basis not defined for elements Z > 86";
  return nao;
}

std::vector<double> minimal_basis_occupation_vector(size_t Z, bool spherical) {

  using impl::update_occupation_subshell;
  std::vector<double> occvec;
  size_t nao = minimal_basis_nao(Z, spherical);
  occvec.reserve(nao);

  int num_of_electrons = Z;
  int dsize = spherical ? 5 : 6;
  int fsize = spherical ? 7 : 10;

  // Fill 1s
  update_occupation_subshell(occvec, 1, num_of_electrons);

  // Fill 2s, 2p
  if (Z > 2) {
    update_occupation_subshell(occvec, 1, num_of_electrons); // 2s
    update_occupation_subshell(occvec, 3, num_of_electrons); // 2p
  }

  // Fill 3s, 3p
  if (Z > 10) {
    update_occupation_subshell(occvec, 1, num_of_electrons); // 3s
    update_occupation_subshell(occvec, 3, num_of_electrons); // 3p
  }

  // Fill 4s, 3d, 4p
  if (Z > 18) {
    update_occupation_subshell(occvec, 1, num_of_electrons);     // 4s
    update_occupation_subshell(occvec, dsize, num_of_electrons); // 3d
    update_occupation_subshell(occvec, 3, num_of_electrons);     // 4p
  }

  // Fill 5s, 4d, 5p
  if (Z > 36) {
    update_occupation_subshell(occvec, 1, num_of_electrons);     // 5s
    update_occupation_subshell(occvec, dsize, num_of_electrons); // 4d
    update_occupation_subshell(occvec, 3, num_of_electrons);     // 5p
  }

  // Fill 6s, 4f, 5d, 6p
  if (Z > 54) {
    update_occupation_subshell(occvec, 1, num_of_electrons);     // 6s
    update_occupation_subshell(occvec, fsize, num_of_electrons); // 4f
    update_occupation_subshell(occvec, dsize, num_of_electrons); // 5d
    update_occupation_subshell(occvec, 3, num_of_electrons);     // 6p
  }

  // Fill 7s, 5f, 6d, 7p
  if (Z > 86) {
    update_occupation_subshell(occvec, 1, num_of_electrons);     // 7s
    update_occupation_subshell(occvec, fsize, num_of_electrons); // 5f
    update_occupation_subshell(occvec, dsize, num_of_electrons); // 6d
    update_occupation_subshell(occvec, 3, num_of_electrons);     // 7p
  }

  // Check for any errors in occupation vector size
  if (occvec.size() != nao) {
    occ::log::warn("Inconsistent number of atomic orbitals in minimal basis "
                   "occupation vector: expected {}, have {}",
                   nao, occvec.size());
  }
  return occvec;
}

Mat compute_sap_matrix(const std::vector<occ::core::Atom> &atoms,
                       const AOBasis &basis,
                       const std::string &sap_basis_name) {
  occ::log::debug("Computing SAP matrix using basis: {}", sap_basis_name);

  auto sap_basis = AOBasis::load_sap_basis(atoms);
  IntegralEngine engine(basis);
  engine.set_auxiliary_basis(sap_basis.shells(), false); // true = dummy atoms

  const auto nbf = basis.nbf();
  const auto naux = sap_basis.nbf();
  Mat V_sap = Mat::Zero(nbf, nbf);

  // Use the existing 3-center kernel to compute (ij|P) integrals
  // where P are the SAP auxiliary functions

  // Lambda to collect 3-center integrals and contract with SAP coefficients
  auto collect_integrals = [&](const IntegralEngine::IntegralResult<3> &args) {
    // Map buffer to matrix for this auxiliary function
    Eigen::Map<const Mat> eri_matrix(args.buffer, args.dims[0], args.dims[1]);

    // Add contribution to SAP matrix (coefficients already have correct sign)
    V_sap.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
        eri_matrix;

    // Handle symmetry (ij) = (ji)
    if (args.bf[0] != args.bf[1]) {
      V_sap.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
          eri_matrix.transpose();
    }
  };

  // Call the 3-center kernel
  auto lambda = [&](int thread_id) {
    if (engine.is_spherical()) {
      detail::three_center_aux_kernel<Shell::Kind::Spherical>(
          collect_integrals, engine.env(), engine.aobasis(), engine.auxbasis(),
          engine.shellpairs(), thread_id);
    } else {
      detail::three_center_aux_kernel<Shell::Kind::Cartesian>(
          collect_integrals, engine.env(), engine.aobasis(), engine.auxbasis(),
          engine.shellpairs(), thread_id);
    }
  };

  occ::parallel::parallel_do(lambda);

  occ::log::debug("SAP matrix computed with {} x {} elements", nbf, nbf);
  return -V_sap;
}

} // namespace occ::qm::guess
