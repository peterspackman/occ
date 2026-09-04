#include <algorithm>
#include <array>
#include "detail/three_center_kernels.h"
#include <occ/core/log.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::guess {

int ground_state_multiplicity(int Z) {
  // Indexed by atomic number; entry 0 is a placeholder so Z indexes directly.
  static constexpr std::array<int, 119> multiplicities{{
      0,
      2, 1,                                                    // H  - He
      2, 1, 2, 3, 4, 3, 2, 1,                                  // Li - Ne
      2, 1, 2, 3, 4, 3, 2, 1,                                  // Na - Ar
      2, 1,                                                    // K  - Ca
      2, 3, 4, 7, 6, 5, 4, 3, 2, 1,                            // Sc - Zn
      2, 3, 4, 3, 2, 1,                                        // Ga - Kr
      2, 1,                                                    // Rb - Sr
      2, 3, 6, 7, 6, 5, 4, 1, 2, 1,                            // Y  - Cd
      2, 3, 4, 3, 2, 1,                                        // In - Xe
      2, 1,                                                    // Cs - Ba
      2, 1, 4, 5, 6, 7, 8, 9, 6, 5, 4, 3, 2, 1, 2,             // La - Lu
      3, 4, 5, 6, 5, 4, 3, 2, 1,                               // Hf - Hg
      2, 3, 4, 3, 2, 1,                                        // Tl - Rn
      2, 1,                                                    // Fr - Ra
      2, 3, 4, 5, 6, 7, 8, 9, 6, 5, 4, 3, 2, 1, 2,             // Ac - Lr
      3, 4, 5, 6, 5, 4, 3, 2, 1,                               // Rf - Cn
      2, 3, 4, 3, 2, 1,                                        // Nh - Og
  }};
  if (Z < 1 || Z >= static_cast<int>(multiplicities.size()))
    return 0;
  return multiplicities[Z];
}

std::vector<SubshellOccupation> minimal_basis_subshell_occupations(int Z) {
  // Madelung filling order, and plain aufbau within it -- so copper comes out
  // 4s2 3d9 rather than its true 4s1 3d10. A guess wants a smooth, spherical
  // starting density more than it wants the right term symbol, and the
  // exceptions are what the atomic-SCF guess is for.
  static constexpr int order[][2] = {
      {1, 0},                          // 1s
      {2, 0}, {2, 1},                  // 2s 2p
      {3, 0}, {3, 1},                  // 3s 3p
      {4, 0}, {3, 2}, {4, 1},          // 4s 3d 4p
      {5, 0}, {4, 2}, {5, 1},          // 5s 4d 5p
      {6, 0}, {4, 3}, {5, 2}, {6, 1},  // 6s 4f 5d 6p
      {7, 0}, {5, 3}, {6, 2}, {7, 1},  // 7s 5f 6d 7p
  };

  std::vector<SubshellOccupation> occupations;
  int remaining = Z;
  for (const auto &[n, l] : order) {
    if (remaining <= 0)
      break;
    const int capacity = 2 * (2 * l + 1);
    const int electrons = std::min(remaining, capacity);
    occupations.push_back({n, l, static_cast<double>(electrons)});
    remaining -= electrons;
  }
  if (remaining > 0)
    occ::log::warn("{} electrons of Z = {} did not fit the minimal basis "
                   "configuration",
                   remaining, Z);
  return occupations;
}

Mat compute_sap_matrix(const std::vector<occ::core::Atom> &atoms,
                       const AOBasis &basis) {
  auto sap_basis = AOBasis::load_sap_basis(atoms);
  occ::log::debug("Computing SAP matrix using {}", sap_basis.name());

  // The GRASP fits are all-electron atomic potentials, but on an ECP atom the
  // core Hamiltonian this is added to already carries V_ecp together with a
  // nuclear attraction built from the *effective* charge (Z minus the ECP
  // electrons). That combination is itself the atom's valence effective
  // potential, so adding an all-electron SAP potential on top counts the core
  // twice. The core screening is repulsive for an electron, which is why the
  // guess came out far too high: 87 Hartree above the converged energy on
  // AuH, 1700 on a two-coordinate Au(I) complex.
  //
  // Drop the SAP contribution on those centres and keep it everywhere else.
  // `AOBasis` fills this with one zero per atom, so it is never empty --
  // emptiness does not mean "no ECP", and testing it ran the whole shell
  // filter below on every system.
  const auto &ecp_electrons = basis.ecp_electrons();
  if (basis.total_ecp_electrons() > 0) {
    const auto &all_shells = sap_basis.shells();
    const auto &shell_to_atom = sap_basis.shell_to_atom();
    std::vector<Shell> kept;
    kept.reserve(all_shells.size());
    for (size_t i = 0; i < all_shells.size(); i++) {
      const size_t a = shell_to_atom[i];
      if (a < ecp_electrons.size() && ecp_electrons[a] > 0) {
        occ::log::debug("SAP: skipping atom {} (Z={}), its ECP already "
                        "describes the core",
                        a, atoms[a].atomic_number);
        continue;
      }
      kept.push_back(all_shells[i]);
    }
    if (kept.empty())
      return Mat::Zero(basis.nbf(), basis.nbf());
    if (kept.size() != all_shells.size())
      sap_basis = AOBasis(atoms, kept, sap_basis.name());
  }

  IntegralEngine engine(basis);
  engine.set_auxiliary_basis(sap_basis.shells(), false); // true = dummy atoms

  const auto nbf = basis.nbf();
  Mat V_sap = Mat::Zero(nbf, nbf);

  // The fit expands the screened electronic charge Z^el(r) in error
  // functions, which are the Coulomb potentials of unit-charge s-Gaussians:
  // V^el_ij = -sum_p c_p (ij|a_p). So these are three-centre *two-electron*
  // integrals, the same ones density fitting uses.

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

  // Call the 3-center kernel directly with TBB
  if (engine.is_spherical()) {
    detail::three_center_aux_kernel<Shell::Kind::Spherical>(
        collect_integrals, engine.env(), engine.aobasis(), engine.auxbasis(),
        engine.shellpairs());
  } else {
    detail::three_center_aux_kernel<Shell::Kind::Cartesian>(
        collect_integrals, engine.env(), engine.aobasis(), engine.auxbasis(),
        engine.shellpairs());
  }

  occ::log::debug("SAP matrix computed with {} x {} elements", nbf, nbf);
  // The tabulated coefficients sum to -Z, and the screening enters as
  // -sum_p c_p (ij|a_p), so the assembled matrix is negated once here.
  return -V_sap;
}

} // namespace occ::qm::guess
