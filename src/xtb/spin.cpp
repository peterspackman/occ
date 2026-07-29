#include <occ/xtb/gamma.h> // for ShellTable
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/spin.h>
#include <stdexcept>
#include <string>

namespace occ::xtb {

Mat spin_coupling_matrix(const std::vector<core::Atom> &atoms,
                         const ShellTable &shells,
                         const Gfn2Parameters &params, double scale) {
  const int n_shells = static_cast<int>(shells.atom.size());
  Mat W = Mat::Zero(n_shells, n_shells);
  if (scale == 0.0)
    return W;

  for (int s = 0; s < n_shells; ++s) {
    const int a = shells.atom[s];
    const int z = atoms[a].atomic_number;
    const auto *element = params.element(z);
    if (!element) {
      throw std::runtime_error("GFN2 spin coupling: no parameters for element "
                               "Z=" + std::to_string(z));
    }
    if (!element->has_spin_constants) {
      throw std::runtime_error(
          "GFN2 spin coupling: the parameter set has no 'spin_constants' "
          "block for element Z=" + std::to_string(z) +
          ", which spin-unrestricted calculations require. Update the "
          "parameter file, or set the spin polarization scale to 0 to run "
          "the common-Fock open-shell treatment instead.");
    }
    for (int t = 0; t < n_shells; ++t) {
      if (shells.atom[t] != a)
        continue; // on-site only
      const int idx = spin_pair_index(shells.ang_mom(s), shells.ang_mom(t));
      // Angular momenta above d have no tabulated constant and contribute
      // nothing to the on-site spin interaction.
      W(s, t) = idx < 0 ? 0.0 : scale * element->spin_constants[idx];
    }
  }
  return W;
}

} // namespace occ::xtb
