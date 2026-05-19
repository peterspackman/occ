#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/sto_ng.h>
#include <stdexcept>
#include <vector>

namespace occ::xtb {

gto::AOBasis build_aobasis(const std::vector<core::Atom> &atoms,
                           const Gfn2Parameters &params) {
  std::vector<gto::Shell> shells;
  shells.reserve(atoms.size() * 3);

  for (const auto &a : atoms) {
    const auto *elem = params.element(a.atomic_number);
    if (!elem) {
      throw std::runtime_error(
          "GFN2: no parameters for element Z=" +
          std::to_string(a.atomic_number));
    }

    const std::array<double, 3> origin = {a.x, a.y, a.z};
    for (const auto &shell : elem->shells) {
      auto fit = slater_to_gauss(shell.n_prim, shell.n, shell.l,
                                 shell.slater_exponent,
                                 /*normalize=*/false);

      gto::Shell s(shell.l, fit.alpha, {fit.coeff}, origin);
      s.incorporate_shell_norm();
      shells.push_back(std::move(s));
    }
  }

  gto::AOBasis basis(atoms, shells, "gfn2-xtb");
  basis.set_pure(true); // GFN2 uses spherical harmonics
  return basis;
}

} // namespace occ::xtb
