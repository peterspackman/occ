#include <cctype>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/io/moldenwriter.h>
#include <occ/qm/wavefunction.h>

namespace occ::io {

namespace {

// Convert MO coefficients from occ's AO convention to the Molden one, with
// shells reordered so each atom's functions are contiguous (as [GTO] lists
// them). Inverse of MoldenReader's conversion.
Mat to_molden_convention(const occ::gto::AOBasis &basis,
                         const std::vector<size_t> &shell_order,
                         const Mat &mos) {
  using occ::gto::ShellOrder;
  const bool pure = basis.is_pure();
  const auto &first_bf = basis.first_bf();
  Mat result(mos.rows(), mos.cols());
  size_t offset = 0;
  for (size_t s : shell_order) {
    const auto &shell = basis[s];
    const int l = shell.l;
    const size_t ours = first_bf[s];
    if (l == 1 && pure) {
      // occ's spherical p order is y, z, x; Molden's is x, y, z
      result.row(offset + 0) = mos.row(ours + 2);
      result.row(offset + 1) = mos.row(ours + 0);
      result.row(offset + 2) = mos.row(ours + 1);
    } else if (l < 2) {
      result.middleRows(offset, shell.size()) =
          mos.middleRows(ours, shell.size());
    } else if (pure) {
      size_t idx = 0;
      auto func = [&](int am, int m) {
        int theirs = occ::gto::shell_index_spherical<ShellOrder::Molden>(am, m);
        result.row(offset + theirs) = mos.row(ours + idx);
        idx++;
      };
      occ::gto::iterate_over_shell<false, ShellOrder::Default>(func, l);
    } else {
      size_t idx = 0;
      auto func = [&](int pi, int pj, int pk, int) {
        int theirs =
            occ::gto::shell_index_cartesian<ShellOrder::Molden>(pi, pj, pk, l);
        result.row(offset + theirs) =
            mos.row(ours + idx) /
            occ::gto::cartesian_normalization_factor(pi, pj, pk);
        idx++;
      };
      occ::gto::iterate_over_shell<true, ShellOrder::Default>(func, l);
    }
    offset += shell.size();
  }
  return result;
}

void write_orbitals(std::ostream &os, const Mat &C, const Vec &energies,
                    const Vec &occupations, const char *spin) {
  for (Eigen::Index j = 0; j < C.cols(); j++) {
    fmt::print(os, " Sym= A\n Ene= {:20.12e}\n Spin= {}\n Occup= {:12.8f}\n",
               energies(j), spin, occupations(j));
    for (Eigen::Index i = 0; i < C.rows(); i++) {
      fmt::print(os, "{:6d} {:22.14e}\n", i + 1, C(i, j));
    }
  }
}

} // namespace

void write_molden(const occ::qm::Wavefunction &wfn, std::ostream &os) {
  using occ::qm::SpinorbitalKind;
  namespace block = occ::qm::block;
  const auto &basis = wfn.basis;
  const auto &mo = wfn.mo;

  if (mo.kind == SpinorbitalKind::General) {
    throw std::runtime_error(
        "Writing general spinorbital wavefunctions to molden is not supported");
  }
  if (mo.C.size() == 0) {
    throw std::runtime_error(
        "Cannot write molden file: wavefunction has no MO coefficients");
  }
  for (const auto &shell : basis.shells()) {
    if (shell.num_contractions() != 1) {
      throw std::runtime_error("Cannot write molden file: generally "
                               "contracted shells are not supported");
    }
  }
  if (basis.have_ecps()) {
    occ::log::warn("Molden files cannot store ECPs: the written file will "
                   "lack the ECP and its core electrons");
  }

  fmt::print(os, "[Molden Format]\n[Title]\nWritten by occ\n");

  fmt::print(os, "[Atoms] AU\n");
  for (size_t i = 0; i < wfn.atoms.size(); i++) {
    const auto &a = wfn.atoms[i];
    fmt::print(os, "{:<3s} {:5d} {:4d} {:20.12f} {:20.12f} {:20.12f}\n",
               occ::core::Element(a.atomic_number).symbol(), i + 1,
               a.atomic_number, a.x, a.y, a.z);
  }

  // [GTO] lists shells atom by atom, so the AO order in [MO] follows that
  std::vector<size_t> shell_order;
  fmt::print(os, "[GTO]\n");
  const auto &atom_to_shell = basis.atom_to_shell();
  for (size_t atom = 0; atom < atom_to_shell.size(); atom++) {
    fmt::print(os, "{} 0\n", atom + 1);
    for (size_t s : atom_to_shell[atom]) {
      shell_order.push_back(s);
      const auto &shell = basis[s];
      fmt::print(os, " {} {} 1.00\n",
                 static_cast<char>(std::tolower(shell.symbol())),
                 shell.num_primitives());
      for (size_t p = 0; p < shell.num_primitives(); p++) {
        fmt::print(os, " {:22.14e} {:22.14e}\n", shell.exponents(p),
                   shell.coeff_normalized(0, p));
      }
    }
    fmt::print(os, "\n");
  }
  if (basis.is_pure()) {
    fmt::print(os, "[5D]\n[7F]\n[9G]\n");
  }

  // occupations: occ stores restricted occupations per spin (0..1)
  const size_t nbf = basis.nbf();
  Vec occupations = mo.occupation;
  const Eigen::Index expected_rows =
      mo.kind == SpinorbitalKind::Unrestricted ? 2 * nbf : nbf;
  if (occupations.size() != expected_rows) {
    occupations = Vec::Zero(expected_rows);
    occupations.head(mo.n_alpha).setConstant(1.0);
    if (mo.kind == SpinorbitalKind::Unrestricted)
      occupations.segment(nbf, mo.n_beta).setConstant(1.0);
  }

  fmt::print(os, "[MO]\n");
  if (mo.kind == SpinorbitalKind::Unrestricted) {
    write_orbitals(os, to_molden_convention(basis, shell_order, block::a(mo.C)),
                   block::a(mo.energies), block::a(occupations), "Alpha");
    write_orbitals(os, to_molden_convention(basis, shell_order, block::b(mo.C)),
                   block::b(mo.energies), block::b(occupations), "Beta");
  } else {
    write_orbitals(os, to_molden_convention(basis, shell_order, mo.C),
                   mo.energies, 2.0 * occupations, "Alpha");
  }
}

void write_molden(const occ::qm::Wavefunction &wfn,
                  const std::string &filename) {
  std::ofstream os(filename);
  if (!os) {
    throw std::runtime_error(
        fmt::format("Unable to open '{}' for writing", filename));
  }
  write_molden(wfn, os);
}

} // namespace occ::io
