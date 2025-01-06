#include <occ/crystal/dimer_labeller.h>

namespace occ::crystal {

std::string SymmetryDimerLabeller::format_symop_with_translation(
    const SymmetryOperation &symop, const IVec3 &translation) const {
  auto s = symop.translated(translation.cast<double>());
  return s.to_string(format);
}

std::string
SymmetryDimerLabeller::format_molecule_part(const std::string &name,
                                            const SymmetryOperation &symop,
                                            const IVec3 &translation) const {
  if (symop.is_identity() && translation.isZero())
    return name;
  return fmt::format("{}({})", name,
                     format_symop_with_translation(symop, translation));
}

std::string
SymmetryDimerLabeller::operator()(const occ::core::Dimer &dimer) const {
  const auto &a = dimer.a();
  const auto &b = dimer.b();

  std::string a_name =
      a.name().empty() ? "mol_" + std::to_string(a.asymmetric_molecule_idx())
                       : a.name();
  std::string b_name =
      b.name().empty() ? "mol_" + std::to_string(b.asymmetric_molecule_idx())
                       : b.name();

  int sa_int = a.asymmetric_unit_symop()(0);
  int sb_int = b.asymmetric_unit_symop()(0);

  SymmetryOperation symop_a(sa_int);
  SymmetryOperation symop_b(sb_int);

  std::string a_part = format_molecule_part(a_name, symop_a, a.cell_shift());
  std::string b_part = format_molecule_part(b_name, symop_b, b.cell_shift());

  return a_part + connection + b_part;
}

} // namespace occ::crystal
