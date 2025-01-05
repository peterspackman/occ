#pragma once

#include <occ/crystal/crystal.h>

namespace occ::crystal {

struct SymmetryDimerLabeller {
  explicit SymmetryDimerLabeller(const Crystal &crystal) : m_crystal(crystal) {}

  std::string operator()(const occ::core::Dimer &dimer) const;

private:
  static std::string
  format_symop_with_translation(const SymmetryOperation &symop,
                                const IVec3 &translation);
  static std::string
  format_molecule_part(const std::string &name, const SymmetryOperation &symop,
                       const IVec3 &translation = IVec3::Zero());

  std::string connection{"<=>"};
  const Crystal &m_crystal;
};

} // namespace occ::crystal
