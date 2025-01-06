#pragma once

#include <occ/crystal/crystal.h>

namespace occ::crystal {

struct SymmetryDimerLabeller {
  explicit SymmetryDimerLabeller(const Crystal &crystal) : m_crystal(crystal) {}

  std::string operator()(const occ::core::Dimer &dimer) const;

  std::string connection{"<=>"};
  SymmetryOperationFormat format{
      "{}", // fmt string
      ","   // delimiter
  };

private:
  std::string format_symop_with_translation(const SymmetryOperation &symop,
                                            const IVec3 &translation) const;
  std::string
  format_molecule_part(const std::string &name, const SymmetryOperation &symop,
                       const IVec3 &translation = IVec3::Zero()) const;

  const Crystal &m_crystal;
};

} // namespace occ::crystal
