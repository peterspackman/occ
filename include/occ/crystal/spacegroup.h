#pragma once
#include <occ/crystal/symmetryoperation.h>
#include <gemmi/symmetry.hpp>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace occ::crystal {

using occ::IVec;
using occ::Mat3N;

using SGData = gemmi::SpaceGroup;

class SpaceGroup {
public:
  SpaceGroup(int);
  SpaceGroup(const std::string &);
  SpaceGroup(const std::vector<std::string> &);
  SpaceGroup(const std::vector<SymmetryOperation> &symops);
  int number() const {
      if(m_sgdata != nullptr)
        return m_sgdata->number;
      return 0;
  }
  const std::string symbol() const {
    if(m_sgdata != nullptr) return m_sgdata->hm;
    return "XX";
  }
  std::string short_name() const {
    if(m_sgdata != nullptr)
        return m_sgdata->short_name();
    return "unknown";
  }
  const std::vector<SymmetryOperation> &symmetry_operations() const {
    return m_symops;
  }
  bool has_H_R_choice() const;
  std::pair<IVec, Mat3N> apply_all_symmetry_operations(const Mat3N &) const;

private:
  const SGData *m_sgdata{nullptr};
  std::vector<SymmetryOperation> m_symops;
};

} // namespace occ::crystal
