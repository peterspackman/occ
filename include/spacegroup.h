#pragma once
#include "symmetryoperation.h"
#include <gemmi/symmetry.hpp>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace craso::crystal {

using craso::IVec;
using craso::Mat3N;

using SGData = gemmi::SpaceGroup;

class SpaceGroup {
public:
  SpaceGroup(const std::string &);
  SpaceGroup(const std::vector<SymmetryOperation> &symops);
  int number() const { return m_sgdata->number; }
  const std::string symbol() const { return m_sgdata->hm; }
  std::string short_name() const { return m_sgdata->short_name(); }
  const std::vector<SymmetryOperation> &symmetry_operations() const {
    return m_symops;
  }
  bool has_H_R_choice() const;
  std::pair<IVec, Mat3N> apply_all_symmetry_operations(const Mat3N &) const;

private:
  const SGData *m_sgdata;
  std::vector<SymmetryOperation> m_symops;
};

} // namespace craso::crystal
