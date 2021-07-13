#pragma once
#include <array>
#include <occ/core/atom.h>

namespace occ::core {
using PointCharge = std::pair<double, std::array<double, 3>>;

std::vector<PointCharge> inline make_point_charges(const std::vector<Atom> &atoms)
{
  std::vector<PointCharge> q(atoms.size());
  for (const auto &atom : atoms) {
    q.emplace_back(static_cast<double>(atom.atomic_number),
                   std::array<double, 3>{{atom.x, atom.y, atom.z}});
  }
  return q;
}

}
