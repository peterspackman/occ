#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

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

std::vector<Vec> inline compute_multipoles(int order, const std::vector<PointCharge> &charges, std::array<double, 3> origin = {0.0, 0.0, 0.0})
{
    std::vector<Vec> result;
    result.reserve(order + 1);
    result.emplace_back(Vec::Zero(1));
    for(int i = 1; i <= order; i++)
    {
        result.emplace_back(Vec::Zero(3));
    }
    std::array<double, 3> d;
    for(int i = 0; i < charges.size(); i++)
    {
        const auto &charge = charges[i].first;
        const auto &pos = charges[i].second;
        result[0](0) += charge;
        if(order > 0)
        {
            d = {pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]};
            result[1](0) += charge * d[0];
            result[1](1) += charge * d[1];
            result[1](2) += charge * d[2];
        }
    }
    return result;
}

}
