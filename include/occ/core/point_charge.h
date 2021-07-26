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

constexpr unsigned int num_multipole_components(unsigned int order)
{
    unsigned int n{0};
    for(unsigned int i = 0; i <= order; i++)
        n += (i + 1) * (i + 2) / 2;
    return n;
}

template<unsigned int order = 0>
inline auto compute_multipoles(const std::vector<PointCharge> &charges,
                               std::array<double, 3> origin = {0.0, 0.0, 0.0})
{
    constexpr unsigned int ncomp = num_multipole_components(order);

    std::array<double, ncomp> result{0.0};
    std::array<double, 3> d;
    for(int i = 0; i < charges.size(); i++)
    {
        const auto &charge = charges[i].first;
        const auto &pos = charges[i].second;
        result[0] += charge;
        d = {pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]};
        if constexpr(order > 0)
        {
            result[1] += charge * d[0];
            result[2] += charge * d[1];
            result[3] += charge * d[2];
        }
        if constexpr(order > 1)
        {
            result[4] += charge * d[0] * d[0]; // xx
            result[5] += charge * d[0] * d[1]; // xy
            result[6] += charge * d[0] * d[2]; // xz
            result[7] += charge * d[1] * d[1]; // yy
            result[8] += charge * d[1] * d[2]; // yz
            result[9] += charge * d[2] * d[2]; // zz
        }
    }
    return result;
}

}
