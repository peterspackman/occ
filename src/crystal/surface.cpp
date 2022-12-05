#include <fmt/core.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/surface.h>

namespace occ::crystal {

template <typename T>
void loop_over_miller_indices(T &func, const Crystal &c, double dmin,
                              double dmax = 1.0, bool unique = true) {
    const auto &uc = c.unit_cell();
    HKL limits = uc.hkl_limits(dmin);
    const auto rasu = c.space_group().reciprocal_asu();
    const Mat3 &lattice = uc.reciprocal();
    HKL m;
    for (m.h = -limits.h; m.h <= limits.h; m.h++)
        for (m.k = -limits.k; m.k <= limits.k; m.k++)
            for (m.l = -limits.l; m.l <= limits.l; m.l++) {
                if (!unique || rasu.is_in(m)) {
                    double d = m.d(lattice);
                    if (d > dmin && d < dmax)
                        func(m);
                }
            }
}

Surface::Surface(const HKL &miller, const Crystal &crystal)
    : m_hkl{miller}, m_lattice(crystal.unit_cell().direct()),
      m_reciprocal_lattice(crystal.unit_cell().reciprocal()) {
    m_depth = 1.0 / d();
}

double Surface::depth() const { return m_depth; }

double Surface::d() const { return m_hkl.d(m_reciprocal_lattice); }

std::array<double, 3> Surface::dipole() const { return {}; }

void Surface::print() const {
    fmt::print("Surface ({:3d} {:3d} {:3d}), depth = {}\n", m_hkl.h, m_hkl.k,
               m_hkl.l, depth());
}

std::vector<Surface> generate_surfaces(const Crystal &c, double d_max) {
    std::vector<Surface> result;
    auto f = [&](const HKL &m) { result.emplace_back(Surface(m, c)); };
    loop_over_miller_indices(f, c, 0.05, 0.5);
    std::sort(result.begin(), result.end(),
              [](const Surface &a, const Surface &b) {
                  return a.depth() > b.depth();
              });
    return result;
}

} // namespace occ::crystal
