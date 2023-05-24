#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/hkl.h>
#include <occ/crystal/unitcell.h>

namespace occ::crystal {

class Crystal;

class Surface {
  public:
    Surface(const HKL &, const Crystal &);

    double depth() const;
    double d() const;
    void print() const;
    Vec3 normal_vector() const;
    inline const auto &hkl() const { return m_hkl; }
    inline const auto &depth_vector() const { return m_depth_vector; }
    inline const auto &a_vector() const { return m_a_vector; };
    inline const auto &b_vector() const { return m_b_vector; };
    inline double area() const { return m_a_vector.cross(m_b_vector).norm(); }
    Vec3 dipole() const;
    bool cuts_line_segment(const Vec3 &origin, const Vec3 &point1,
                           const Vec3 &point2) const;

    static bool check_systematic_absence(const Crystal &, const HKL &);
    static bool faces_are_equivalent(const Crystal &, const HKL &, const HKL &);

  private:
    HKL m_hkl;
    double m_depth{0.0};
    UnitCell m_crystal_unit_cell;
    Vec3 m_a_vector;
    Vec3 m_b_vector;
    Vec3 m_depth_vector;
    double m_angle{0.0};
};

struct CrystalSurfaceGenerationParameters {
    double d_min{0.1};
    double d_max{1.0};
    bool unique{true};
};

std::vector<Surface>
generate_surfaces(const Crystal &c,
                  const CrystalSurfaceGenerationParameters & = {});

} // namespace occ::crystal
