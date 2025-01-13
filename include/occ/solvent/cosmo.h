#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::solvent {
using occ::Mat;
using occ::Mat3N;
using occ::Vec;

namespace cosmo {
occ::Vec solvation_radii(const occ::IVec &);
}

class COSMO {

public:
  struct Result {
    occ::Vec initial;
    occ::Vec converged;
    double energy;
  };

  COSMO(double dielectric, double x = 0.0) : m_x(x), m_dielectric(dielectric) {}

  Result operator()(const Mat3N &, const Vec &, const Vec &) const;

  auto surface_charge(const Vec &charges) const {
    return charges.array() * (m_dielectric - 1) / (m_dielectric + m_x);
  }

  void set_x(double x) { m_x = x; }

  double dielectric() const { return m_dielectric; }

private:
  double m_x{0.0};
  double m_dielectric;
};
} // namespace occ::solvent
