#pragma once
#include <occ/slater/promolecule.h>

namespace occ::slater {

class StockholderWeight {
public:
  StockholderWeight(const PromoleculeDensity &inside,
                    const PromoleculeDensity &outside);

  void set_background_density(float value);
  inline float background_density() const { return m_background_density; }

  OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
    float total_inside = m_inside(pos);
    float total_outside = m_outside(pos) + m_background_density;
    float total = total_inside + total_outside;
    // In vacuum (no promolecule density from either side, e.g. a corner
    // of the bounding box farther than the per-atom cutoff) the ratio is
    // 0/0; treat such points as exterior of the inside molecule.
    if (total <= 0.0f)
      return 0.0f;
    return total_inside / total;
  }

  OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
    auto [total_inside, grad_inside] = m_inside.density_and_gradient(pos);
    auto [total_outside, grad_outside] = m_outside.density_and_gradient(pos);
    total_outside += m_background_density;
    float total = total_inside + total_outside;
    if (total <= 0.0f)
      return FVec3::Zero();
    float denom = total * total;
    return (grad_inside * (2 * total_inside + total_outside) +
            grad_outside * total_inside) /
           denom;
  }

private:
  float m_background_density{0.0f};
  PromoleculeDensity m_inside;
  PromoleculeDensity m_outside;
};

} // namespace occ::slater
