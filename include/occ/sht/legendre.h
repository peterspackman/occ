#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::sht {

class AssocLegendreP {
  public:
    AssocLegendreP(size_t lm);
    double operator()(size_t l, size_t m, double x) const;

    Vec evaluate_batch(double x) const;
    void evaluate_batch(double x, Vec &) const;

    static double amm(size_t m);
    static double alm(size_t l, size_t m);
    static double blm(size_t l, size_t m);

    Vec work_array() const;

  private:
    size_t l_max{0};
    Mat m_a;
    Mat m_b;
    mutable Mat m_cache;
};

} // namespace occ::sht
