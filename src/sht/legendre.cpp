#include <occ/sht/legendre.h>

namespace occ::sht {

AssocLegendreP::AssocLegendreP(size_t lm)
    : l_max(lm), m_a(lm + 1, lm + 1), m_b(lm + 1, lm + 1),
      m_cache(lm + 1, lm + 1) {
    for (size_t m = 0; m <= l_max; m++) {
        m_a(m, m) = amm(m);
        for (size_t l = m + 1; l <= l_max; l++) {
            m_a(l, m) = alm(l, m);
            m_b(l, m) = blm(l, m);
        }
    }
}

double AssocLegendreP::operator()(size_t l, size_t m, double x) const {
    if (m == l) {
        // abs(m) / 2
        return m_a(l, m) * std::pow(1 - x * x, 0.5 * m);
    } else if (m + 1 == l) {
        return m_a(l, m) * x * (*this)(m, m, x);
    } else {
        return (m_a(l, m) * x * (*this)(l - 1, m, x) +
                m_b(l, m) * (*this)(l - 2, m, x));
    }
}

void AssocLegendreP::evaluate_batch(double x, Vec &result) const {
    size_t idx = 0;
    for (size_t m = 0; m <= l_max; m++) {
        for (size_t l = m; l <= l_max; l++) {
            if (l == m) {
                result(idx) = m_a(l, m) * std::pow(1 - x * x, 0.5 * m);
            } else if (l == (m + 1)) {
                result(idx) = m_a(l, m) * x * m_cache(l - 1, m);
            } else {
                result(idx) = m_a(l, m) * x * m_cache(l - 1, m) +
                              m_b(l, m) * m_cache(l - 2, m);
            }
            m_cache(l, m) = result(idx);
            idx++;
        }
    }
}

Vec AssocLegendreP::work_array() const {
    return Vec((l_max + 1) * (l_max + 2) / 2);
}

Vec AssocLegendreP::evaluate_batch(double x) const {
    Vec result = work_array();
    evaluate_batch(x, result);
    return result;
}

double AssocLegendreP::amm(size_t m) {
    const double pi4 = 4 * M_PI;
    double result{1.0};
    for (int k = 1; k <= m; k++) {
        result *= (2.0 * k + 1.0) / (2.0 * k);
    }
    return std::sqrt(result / pi4);
}

double AssocLegendreP::alm(size_t l, size_t m) {
    return std::sqrt((4.0 * l * l - 1) / (1.0 * l * l - m * m));
}

double AssocLegendreP::blm(size_t l, size_t m) {
    return -std::sqrt((2.0 * l + 1) * ((l - 1.0) * (l - 1.0) - m * m) /
                      ((2.0 * l - 3) * (1.0 * l * l - m * m)));
}

} // namespace occ::sht
