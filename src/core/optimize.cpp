#include <cmath>
#include <functional>
#include <occ/core/optimize.h>
#include <stddef.h>

namespace occ::core::opt {

Brent::Brent(Brent::func_type func, double tolerance, size_t max_iter)
    : m_func(func), m_tolerance(tolerance), m_max_iter(max_iter) {}

void Brent::optimize() {
    m_bracket.bracket_function(m_func);
    auto xa = m_bracket.xa;
    auto xb = m_bracket.xb;
    auto xc = m_bracket.xc;
    auto fa = m_bracket.fa;
    auto fb = m_bracket.fb;
    auto fc = m_bracket.fc;
    size_t num_calls = m_bracket.num_calls;
    double min_tol{1e-11};
    double cg{0.3819660};

    double x{xb}, w{xb}, v{xb};
    double fx = m_func(x);
    double fv{fx}, fw{fx};

    double a, b;
    if (xa < xc) {
        a = xa;
        b = xc;
    } else {
        a = xc;
        b = xa;
    }
    double deltax = 0.0;
    num_calls++;
    size_t iter{0};
    double rat{0.0};
    while (iter < m_max_iter) {
        double tol1 = m_tolerance * std::abs(x) + min_tol;
        double tol2 = 2 * tol1;
        double xmid = 0.5 * (a + b);
        double u = 0.0;
        if (std::abs(x - xmid) < (tol2 - 0.5 * (b - a))) {
            break;
        }
        if (std::abs(deltax) <= tol1) {
            if (x >= xmid)
                deltax = a - x;
            else
                deltax = b - x;
            rat = cg * deltax;
        } else {
            double tmp1 = (x - w) * (fx - fv);
            double tmp2 = (x - v) * (fx - fw);
            double p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if (tmp2 > 0.0)
                p = -p;
            tmp2 = std::abs(tmp2);
            double dx_temp = deltax;
            deltax = rat;
            if ((p > tmp2 * (a - x)) && (p < tmp2 * (b - x)) &&
                (std::abs(p), std::abs(0.5 * tmp2 * dx_temp))) {
                rat = p * 1.0 / tmp2;
                u = x + rat;
                if (((u - a) < tol2) || ((b - u) < tol2)) {
                    if ((xmid - x) >= 0)
                        rat = tol1;
                    else
                        rat = -tol1;
                }
            } else {
                if (x >= xmid)
                    deltax = a - x;
                else
                    deltax = b - x;
                rat = cg * deltax;
            }
        }
        if (std::abs(rat) < tol1) {
            if (rat >= 0)
                u = x + tol1;
            else
                u = x - tol1;
        } else {
            u = x + rat;
        }
        double fu = m_func(u);
        num_calls++;

        if (fu > fx) {
            if (u < x)
                a = u;
            else
                b = u;
            if ((fu <= fw) || (w == x)) {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if ((fu <= fv) || (v == x) || (v == w)) {
                v = u;
                fv = fu;
            }
        } else {
            if (u >= x)
                a = x;
            else
                b = x;
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
        iter++;
    }
    m_xmin = x;
    m_fval = fx;
    m_iter = iter;
    m_num_calls = num_calls;
}

} // namespace occ::core::opt
