#pragma once
#include <functional>

namespace occ::opt {

struct bracket1d {
    typedef const std::function<double(const double &)> &func_type;
    static constexpr double golden_ratio{1.618034};
    static constexpr double small_eps{1e-21};
    double xa{0.0}, xb{1.0}, xc;
    double fa, fb, fc;
    double grow_limit{110.0};
    size_t max_iter{1000};
    size_t num_calls{0};

    void bracket_function(func_type func) {
        fa = func(xa);
        fb = func(xb);
        if (fa < fb) {
            std::swap(xa, xb);
            std::swap(fa, fb);
        }
        xc = xb + golden_ratio * (xb - xa);
        fc = func(xc);
        num_calls = 3;
        auto iter = 0;
        while (fc < fb) {
            auto tmp1 = (xb - xa) * (fb - fc);
            auto tmp2 = (xb - xc) * (fb - fa);
            auto val = tmp2 - tmp1;
            double denom =
                (std::abs(val) < small_eps) ? 2 * small_eps : 2 * val;
            auto w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
            auto wlim = xb + grow_limit * (xc - xb);
            double fw{0.0};
            if (iter > max_iter)
                break;
            iter++;

            if ((w - xc) * (xb - w) > 0.0) {
                fw = func(w);
                num_calls++;
                if (fw < fc) {
                    xa = xb;
                    xb = w;
                    fa = fb;
                    fb = fw;
                    return;
                } else if (fw > fb) {
                    xc = w;
                    fc = fw;
                    return;
                }
                w = xc + golden_ratio * (xc - xb);
                fw = func(w);
                num_calls++;
            } else if ((w - wlim) * (wlim - xc) >= 0.0) {
                w = wlim;
                fw = func(w);
                num_calls++;
            } else if ((w - wlim) * (xc - w) > 0.0) {
                fw = func(w);
                num_calls++;
                if (fw < fc) {
                    xb = xc;
                    xc = w;
                    w = xc + golden_ratio * (xc - xb);
                    fb = fc;
                    fc = fw;
                    fw = func(w);
                    num_calls++;
                }
            } else {
                w = xc + golden_ratio * (xc - xb);
                fw = func(w);
                num_calls++;
            }
            xa = xb;
            xb = xc;
            xc = w;
            fa = fb;
            fb = fc;
            fc = fw;
        }
    }
};

class Brent {
  public:
    typedef const std::function<double(const double &)> &func_type;

    Brent(func_type func, double tol = 1e-8, size_t maxiter = 500);

    const auto num_calls() const { return m_num_calls; }
    double xmin() {
        if (m_num_calls < 1)
            optimize();
        return m_xmin;
    }

    double f_xmin() {
        if (m_num_calls < 1)
            optimize();
        return m_fval;
    }

  private:
    void optimize();

    bracket1d m_bracket;
    func_type m_func;
    double m_tolerance{1e-8};
    size_t m_max_iter{500};
    double m_xmin{0.0};
    double m_fval{0.0};
    size_t m_iter{0};
    size_t m_num_calls{0};
};

} // namespace occ::opt
