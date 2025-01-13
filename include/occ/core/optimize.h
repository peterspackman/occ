#pragma once
#include <functional>
#include <occ/core/log.h>

namespace occ::core::opt {

template <typename Function> struct Bracket1D {
  static constexpr double golden_ratio{1.618034};
  static constexpr double small_eps{1e-21};
  double xa{0.0}, xb{1.0}, xc;
  double fa, fb, fc;
  double grow_limit{110.0};
  size_t max_iter{1000};
  size_t num_calls{0};

  void bracket_function(Function &func) {
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
      double denom = (std::abs(val) < small_eps) ? 2 * small_eps : 2 * val;
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

template <class Function> class LineSearch {
public:
  LineSearch(Function &func, double tol = 1e-8, size_t maxiter = 500)
      : m_func(func), m_tolerance(tol), m_max_iter(maxiter) {}

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

  void set_left(double x) { m_bracket.xa = x; }
  double left() const { return m_bracket.xa; }

  void set_right(double x) { m_bracket.xb = x; }
  double right() const { return m_bracket.xb; }

  void set_guess(double x) {
    m_have_guess = true;
    m_bracket.xc = x;
  }
  double guess() const { return m_bracket.xc; }

private:
  void optimize() {
    if (m_have_guess) {
      m_bracket.fa = m_func(m_bracket.xa);
      m_bracket.fb = m_func(m_bracket.xb);
      m_bracket.fc = m_func(m_bracket.xc);
    } else {
      m_bracket.bracket_function(m_func);
      if (m_bracket.num_calls == 0) {
        occ::log::error("Bracketing failed");
        return;
      }
    }

    auto xa = m_bracket.xa;
    auto xb = m_bracket.xb;
    auto xc = m_bracket.xc;
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
            (std::abs(p) < std::abs(0.5 * tmp2 * dx_temp))) {
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

  Bracket1D<Function> m_bracket;
  Function &m_func;
  bool m_have_guess{false};
  double m_tolerance{1e-8};
  size_t m_max_iter{500};
  double m_xmin{0.0};
  double m_fval{0.0};
  double m_guess{0.0};
  size_t m_iter{0};
  size_t m_num_calls{0};
};

} // namespace occ::core::opt
