#include <occ/mults/lbfgs.h>
#include <cmath>
#include <algorithm>

namespace occ::mults {

LBFGS::LBFGS(const LBFGSSettings& settings) : m_settings(settings) {}

LBFGSResult LBFGS::minimize(Objective f, const Vec& x0, int max_iter) {
    return minimize(f, x0, nullptr, max_iter);
}

LBFGSResult LBFGS::minimize(Objective f, const Vec& x0, Callback callback, int max_iter) {
    const int n = x0.size();
    LBFGSResult result;
    result.x = x0;
    result.iterations = 0;
    result.function_evaluations = 0;
    result.converged = false;

    // Initialize storage
    m_s.clear();
    m_y.clear();
    m_rho.clear();
    m_s.reserve(m_settings.memory);
    m_y.reserve(m_settings.memory);
    m_rho.reserve(m_settings.memory);

    // Evaluate initial point
    Vec x = x0;
    Vec g(n);
    double fx = f(x, g);
    result.function_evaluations++;

    // Check initial gradient
    double gnorm = g.norm();
    if (gnorm < m_settings.gradient_tol) {
        result.final_energy = fx;
        result.final_gradient = g;
        result.converged = true;
        result.termination_reason = "Initial gradient below tolerance";
        return result;
    }

    // Call callback for iteration 0 (initial state)
    if (callback && !callback(0, x, fx, g)) {
        result.final_energy = fx;
        result.final_gradient = g;
        result.termination_reason = "Stopped by callback";
        return result;
    }

    Vec x_new(n), g_new(n);
    double fx_new;
    Vec d(n);

    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Compute search direction
        d = compute_direction(g, static_cast<int>(m_s.size()));

        // Check descent direction
        double dg = d.dot(g);
        if (dg > 0) {
            // Not a descent direction, reset to steepest descent
            d = -g;
            dg = -gnorm * gnorm;
            m_s.clear();
            m_y.clear();
            m_rho.clear();
        }

        // Line search
        int ls_evals = 0;
        double alpha = line_search(f, x, g, d, fx, x_new, g_new, fx_new, ls_evals);
        result.function_evaluations += ls_evals;

        if (alpha <= 0) {
            // Line search failed
            result.final_energy = fx;
            result.final_gradient = g;
            result.x = x;
            result.termination_reason = "Line search failed";
            return result;
        }

        // Store correction pair
        Vec s = x_new - x;
        Vec y = g_new - g;
        double ys = y.dot(s);

        if (ys > 1e-15) {
            // Curvature condition satisfied
            if (static_cast<int>(m_s.size()) >= m_settings.memory) {
                // Remove oldest pair
                m_s.erase(m_s.begin());
                m_y.erase(m_y.begin());
                m_rho.erase(m_rho.begin());
            }
            m_s.push_back(s);
            m_y.push_back(y);
            m_rho.push_back(1.0 / ys);
        }

        // Check convergence
        double gnorm_new = g_new.norm();
        double xdiff = s.norm();
        double fdiff = std::abs(fx_new - fx);

        // Callback
        if (callback && !callback(iter + 1, x_new, fx_new, g_new)) {
            result.x = x_new;
            result.final_energy = fx_new;
            result.final_gradient = g_new;
            result.termination_reason = "Stopped by callback";
            return result;
        }

        // Update for next iteration
        x = x_new;
        g = g_new;
        fx = fx_new;
        gnorm = gnorm_new;

        // Convergence tests
        if (gnorm < m_settings.gradient_tol) {
            result.x = x;
            result.final_energy = fx;
            result.final_gradient = g;
            result.converged = true;
            result.termination_reason = "Gradient norm below tolerance";
            return result;
        }

        if (fdiff < m_settings.energy_tol) {
            result.x = x;
            result.final_energy = fx;
            result.final_gradient = g;
            result.converged = true;
            result.termination_reason = "Energy change below tolerance";
            return result;
        }

        if (xdiff < m_settings.x_tol) {
            result.x = x;
            result.final_energy = fx;
            result.final_gradient = g;
            result.converged = true;
            result.termination_reason = "Step size below tolerance";
            return result;
        }
    }

    // Maximum iterations reached
    result.x = x;
    result.final_energy = fx;
    result.final_gradient = g;
    result.termination_reason = "Maximum iterations reached";
    return result;
}

Vec LBFGS::compute_direction(const Vec& g, int num_stored) {
    if (num_stored == 0) {
        // No history: steepest descent
        return -g;
    }

    // Two-loop recursion
    Vec q = g;
    std::vector<double> alpha_storage(num_stored);

    // First loop: most recent to oldest
    for (int i = num_stored - 1; i >= 0; --i) {
        alpha_storage[i] = m_rho[i] * m_s[i].dot(q);
        q -= alpha_storage[i] * m_y[i];
    }

    // Initial Hessian approximation: H_0 = gamma * I
    // gamma = s_k^T y_k / (y_k^T y_k) for most recent pair
    double gamma = m_s.back().dot(m_y.back()) / m_y.back().squaredNorm();
    Vec r = gamma * q;

    // Second loop: oldest to most recent
    for (int i = 0; i < num_stored; ++i) {
        double beta = m_rho[i] * m_y[i].dot(r);
        r += (alpha_storage[i] - beta) * m_s[i];
    }

    return -r;
}

double LBFGS::line_search(Objective& f, const Vec& x, const Vec& g, const Vec& d,
                          double f0, Vec& x_new, Vec& g_new, double& f_new,
                          int& num_evals) {
    const double c1 = m_settings.ftol;
    double dg0 = d.dot(g);
    if (dg0 >= 0) {
        // Not a descent direction
        return 0;
    }

    num_evals = 0;
    double alpha = m_settings.initial_step;

    // Simple backtracking line search (Armijo only)
    if (m_settings.backtracking_only) {
        const int max_iter = m_settings.max_linesearch;
        const double rho = m_settings.backtrack_factor;

        for (int iter = 0; iter < max_iter; ++iter) {
            x_new = x + alpha * d;
            f_new = f(x_new, g_new);
            num_evals++;

            // Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*dg0
            if (f_new <= f0 + c1 * alpha * dg0) {
                return alpha;
            }

            // Reduce step size
            alpha *= rho;
            if (alpha < m_settings.min_step) {
                // Accept tiny step anyway
                return alpha / rho;
            }
        }
        // Return best found
        return alpha;
    }

    // Strong Wolfe line search with bracketing and zoom
    const double c2 = m_settings.gtol;
    double alpha_lo = 0;
    double alpha_hi = m_settings.max_step;
    double f_lo = f0;
    double dg_lo = dg0;

    const int max_iter = m_settings.max_linesearch;

    // Bracket phase
    for (int iter = 0; iter < max_iter; ++iter) {
        x_new = x + alpha * d;
        f_new = f(x_new, g_new);
        num_evals++;

        double dg = d.dot(g_new);

        // Armijo condition
        if (f_new > f0 + c1 * alpha * dg0 || (iter > 0 && f_new >= f_lo)) {
            // Found bracket [alpha_lo, alpha]
            alpha_hi = alpha;
            break;
        }

        // Strong Wolfe condition
        if (std::abs(dg) <= -c2 * dg0) {
            // Wolfe conditions satisfied
            return alpha;
        }

        if (dg >= 0) {
            // Found bracket [alpha, alpha_lo]
            alpha_hi = alpha_lo;
            alpha_lo = alpha;
            f_lo = f_new;
            dg_lo = dg;
            break;
        }

        // Increase step size
        alpha_lo = alpha;
        f_lo = f_new;
        dg_lo = dg;
        alpha = std::min(2.0 * alpha, m_settings.max_step);

        if (alpha >= m_settings.max_step) {
            return alpha;  // Accept large step
        }
    }

    // Zoom phase: bisection in [alpha_lo, alpha_hi]
    for (int iter = 0; iter < max_iter; ++iter) {
        // Bisection (could use cubic interpolation for faster convergence)
        alpha = 0.5 * (alpha_lo + alpha_hi);

        if (std::abs(alpha_hi - alpha_lo) < m_settings.min_step) {
            return alpha_lo > 0 ? alpha_lo : alpha;
        }

        x_new = x + alpha * d;
        f_new = f(x_new, g_new);
        num_evals++;

        double dg = d.dot(g_new);

        if (f_new > f0 + c1 * alpha * dg0 || f_new >= f_lo) {
            alpha_hi = alpha;
        } else {
            if (std::abs(dg) <= -c2 * dg0) {
                // Wolfe conditions satisfied
                return alpha;
            }

            if (dg * (alpha_hi - alpha_lo) >= 0) {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha;
            f_lo = f_new;
            dg_lo = dg;
        }
    }

    // Return best found
    if (f_new <= f0) {
        return alpha;
    }

    // Try alpha_lo if zoom failed
    if (alpha_lo > 0) {
        x_new = x + alpha_lo * d;
        f_new = f(x_new, g_new);
        num_evals++;
        if (f_new < f0) {
            return alpha_lo;
        }
    }

    return 0;  // Line search failed
}

} // namespace occ::mults
