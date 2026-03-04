#include <occ/mults/trust_region.h>
#include <occ/core/log.h>
#include <cmath>
#include <algorithm>

namespace occ::mults {

TrustRegion::TrustRegion(const TrustRegionSettings& settings)
    : m_settings(settings) {}

TrustRegionResult TrustRegion::minimize(Objective objective,
                                        HessianFunc hessian,
                                        const Vec& x0,
                                        Callback callback) {
    TrustRegionResult result;
    result.function_evaluations = 0;
    result.hessian_evaluations = 0;

    int n = x0.size();
    Vec x = x0;
    double delta = m_settings.initial_radius;

    // Initial evaluation
    auto [f, g] = objective(x);
    result.function_evaluations++;

    double gnorm = g.norm();
    if (m_settings.verbose) {
        occ::log::info("TR iter {:4d}: f = {:14.8e}  |g| = {:10.4e}  delta = {:8.4e}",
                       0, f, gnorm, delta);
    }

    // Call callback for iteration 0
    if (callback && !callback(0, x, f, g)) {
        result.x = x;
        result.final_value = f;
        result.final_gradient_norm = gnorm;
        result.iterations = 0;
        result.converged = false;
        result.termination_reason = "Stopped by callback";
        return result;
    }

    // Check initial convergence
    if (gnorm < m_settings.gradient_tol) {
        result.x = x;
        result.final_value = f;
        result.final_gradient_norm = gnorm;
        result.iterations = 0;
        result.converged = true;
        result.termination_reason = "Initial gradient below tolerance";
        return result;
    }

    Vec D = Vec::Ones(n);  // Diagonal scaling
    Mat H;  // Hessian (may be true or SR1-updated)
    int last_hessian_iter = 0;  // Track when we last computed true Hessian

    for (int iter = 1; iter <= m_settings.max_iterations; ++iter) {
        // Decide whether to compute true Hessian or use SR1 update
        bool compute_true_hessian = (iter == 1) ||
            (m_settings.hessian_update_interval > 0 &&
             (iter - last_hessian_iter) >= m_settings.hessian_update_interval);

        if (compute_true_hessian) {
            H = hessian(x);
            result.hessian_evaluations++;
            last_hessian_iter = iter;
        }
        // Otherwise H is already set from previous iteration (with SR1 updates applied)

        // Optional diagonal scaling
        if (m_settings.use_diagonal_scaling) {
            for (int i = 0; i < n; ++i) {
                double hii = std::abs(H(i, i));
                D[i] = (hii > 1e-8) ? 1.0 / std::sqrt(hii) : 1.0;
            }
        }

        // Solve trust region subproblem
        Vec p = solve_subproblem(g, H, delta, D);

        // Compute actual reduction
        auto [f_new, g_new] = objective(x + p);
        result.function_evaluations++;

        double actual_reduction = f - f_new;
        double predicted = predicted_reduction(g, H, p);

        // Ratio of actual to predicted reduction
        double rho = (std::abs(predicted) > 1e-15) ? actual_reduction / predicted : 0.0;

        if (m_settings.verbose) {
            occ::log::info("TR iter {:4d}: f = {:14.8e}  |g| = {:10.4e}  "
                          "|p| = {:8.4e}  rho = {:8.4f}  delta = {:8.4e}{}",
                          iter, f_new, g_new.norm(), p.norm(), rho, delta,
                          compute_true_hessian ? " [H]" : "");
        }

        // Update trust region radius
        if (rho < m_settings.eta1) {
            // Reject step, shrink trust region
            delta = m_settings.gamma1 * p.norm();
            delta = std::max(delta, 1e-10);
        } else {
            // Accept step - apply SR1 update to Hessian if not recomputing next iter
            Vec s = p;  // Step
            Vec y = g_new - g;  // Gradient difference

            // SR1 update: H = H + ((y - Hs)(y - Hs)^T) / ((y - Hs)^T s)
            if (m_settings.hessian_update_interval > 1) {
                Vec Hs = H * s;
                Vec r = y - Hs;  // y - Hs
                double rTs = r.dot(s);

                // Skip update if denominator is too small (numerical stability)
                double skip_threshold = m_settings.sr1_skip_tol * r.norm() * s.norm();
                if (std::abs(rTs) > skip_threshold) {
                    H += (r * r.transpose()) / rTs;
                }
            }

            x = x + p;
            f = f_new;
            g = g_new;
            gnorm = g.norm();

            if (rho > m_settings.eta2 && p.norm() > 0.9 * delta) {
                // Very good step at boundary, expand trust region
                delta = std::min(m_settings.gamma2 * delta, m_settings.max_radius);
            }

            // Call callback after accepted step
            if (callback && !callback(iter, x, f, g)) {
                result.x = x;
                result.final_value = f;
                result.final_gradient_norm = gnorm;
                result.iterations = iter;
                result.converged = false;
                result.termination_reason = "Stopped by callback";
                return result;
            }
        }

        // Check convergence
        if (gnorm < m_settings.gradient_tol) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Gradient norm below tolerance";
            return result;
        }

        if (p.norm() < m_settings.step_tol * (1.0 + x.norm())) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Step size below tolerance";
            return result;
        }

        if (std::abs(actual_reduction) < m_settings.energy_tol * (1.0 + std::abs(f))) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Energy change below tolerance";
            return result;
        }
    }

    result.x = x;
    result.final_value = f;
    result.final_gradient_norm = gnorm;
    result.iterations = m_settings.max_iterations;
    result.converged = false;
    result.termination_reason = "Maximum iterations reached";
    return result;
}

TrustRegionResult TrustRegion::minimize_hvp(Objective objective,
                                            HessianVectorProduct hvp,
                                            const Vec& x0) {
    TrustRegionResult result;
    result.function_evaluations = 0;
    result.hessian_evaluations = 0;

    int n = x0.size();
    Vec x = x0;
    double delta = m_settings.initial_radius;

    // Initial evaluation
    auto [f, g] = objective(x);
    result.function_evaluations++;

    double gnorm = g.norm();
    if (m_settings.verbose) {
        occ::log::info("TR iter {:4d}: f = {:14.8e}  |g| = {:10.4e}  delta = {:8.4e}",
                       0, f, gnorm, delta);
    }

    if (gnorm < m_settings.gradient_tol) {
        result.x = x;
        result.final_value = f;
        result.final_gradient_norm = gnorm;
        result.iterations = 0;
        result.converged = true;
        result.termination_reason = "Initial gradient below tolerance";
        return result;
    }

    Vec D = Vec::Ones(n);

    for (int iter = 1; iter <= m_settings.max_iterations; ++iter) {
        // Create Hv function for current x
        auto Hv = [&](const Vec& v) { return hvp(x, v); };
        result.hessian_evaluations++;  // Count setup as one "evaluation"

        // Solve trust region subproblem
        Vec p = solve_subproblem_hvp(g, Hv, delta, D);

        // Compute actual reduction
        auto [f_new, g_new] = objective(x + p);
        result.function_evaluations++;

        double actual_reduction = f - f_new;
        double predicted = predicted_reduction_hvp(g, Hv, p);

        double rho = (std::abs(predicted) > 1e-15) ? actual_reduction / predicted : 0.0;

        if (m_settings.verbose) {
            occ::log::info("TR iter {:4d}: f = {:14.8e}  |g| = {:10.4e}  "
                          "|p| = {:8.4e}  rho = {:8.4f}  delta = {:8.4e}",
                          iter, f_new, g_new.norm(), p.norm(), rho, delta);
        }

        if (rho < m_settings.eta1) {
            delta = m_settings.gamma1 * p.norm();
            delta = std::max(delta, 1e-10);
        } else {
            x = x + p;
            f = f_new;
            g = g_new;
            gnorm = g.norm();

            if (rho > m_settings.eta2 && p.norm() > 0.9 * delta) {
                delta = std::min(m_settings.gamma2 * delta, m_settings.max_radius);
            }
        }

        if (gnorm < m_settings.gradient_tol) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Gradient norm below tolerance";
            return result;
        }

        if (p.norm() < m_settings.step_tol * (1.0 + x.norm())) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Step size below tolerance";
            return result;
        }

        if (std::abs(actual_reduction) < m_settings.energy_tol * (1.0 + std::abs(f))) {
            result.x = x;
            result.final_value = f;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Energy change below tolerance";
            return result;
        }
    }

    result.x = x;
    result.final_value = f;
    result.final_gradient_norm = gnorm;
    result.iterations = m_settings.max_iterations;
    result.converged = false;
    result.termination_reason = "Maximum iterations reached";
    return result;
}

TrustRegionResult TrustRegion::minimize_bfgs(GradObjective f, const Vec& x0,
                                              Callback callback) {
    TrustRegionResult result;
    result.function_evaluations = 0;
    result.hessian_evaluations = 0;

    const int n = x0.size();
    Vec x = x0;
    Vec g(n);
    double fx = f(x, g);
    result.function_evaluations++;

    double gnorm = g.norm();
    if (callback && !callback(0, x, fx, g)) {
        result.x = x;
        result.final_value = fx;
        result.final_gradient_norm = gnorm;
        result.iterations = 0;
        result.converged = false;
        result.termination_reason = "Stopped by callback";
        return result;
    }

    if (gnorm < m_settings.gradient_tol) {
        result.x = x;
        result.final_value = fx;
        result.final_gradient_norm = gnorm;
        result.iterations = 0;
        result.converged = true;
        result.termination_reason = "Initial gradient below tolerance";
        return result;
    }

    // Start with identity Hessian (like MSTMIN)
    Mat H = Mat::Identity(n, n);
    double delta = m_settings.initial_radius;

    for (int iter = 1; iter <= m_settings.max_iterations; ++iter) {
        // Solve trust region subproblem: min g'p + 0.5 p'Hp  s.t. ||p|| <= delta
        Vec D = Vec::Ones(n);
        if (m_settings.use_diagonal_scaling) {
            for (int i = 0; i < n; ++i) {
                double hii = std::abs(H(i, i));
                D[i] = (hii > 1e-8) ? 1.0 / std::sqrt(hii) : 1.0;
            }
        }
        Vec p = solve_subproblem(g, H, delta, D);

        // Trial point
        Vec x_new(n);
        Vec g_new(n);
        double fx_new = f(x + p, g_new);
        x_new = x + p;
        result.function_evaluations++;

        double actual_reduction = fx - fx_new;
        double predicted = predicted_reduction(g, H, p);
        double rho = (std::abs(predicted) > 1e-15) ? actual_reduction / predicted : 0.0;

        if (rho < m_settings.eta1) {
            // Reject step, shrink trust region
            delta = m_settings.gamma1 * p.norm();
            delta = std::max(delta, 1e-10);
        } else {
            // Accept step — apply BFGS update to Hessian
            Vec s = p;
            Vec y = g_new - g;
            double ys = y.dot(s);

            if (ys > 1e-15) {
                // BFGS: H += yy'/ys - Hss'H/(s'Hs)
                Vec Hs = H * s;
                double sHs = s.dot(Hs);
                if (std::abs(sHs) > 1e-15) {
                    H += (y * y.transpose()) / ys - (Hs * Hs.transpose()) / sHs;
                }
            }

            x = x_new;
            fx = fx_new;
            g = g_new;
            gnorm = g.norm();

            if (rho > m_settings.eta2 && p.norm() > 0.9 * delta) {
                delta = std::min(m_settings.gamma2 * delta, m_settings.max_radius);
            }

            if (callback && !callback(iter, x, fx, g)) {
                result.x = x;
                result.final_value = fx;
                result.final_gradient_norm = gnorm;
                result.iterations = iter;
                result.converged = false;
                result.termination_reason = "Stopped by callback";
                return result;
            }
        }

        // Convergence checks
        if (gnorm < m_settings.gradient_tol) {
            result.x = x;
            result.final_value = fx;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Gradient norm below tolerance";
            return result;
        }

        if (p.norm() < m_settings.step_tol * (1.0 + x.norm())) {
            result.x = x;
            result.final_value = fx;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Step size below tolerance";
            return result;
        }

        if (std::abs(actual_reduction) < m_settings.energy_tol * (1.0 + std::abs(fx))) {
            result.x = x;
            result.final_value = fx;
            result.final_gradient_norm = gnorm;
            result.iterations = iter;
            result.converged = true;
            result.termination_reason = "Energy change below tolerance";
            return result;
        }
    }

    result.x = x;
    result.final_value = fx;
    result.final_gradient_norm = gnorm;
    result.iterations = m_settings.max_iterations;
    result.converged = false;
    result.termination_reason = "Maximum iterations reached";
    return result;
}

Vec TrustRegion::solve_subproblem(const Vec& g, const Mat& H, double delta,
                                   const Vec& D) {
    // Steihaug-CG: Truncated conjugate gradient for trust region
    int n = g.size();
    Vec z = Vec::Zero(n);  // Solution
    Vec r = g;             // Residual = g + Hz = g initially
    Vec d = -r;            // Search direction

    double r_norm_sq = r.squaredNorm();
    double tol = m_settings.cg_tol * g.norm();
    tol = std::max(tol, 1e-12);

    for (int j = 0; j < m_settings.max_cg_iterations; ++j) {
        Vec Hd = H * d;
        double dHd = d.dot(Hd);

        // Check for negative curvature
        if (dHd <= 0) {
            // Move to boundary along d
            double tau = find_boundary_step(z, d, delta);
            return z + tau * d;
        }

        double alpha = r_norm_sq / dHd;
        Vec z_new = z + alpha * d;

        // Check if we hit the boundary
        if (z_new.norm() >= delta) {
            double tau = find_boundary_step(z, d, delta);
            return z + tau * d;
        }

        z = z_new;
        r = r + alpha * Hd;

        double r_norm_sq_new = r.squaredNorm();

        // Check CG convergence
        if (std::sqrt(r_norm_sq_new) < tol) {
            return z;
        }

        double beta = r_norm_sq_new / r_norm_sq;
        d = -r + beta * d;
        r_norm_sq = r_norm_sq_new;
    }

    return z;
}

Vec TrustRegion::solve_subproblem_hvp(const Vec& g,
                                       const std::function<Vec(const Vec&)>& Hv,
                                       double delta,
                                       const Vec& D) {
    int n = g.size();
    Vec z = Vec::Zero(n);
    Vec r = g;
    Vec d = -r;

    double r_norm_sq = r.squaredNorm();
    double tol = m_settings.cg_tol * g.norm();
    tol = std::max(tol, 1e-12);

    for (int j = 0; j < m_settings.max_cg_iterations; ++j) {
        Vec Hd = Hv(d);
        double dHd = d.dot(Hd);

        if (dHd <= 0) {
            double tau = find_boundary_step(z, d, delta);
            return z + tau * d;
        }

        double alpha = r_norm_sq / dHd;
        Vec z_new = z + alpha * d;

        if (z_new.norm() >= delta) {
            double tau = find_boundary_step(z, d, delta);
            return z + tau * d;
        }

        z = z_new;
        r = r + alpha * Hd;

        double r_norm_sq_new = r.squaredNorm();

        if (std::sqrt(r_norm_sq_new) < tol) {
            return z;
        }

        double beta = r_norm_sq_new / r_norm_sq;
        d = -r + beta * d;
        r_norm_sq = r_norm_sq_new;
    }

    return z;
}

double TrustRegion::predicted_reduction(const Vec& g, const Mat& H, const Vec& p) const {
    // Predicted reduction: m(0) - m(p) = -g^T p - 0.5 p^T H p
    return -g.dot(p) - 0.5 * p.dot(H * p);
}

double TrustRegion::predicted_reduction_hvp(const Vec& g,
                                            const std::function<Vec(const Vec&)>& Hv,
                                            const Vec& p) const {
    return -g.dot(p) - 0.5 * p.dot(Hv(p));
}

double TrustRegion::find_boundary_step(const Vec& p, const Vec& d, double delta) const {
    // Find tau >= 0 such that ||p + tau*d|| = delta
    // Solve: ||p||^2 + 2*tau*(p.d) + tau^2*||d||^2 = delta^2
    double pp = p.squaredNorm();
    double pd = p.dot(d);
    double dd = d.squaredNorm();

    double a = dd;
    double b = 2.0 * pd;
    double c = pp - delta * delta;

    double disc = b * b - 4.0 * a * c;
    if (disc < 0) disc = 0;  // Numerical protection

    // We want the positive root
    double tau = (-b + std::sqrt(disc)) / (2.0 * a);
    return std::max(0.0, tau);
}

} // namespace occ::mults
