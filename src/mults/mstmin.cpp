#include <occ/mults/mstmin.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace occ::mults {

namespace {

constexpr double kTiny = 1e-30;

inline double clamp_abs(double v, double bound) {
    if (bound <= 0.0) return v;
    return std::max(-bound, std::min(bound, v));
}

} // namespace

MSTMIN::MSTMIN(const MSTMINSettings& settings)
    : m_settings(settings) {}

MSTMINResult MSTMIN::minimize(Objective f, const Vec& x0, int max_iter) {
    return minimize(f, x0, nullptr, max_iter);
}

MSTMINResult MSTMIN::minimize(Objective f, const Vec& x0,
                              Callback callback, int max_iter) {
    MSTMINResult result;
    result.x = x0;

    const int n = static_cast<int>(x0.size());
    if (n == 0) {
        result.converged = true;
        result.termination_reason = "No parameters";
        result.final_gradient = Vec(0);
        return result;
    }

    Vec x = x0;
    Vec g(n);
    double fx = f(x, g);
    result.function_evaluations = 1;

    if (!std::isfinite(fx) || !g.allFinite()) {
        result.final_energy = fx;
        result.final_gradient = g;
        result.termination_reason = "Initial objective produced non-finite values";
        return result;
    }

    if (m_settings.max_function_evaluations > 0 &&
        result.function_evaluations >= m_settings.max_function_evaluations) {
        result.final_energy = fx;
        result.final_gradient = g;
        result.termination_reason =
            "Maximum function evaluations reached before optimization step";
        return result;
    }

    if (callback && !callback(0, x, fx, g)) {
        result.final_energy = fx;
        result.final_gradient = g;
        result.termination_reason = "Stopped by callback";
        return result;
    }

    InternalState st;
    st.del = Vec::Zero(n);
    st.g_trial = Vec::Zero(n);
    st.wrk = Vec::Zero(n);
    st.w = Mat::Identity(n, n);

    SearchState state = SearchState::Setup;
    initialize_direction(g, st, state, true);

    const double gd_tol =
        std::max(1e-16, m_settings.step_tol * m_settings.directional_tol_factor);

    if (g.norm() < m_settings.gradient_tol) {
        result.converged = true;
        result.termination_reason = "Initial gradient norm below tolerance";
        result.final_energy = fx;
        result.final_gradient = g;
        return result;
    }

    if (st.delmax < m_settings.step_tol) {
        result.converged = true;
        result.termination_reason = "Initial search direction below step tolerance";
        result.final_energy = fx;
        result.final_gradient = g;
        return result;
    }

    if (std::abs(st.gd1) < gd_tol) {
        result.converged = true;
        result.termination_reason = "Initial directional derivative below tolerance";
        result.final_energy = fx;
        result.final_gradient = g;
        return result;
    }

    bool reached_max_iter = true;
    int line_search_restarts = 0;
    bool eval_budget_exhausted = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool accepted = false;
        bool reset_hessian = false;

        for (int ls_iter = 0; ls_iter < m_settings.max_line_search; ++ls_iter) {
            Vec x_trial = x + st.alpha2 * st.del;
            double f_trial = f(x_trial, st.g_trial);
            result.function_evaluations++;

            if (m_settings.max_function_evaluations > 0 &&
                result.function_evaluations >= m_settings.max_function_evaluations) {
                eval_budget_exhausted = true;
                break;
            }

            if (!std::isfinite(f_trial) || !st.g_trial.allFinite()) {
                st.alpha2 *= 0.5;
                st.alpha2 = clamp_abs(st.alpha2, std::abs(st.alpham));
                if (std::abs(st.alpha2) < m_settings.step_tol) {
                    reset_hessian = true;
                    break;
                }
                continue;
            }

            // Robust guardrail: reject clearly uphill trial steps and backtrack.
            // This prevents occasional large-energy excursions from bad curvature updates.
            if (m_settings.enforce_energy_decrease) {
                const double f_increase_tol =
                    m_settings.energy_increase_tol_abs +
                    m_settings.energy_increase_tol_rel * std::max(1.0, std::abs(fx));
                if (f_trial > fx + f_increase_tol) {
                    st.alpha2 *= 0.5;
                    st.alpha2 = clamp_abs(st.alpha2, std::abs(st.alpham));
                    if (std::abs(st.alpha2) < m_settings.step_tol) {
                        reset_hessian = true;
                        break;
                    }
                    continue;
                }
            }

            double gd2 = st.g_trial.dot(st.del);
            const double alphap = st.alpha2;

            const bool lreset = (st.ireset >= 8 && gd2 > st.gd1);
            const double gd1_safe = (std::abs(st.gd1) > kTiny)
                                        ? st.gd1
                                        : std::copysign(kTiny, st.gd1 == 0.0 ? -1.0 : st.gd1);
            const bool updt =
                (1.0 + m_settings.usalp * st.alpha2 / st.alpham) *
                    (1.0 - std::abs(gd2 / gd1_safe)) >
                0.75;

            if (!updt && !lreset && state != SearchState::ForceUpdate) {
                if (state == SearchState::Setup) {
                    if (gd2 < st.gd1) {
                        state = SearchState::NegativeCurvature;
                        if (!negative_curvature_step(
                                st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                st.gd1, gd2, st.gd3, st.gd4, st.alpham,
                                st.negcur, state, st.nsrch)) {
                            result.termination_reason =
                                "Invalid minimization: persistent negative curvature";
                            result.final_energy = fx;
                            result.final_gradient = g;
                            result.x = x;
                            return result;
                        }
                    } else if (gd2 > 0.0) {
                        state = SearchState::LinearInterpolation;
                        st.gd4 = st.gd1;
                        st.alpha4 = 0.0;
                        st.ireset++;
                        linear_search_step(st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                           gd2, st.gd3, st.gd4, st.alpham);
                    } else {
                        state = SearchState::Extrapolation;
                        if (st.alpha2 - st.alpham > -m_settings.step_tol) {
                            st.alpham += 1.0;
                        }
                        st.gd4 = st.gd1;
                        st.alpha4 = 0.0;
                        linear_search_step(st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                           gd2, st.gd3, st.gd4, st.alpham);
                    }
                } else if (state == SearchState::LinearInterpolation) {
                    st.ireset++;
                    if (!quadratic_interpolation(st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                                 gd2, st.gd3, st.gd4, alphap)) {
                        result.termination_reason = "Quadratic interpolation failed";
                        result.final_energy = fx;
                        result.final_gradient = g;
                        result.x = x;
                        return result;
                    }
                } else if (state == SearchState::Extrapolation ||
                           state == SearchState::NegativeCurvature) {
                    if (gd2 < st.gd1) {
                        state = SearchState::NegativeCurvature;
                        if (!negative_curvature_step(
                                st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                st.gd1, gd2, st.gd3, st.gd4, st.alpham,
                                st.negcur, state, st.nsrch)) {
                            result.termination_reason =
                                "Invalid minimization: persistent negative curvature";
                            result.final_energy = fx;
                            result.final_gradient = g;
                            result.x = x;
                            return result;
                        }
                    } else if (gd2 > 0.0) {
                        st.negcur = false;
                        std::swap(st.alpha2, st.alpha3);
                        std::swap(gd2, st.gd3);
                        state = SearchState::LinearInterpolation;
                        if (!quadratic_interpolation(st.alpha, st.alpha2, st.alpha3, st.alpha4,
                                                     gd2, st.gd3, st.gd4, alphap)) {
                            result.termination_reason = "Quadratic interpolation failed";
                            result.final_energy = fx;
                            result.final_gradient = g;
                            result.x = x;
                            return result;
                        }
                    } else {
                        st.negcur = false;
                        state = SearchState::Extrapolation;
                        if (st.alpha2 - st.alpham > -m_settings.step_tol) {
                            st.alpham += 1.0;
                        }
                        if (!quadratic_extrapolation(st.alpha, st.alpha2, st.alpha3,
                                                     st.alpha4, gd2, st.gd3, st.gd4,
                                                     st.alpham, state)) {
                            result.termination_reason = "Quadratic extrapolation failed";
                            result.final_energy = fx;
                            result.final_gradient = g;
                            result.x = x;
                            return result;
                        }
                    }
                }
                continue;
            }

            if (st.nupd >= m_settings.max_updates) {
                st.w.setIdentity();
                st.nupd = 0;
                reset_hessian = true;
                break;
            }

            st.alpha = st.alpha2;
            Vec s = st.alpha * st.del;
            Vec y = st.g_trial - g;
            const double dg = s.dot(y);
            st.wrk.noalias() = st.w * y;
            const double gwg = y.dot(st.wrk);

            if (!std::isfinite(dg) || !std::isfinite(gwg) ||
                std::abs(dg) <= kTiny || std::abs(gwg) <= kTiny) {
                st.w.setIdentity();
                st.nupd = 0;
                reset_hessian = true;
                break;
            }

            if (dg < gwg) {
                st.w.noalias() += (s * s.transpose()) / dg -
                                  (st.wrk * st.wrk.transpose()) / gwg;
            } else {
                const double q = 1.0 + gwg / dg;
                Vec r = -s / dg;
                Vec t = (-st.wrk + q * s) / dg;
                st.w.noalias() += r * st.wrk.transpose() + t * s.transpose();
            }

            st.w = 0.5 * (st.w + st.w.transpose());

            const double fx_old = fx;
            x = std::move(x_trial);
            g = st.g_trial;
            fx = f_trial;

            result.iterations = iter + 1;

            if (callback && !callback(result.iterations, x, fx, g)) {
                result.final_energy = fx;
                result.final_gradient = g;
                result.x = x;
                result.termination_reason = "Stopped by callback";
                return result;
            }

            if (g.norm() < m_settings.gradient_tol) {
                result.converged = true;
                result.termination_reason = "Gradient norm below tolerance";
                accepted = true;
                reached_max_iter = false;
                break;
            }

            if (std::abs(fx - fx_old) < m_settings.energy_tol) {
                result.converged = true;
                result.termination_reason = "Energy change below tolerance";
                accepted = true;
                reached_max_iter = false;
                break;
            }

            if (!st.negcur) {
                st.nsrch = 0;
            }
            st.nupd++;

            state = SearchState::Setup;
            initialize_direction(g, st, state, false);

            if (st.delmax < m_settings.step_tol) {
                result.converged = true;
                result.termination_reason = "Step size below tolerance";
                accepted = true;
                reached_max_iter = false;
                break;
            }

            if (std::abs(st.gd1) < gd_tol) {
                result.converged = true;
                result.termination_reason = "Directional derivative below tolerance";
                accepted = true;
                reached_max_iter = false;
                break;
            }

            // Accepted step: reset failure/restart counter.
            line_search_restarts = 0;
            accepted = true;
            break;
        }

        if (reset_hessian) {
            line_search_restarts++;
            if (line_search_restarts > m_settings.max_line_search_restarts) {
                result.termination_reason =
                    "Line search repeatedly reset Hessian without acceptable step";
                reached_max_iter = false;
                break;
            }

            state = SearchState::Setup;
            initialize_direction(g, st, state, true);

            if (st.delmax < m_settings.step_tol) {
                result.converged = true;
                result.termination_reason = "Step size below tolerance";
                reached_max_iter = false;
                break;
            }

            if (std::abs(st.gd1) < gd_tol) {
                result.converged = true;
                result.termination_reason = "Directional derivative below tolerance";
                reached_max_iter = false;
                break;
            }
            continue;
        }

        if (eval_budget_exhausted) {
            result.termination_reason =
                "Maximum function evaluations reached during line search";
            reached_max_iter = false;
            break;
        }

        if (result.converged) {
            break;
        }

        if (!accepted) {
            line_search_restarts++;
            if (line_search_restarts <= m_settings.max_line_search_restarts) {
                st.w.setIdentity();
                st.nupd = 0;
                state = SearchState::Setup;
                initialize_direction(g, st, state, true);

                if (st.delmax < m_settings.step_tol) {
                    result.converged = true;
                    result.termination_reason = "Step size below tolerance";
                    reached_max_iter = false;
                    break;
                }

                if (std::abs(st.gd1) < gd_tol) {
                    result.converged = true;
                    result.termination_reason = "Directional derivative below tolerance";
                    reached_max_iter = false;
                    break;
                }
                continue;
            }

            result.termination_reason =
                "Line search failed to find acceptable step";
            reached_max_iter = false;
            break;
        }
    }

    if (result.termination_reason.empty()) {
        if (reached_max_iter) {
            result.termination_reason = "Maximum iterations reached";
        } else if (!result.converged) {
            result.termination_reason = "Terminated";
        }
    }

    result.x = x;
    result.final_energy = fx;
    result.final_gradient = g;
    return result;
}

void MSTMIN::initialize_direction(const Vec& g, InternalState& st,
                                  SearchState& state, bool reset_search) {
    st.del.noalias() = -st.w * g;

    st.delmax = 0.0;
    st.imax = 0;
    for (int i = 0; i < st.del.size(); ++i) {
        const double v = std::abs(st.del[i]);
        if (v > st.delmax) {
            st.delmax = v;
            st.imax = i + 1;
        }
    }

    st.gd1 = g.dot(st.del);
    st.stdsc = false;

    double factor = 1.0;

    if (st.delmax < kTiny) {
        st.del.setZero();
        st.gd1 = 0.0;
        st.alpha = 1.0;
        st.alpha2 = 1.0;
        st.alpha3 = 0.0;
        st.alpha4 = 0.0;
        st.alpham = 1.0;
        st.ireset = 0;
        state = SearchState::Setup;
        if (reset_search) {
            st.nsrch = 0;
        }
        return;
    }

    if (st.gd1 >= 0.0) {
        if (m_settings.steepest_descent_on_positive_gd) {
            st.del = -g;
            st.gd1 = -g.squaredNorm();
            st.delmax = st.del.cwiseAbs().maxCoeff();
            st.stdsc = true;

            factor = std::min(1.0, m_settings.max_displacement /
                                       std::max(st.delmax, kTiny));
            st.delmax *= factor;
        } else {
            factor = std::min(1.0, m_settings.max_displacement /
                                       std::max(st.delmax, kTiny));
            st.delmax *= factor;
            st.stdsc = st.gd1 > 0.0;
            if (st.stdsc) {
                factor = -factor;
            }
        }
    } else {
        factor = std::min(1.0, m_settings.max_displacement /
                                   std::max(st.delmax, kTiny));
        st.delmax *= factor;
    }

    st.alpham = m_settings.max_displacement / std::max(st.delmax, kTiny);
    st.gd1 *= factor;
    st.del *= factor;

    st.alpha = 1.0;
    st.alpha2 = 1.0;
    st.alpha3 = 0.0;
    st.alpha4 = 0.0;
    st.ireset = 0;
    st.negcur = false;
    state = SearchState::Setup;

    if (reset_search) {
        st.nsrch = 0;
    }
}

MSTMIN::QuadRoots MSTMIN::quadratic_roots(double x1, double y1,
                                          double x2, double y2,
                                          double x3, double y3) {
    QuadRoots out;

    const double z1 = (x3 - x2) * y1;
    const double z2 = (x3 - x1) * y2;
    const double z3 = (x2 - x1) * y3;

    const double a = 2.0 * (z3 - z2 + z1);
    const double b = -(x1 + x2) * z3 + (x1 + x3) * z2 - (x2 + x3) * z1;
    const double c = x1 * x2 * z3 - x1 * x3 * z2 + x2 * x3 * z1;

    if (std::abs(a) < kTiny) {
        out.root1 = x2;
        out.root2 = x2;
        out.has_real_roots = false;
        return out;
    }

    const double disc = b * b - 2.0 * a * c;
    if (disc < 0.0) {
        out.root1 = -b / a;
        out.root2 = -b / a;
        out.has_real_roots = false;
    } else {
        const double sqrt_disc = std::sqrt(disc);
        out.root1 = (-b + sqrt_disc) / a;
        out.root2 = (-b - sqrt_disc) / a;
        out.has_real_roots = true;
    }

    return out;
}

void MSTMIN::linear_search_step(double& alpha, double& alpha2, double& alpha3,
                                double& alpha4, double& gd2, double& gd3,
                                double& gd4, double alpham) const {
    const double denom = gd2 - gd4;
    if (std::abs(denom) < kTiny) {
        alpha = 0.5 * (alpha3 - alpha2);
        alpha2 = alpha3 + alpha;
    } else {
        const double alpha0 = (alpha4 * gd2 - alpha2 * gd4) / denom;
        alpha = alpha0 - alpha2;
        gd3 = gd2;
        alpha3 = alpha2;
        alpha2 = alpha0;
    }

    if (std::abs(alpha2) > std::abs(alpham)) {
        alpha2 = std::copysign(std::abs(alpham), alpha2);
        alpha = alpha2 - alpha3;
    }
}

bool MSTMIN::quadratic_interpolation(double& alpha, double& alpha2, double& alpha3,
                                     double& alpha4, double& gd2, double& gd3,
                                     double& gd4, double alphap) const {
    const QuadRoots q = quadratic_roots(alpha2, gd2, alpha3, gd3, alpha4, gd4);

    if (gd2 <= 0.0) {
        gd4 = gd2;
        alpha4 = alpha2;
    } else {
        gd3 = gd2;
        alpha3 = alpha2;
    }

    int iroot = 1;
    if (q.root1 > alpha4 && q.root1 < alpha3) iroot += 1;
    if (q.root2 > alpha4 && q.root2 < alpha3) iroot += 2;

    double alpha0 = 0.0;
    if (iroot == 2) {
        alpha0 = q.root1;
    } else if (iroot == 3) {
        alpha0 = q.root2;
    } else if (iroot == 4) {
        alpha0 = 0.5 * (q.root1 + q.root2);
    } else {
        return false;
    }

    alpha = alpha0 - alphap;
    alpha2 = alpha0;
    return std::isfinite(alpha2);
}

bool MSTMIN::quadratic_extrapolation(double& alpha, double& alpha2, double& alpha3,
                                     double& alpha4, double& gd2, double& gd3,
                                     double& gd4, double alpham,
                                     SearchState& state) const {
    const QuadRoots q = quadratic_roots(alpha2, gd2, alpha3, gd3, alpha4, gd4);

    double alpha0 = q.root1;

    if (!q.has_real_roots) {
        if (alpha0 > alpham) {
            alpha0 = alpham;
        } else {
            state = SearchState::ForceUpdate;
        }
    } else {
        int iroot = 1;
        if (q.root1 > alpha2) iroot += 1;
        if (q.root2 > alpha2) iroot += 2;

        if (iroot == 1) {
            alpha4 = alpha3;
            gd4 = gd3;
            linear_search_step(alpha, alpha2, alpha3, alpha4,
                               gd2, gd3, gd4, alpham);
            alpha0 = alpha2;
        } else if (iroot == 2) {
            alpha0 = q.root1;
        } else if (iroot == 3) {
            alpha0 = q.root2;
        } else {
            alpha0 = std::min(q.root1, q.root2);
        }
        alpha0 = std::min(alpha0, alpham);
    }

    alpha4 = alpha3;
    gd4 = gd3;
    alpha3 = alpha2;
    gd3 = gd2;
    alpha2 = alpha0;
    alpha = alpha2 - alpha3;

    return std::isfinite(alpha2);
}

bool MSTMIN::negative_curvature_step(double& alpha, double& alpha2, double& alpha3,
                                     double& alpha4, double gd1, double& gd2,
                                     double& gd3, double& gd4, double& alpham,
                                     bool& negcur, SearchState& state,
                                     int& nsrch) const {
    negcur = true;
    nsrch++;
    gd4 = gd1;
    alpha4 = 0.0;

    if (nsrch > m_settings.max_negative_curvature &&
        alpha2 - alpham > -m_settings.step_tol) {
        return false;
    }

    if (alpha2 - alpham > -m_settings.step_tol) {
        alpham *= 2.0;
    }

    alpha *= 2.0;
    alpha3 = alpha2;
    gd3 = gd2;
    alpha2 = alpha3 + alpha;

    if (alpha2 > alpham) {
        alpha2 = alpham;
        alpha = alpha2 - alpha3;
    }

    state = SearchState::NegativeCurvature;
    return true;
}

} // namespace occ::mults
