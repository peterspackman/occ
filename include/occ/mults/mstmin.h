#pragma once

#include <functional>
#include <string>
#include <occ/core/linear_algebra.h>

namespace occ::mults {

/**
 * @brief Settings for DMACRYS-style MSTMIN quasi-Newton optimizer.
 */
struct MSTMINSettings {
    double max_displacement = 0.05;    ///< Maximum component displacement per cycle
    int max_updates = 1000;            ///< Rebuild inverse Hessian after this many updates
    double usalp = 0.0;                ///< DMACRYS update aggressiveness factor
    double step_tol = 1e-5;            ///< Converge when max |step component| < tol
    double gradient_tol = 1e-5;        ///< Converge when ||g|| < tol
    double directional_tol_factor = 1e-4; ///< gd tolerance = step_tol * factor
    double energy_tol = 1e-10;         ///< Converge when |dE| < tol
    bool enforce_energy_decrease = false; ///< Reject uphill trial steps during line search
    double energy_increase_tol_abs = 1e-10; ///< Allow tiny absolute uphill noise in line search
    double energy_increase_tol_rel = 1e-12; ///< Allow tiny relative uphill noise in line search
    int max_line_search = 80;          ///< Safety cap for line-search iterations
    int max_line_search_restarts = 3;  ///< Auto-restarts after line-search failure (Hessian reset)
    int max_function_evaluations = 4000; ///< Global cap on objective evaluations (prevents long stalls)
    int max_negative_curvature = 20;   ///< Maximum persistent negative-curvature expansions
    bool steepest_descent_on_positive_gd = false; ///< DMACRYS STDC option
};

/**
 * @brief Result from MSTMIN minimization.
 */
struct MSTMINResult {
    Vec x;                             ///< Final parameters
    double final_energy = 0.0;         ///< Final objective value
    Vec final_gradient;                ///< Final gradient
    int iterations = 0;                ///< Number of accepted updates
    int function_evaluations = 0;      ///< Total objective evaluations
    bool converged = false;            ///< Whether optimization converged
    std::string termination_reason;    ///< Reason for termination
};

/**
 * @brief DMACRYS MSTMIN-style quasi-Newton optimizer.
 */
class MSTMIN {
public:
    using Objective = std::function<double(const Vec&, Vec&)>;
    using Callback = std::function<bool(int, const Vec&, double, const Vec&)>;

    explicit MSTMIN(const MSTMINSettings& settings = {});

    MSTMINResult minimize(Objective f, const Vec& x0, int max_iter = 200);
    MSTMINResult minimize(Objective f, const Vec& x0, Callback callback,
                          int max_iter = 200);

    const MSTMINSettings& settings() const { return m_settings; }
    MSTMINSettings& settings() { return m_settings; }

private:
    enum class SearchState {
        Setup = 2,
        LinearInterpolation = 3,
        Extrapolation = 4,
        NegativeCurvature = 5,
        ForceUpdate = 6
    };

    struct InternalState {
        Vec del;
        Vec g_trial;
        Vec wrk;
        Mat w;

        double gd1 = 0.0;
        double gd3 = 0.0;
        double gd4 = 0.0;

        double alpha = 1.0;
        double alpha2 = 1.0;
        double alpha3 = 0.0;
        double alpha4 = 0.0;
        double alpham = 1.0;

        bool negcur = false;
        bool stdsc = false;

        int ireset = 0;
        int nupd = 0;
        int nsrch = 0;

        double delmax = 0.0;
        int imax = 0;
    };

    struct QuadRoots {
        double root1 = 0.0;
        double root2 = 0.0;
        bool has_real_roots = false;
    };

    void initialize_direction(const Vec& g, InternalState& st,
                              SearchState& state, bool reset_search);

    static QuadRoots quadratic_roots(double x1, double y1,
                                     double x2, double y2,
                                     double x3, double y3);

    void linear_search_step(double& alpha, double& alpha2, double& alpha3,
                            double& alpha4, double& gd2, double& gd3,
                            double& gd4, double alpham) const;

    bool quadratic_interpolation(double& alpha, double& alpha2, double& alpha3,
                                 double& alpha4, double& gd2, double& gd3,
                                 double& gd4, double alphap) const;

    bool quadratic_extrapolation(double& alpha, double& alpha2, double& alpha3,
                                 double& alpha4, double& gd2, double& gd3,
                                 double& gd4, double alpham,
                                 SearchState& state) const;

    bool negative_curvature_step(double& alpha, double& alpha2, double& alpha3,
                                 double& alpha4, double gd1, double& gd2,
                                 double& gd3, double& gd4, double& alpham,
                                 bool& negcur, SearchState& state,
                                 int& nsrch) const;

    MSTMINSettings m_settings;
};

} // namespace occ::mults
