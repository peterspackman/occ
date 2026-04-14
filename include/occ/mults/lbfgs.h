#pragma once
#include <occ/core/linear_algebra.h>
#include <functional>
#include <vector>
#include <string>

namespace occ::mults {

/**
 * @brief Settings for L-BFGS optimizer.
 */
struct LBFGSSettings {
    int memory = 10;              ///< Number of correction pairs to store (m)
    double ftol = 1e-4;           ///< Armijo condition parameter (c1)
    double gtol = 0.9;            ///< Wolfe condition parameter (c2)
    int max_linesearch = 40;      ///< Maximum line search iterations
    double initial_step = 1.0;    ///< Initial step size for line search
    double min_step = 1e-20;      ///< Minimum step size
    double max_step = 1e20;       ///< Maximum step size
    double gradient_tol = 1e-5;   ///< Convergence criterion: ||g|| < tol
    double energy_tol = 1e-10;    ///< Convergence criterion: |f_new - f_old| < tol
    double x_tol = 1e-10;         ///< Convergence criterion: ||x_new - x_old|| < tol
    bool backtracking_only = false; ///< Use simple backtracking (Armijo only, no Wolfe)
    double backtrack_factor = 0.5;  ///< Step reduction factor for backtracking
};

/**
 * @brief Result structure from L-BFGS optimization.
 */
struct LBFGSResult {
    Vec x;                        ///< Final parameters
    double final_energy;          ///< Final objective value
    Vec final_gradient;           ///< Final gradient
    int iterations;               ///< Number of iterations
    int function_evaluations;     ///< Number of function evaluations
    bool converged;               ///< Whether optimization converged
    std::string termination_reason; ///< Reason for termination
};

/**
 * @brief L-BFGS optimizer for unconstrained minimization.
 *
 * Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 * with strong Wolfe line search.
 *
 * Usage:
 * @code
 * LBFGS optimizer(settings);
 * auto result = optimizer.minimize(
 *     [](const Vec& x, Vec& g) {
 *         // Compute f(x) and gradient g
 *         return f;
 *     },
 *     x0, max_iter);
 * @endcode
 */
class LBFGS {
public:
    /// Objective function signature: f(x, gradient) -> energy
    using Objective = std::function<double(const Vec&, Vec&)>;

    /// Optional callback after each iteration: callback(iter, x, f, g) -> continue?
    using Callback = std::function<bool(int, const Vec&, double, const Vec&)>;

    explicit LBFGS(const LBFGSSettings& settings = {});

    /**
     * @brief Minimize the objective function starting from x0.
     *
     * @param f Objective function computing f(x) and storing gradient in g
     * @param x0 Initial parameters
     * @param max_iter Maximum number of iterations
     * @return Optimization result
     */
    LBFGSResult minimize(Objective f, const Vec& x0, int max_iter = 200);

    /**
     * @brief Minimize with iteration callback.
     *
     * @param f Objective function
     * @param x0 Initial parameters
     * @param callback Called after each iteration; return false to stop
     * @param max_iter Maximum number of iterations
     * @return Optimization result
     */
    LBFGSResult minimize(Objective f, const Vec& x0, Callback callback, int max_iter = 200);

    /// Access settings
    const LBFGSSettings& settings() const { return m_settings; }
    LBFGSSettings& settings() { return m_settings; }

private:
    LBFGSSettings m_settings;

    // Storage for L-BFGS correction pairs
    std::vector<Vec> m_s;  // s_k = x_{k+1} - x_k
    std::vector<Vec> m_y;  // y_k = g_{k+1} - g_k
    std::vector<double> m_rho;  // rho_k = 1 / (y_k^T s_k)

    /**
     * @brief Compute search direction using two-loop recursion.
     *
     * Computes H_k * g where H_k is the L-BFGS approximation to the inverse Hessian.
     *
     * @param g Current gradient
     * @param num_stored Number of correction pairs stored
     * @return Search direction d = -H_k * g
     */
    Vec compute_direction(const Vec& g, int num_stored);

    /**
     * @brief Perform line search satisfying strong Wolfe conditions.
     *
     * @param f Objective function
     * @param x Current point
     * @param g Current gradient
     * @param d Search direction
     * @param f0 Current function value
     * @param[out] x_new New point
     * @param[out] g_new Gradient at new point
     * @param[out] f_new Function value at new point
     * @param[out] num_evals Number of function evaluations
     * @return Step size alpha (0 if line search failed)
     */
    double line_search(Objective& f, const Vec& x, const Vec& g, const Vec& d,
                       double f0, Vec& x_new, Vec& g_new, double& f_new,
                       int& num_evals);
};

} // namespace occ::mults
