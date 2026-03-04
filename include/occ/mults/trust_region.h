#pragma once
#include <occ/core/linear_algebra.h>
#include <functional>
#include <string>

namespace occ::mults {

/**
 * @brief Settings for Trust Region Newton optimization.
 */
struct TrustRegionSettings {
    double initial_radius = 1.0;      ///< Initial trust region radius
    double max_radius = 100.0;        ///< Maximum trust region radius
    double eta1 = 0.25;               ///< Accept if actual/predicted >= eta1
    double eta2 = 0.75;               ///< Expand radius if actual/predicted >= eta2
    double gamma1 = 0.25;             ///< Shrink factor when rejected
    double gamma2 = 2.0;              ///< Expand factor when very good
    double gradient_tol = 1e-6;       ///< Convergence: gradient norm
    double step_tol = 1e-8;           ///< Convergence: step norm
    double energy_tol = 1e-10;        ///< Convergence: energy change
    int max_iterations = 200;         ///< Maximum iterations
    int max_cg_iterations = 50;       ///< Max CG iterations for subproblem
    double cg_tol = 0.1;              ///< CG relative tolerance
    bool use_diagonal_scaling = true; ///< Scale by diagonal of Hessian
    bool verbose = false;             ///< Print iteration info

    /// Hessian update settings (for SR1 quasi-Newton updates between full recomputations)
    int hessian_update_interval = 1;  ///< Recompute true Hessian every N iterations (1 = always)
    double sr1_skip_tol = 1e-8;       ///< Skip SR1 update if |s^T y| < tol * |s| * |y|
};

/**
 * @brief Result from Trust Region Newton optimization.
 */
struct TrustRegionResult {
    Vec x;                    ///< Final parameters
    double final_value;       ///< Final objective value
    double final_gradient_norm;  ///< Final gradient norm
    int iterations;           ///< Number of iterations
    int function_evaluations; ///< Number of function evaluations
    int hessian_evaluations;  ///< Number of Hessian evaluations
    bool converged;           ///< Whether converged
    std::string termination_reason;
};

/**
 * @brief Trust Region Newton optimizer with Steihaug-CG subproblem solver.
 *
 * This is a robust second-order optimization method that:
 * - Naturally handles indefinite Hessians (negative curvature)
 * - Uses analytical Hessians when available
 * - Solves the trust region subproblem using Steihaug's truncated CG
 *
 * The trust region subproblem is:
 *   min_p  g^T p + 0.5 p^T H p   subject to ||p|| <= delta
 *
 * Usage:
 * @code
 * TrustRegion optimizer(settings);
 * auto result = optimizer.minimize(objective, hessian, x0);
 * @endcode
 */
class TrustRegion {
public:
    /// Objective function: returns (value, gradient)
    using Objective = std::function<std::pair<double, Vec>(const Vec&)>;

    /// Hessian function: returns Hessian matrix
    using HessianFunc = std::function<Mat(const Vec&)>;

    /// Hessian-vector product: returns H*v (more efficient for large problems)
    using HessianVectorProduct = std::function<Vec(const Vec&, const Vec&)>;

    /// Iteration callback: (iteration, x, value, gradient) -> continue?
    using Callback = std::function<bool(int, const Vec&, double, const Vec&)>;

    explicit TrustRegion(const TrustRegionSettings& settings = {});

    /**
     * @brief Minimize using explicit Hessian matrix.
     *
     * @param objective Function returning (value, gradient)
     * @param hessian Function returning Hessian matrix
     * @param x0 Initial parameters
     * @param callback Optional iteration callback (return false to stop)
     * @return Optimization result
     */
    TrustRegionResult minimize(Objective objective,
                               HessianFunc hessian,
                               const Vec& x0,
                               Callback callback = nullptr);

    /**
     * @brief Minimize using Hessian-vector products (matrix-free).
     *
     * @param objective Function returning (value, gradient)
     * @param hvp Function returning H*v given x and v
     * @param x0 Initial parameters
     * @return Optimization result
     */
    TrustRegionResult minimize_hvp(Objective objective,
                                   HessianVectorProduct hvp,
                                   const Vec& x0);

    /// Gradient-only objective: f(x, grad) -> energy (same signature as LBFGS/MSTMIN)
    using GradObjective = std::function<double(const Vec&, Vec&)>;

    /**
     * @brief Minimize using BFGS Hessian approximation (gradient-only).
     *
     * Starts from identity Hessian, applies BFGS rank-2 updates each
     * accepted step.  No analytic Hessian needed — same interface as LBFGS.
     *
     * @param f Gradient-only objective: f(x, grad) -> energy
     * @param x0 Initial parameters
     * @param callback Optional iteration callback
     * @return Optimization result
     */
    TrustRegionResult minimize_bfgs(GradObjective f, const Vec& x0,
                                    Callback callback = nullptr);

private:
    TrustRegionSettings m_settings;

    /**
     * @brief Solve trust region subproblem using Steihaug-CG.
     *
     * Finds approximate solution to:
     *   min_p  g^T p + 0.5 p^T H p   subject to ||p|| <= delta
     *
     * Returns early if:
     * - Negative curvature detected (moves to boundary)
     * - CG converges
     * - Step reaches trust region boundary
     *
     * @param g Gradient
     * @param H Hessian matrix
     * @param delta Trust region radius
     * @param D Diagonal scaling (optional)
     * @return Step direction p
     */
    Vec solve_subproblem(const Vec& g, const Mat& H, double delta,
                         const Vec& D = Vec());

    /**
     * @brief Solve subproblem using Hessian-vector products.
     */
    Vec solve_subproblem_hvp(const Vec& g,
                             const std::function<Vec(const Vec&)>& Hv,
                             double delta,
                             const Vec& D = Vec());

    /// Compute predicted reduction: -g^T p - 0.5 p^T H p
    double predicted_reduction(const Vec& g, const Mat& H, const Vec& p) const;

    /// Compute predicted reduction using Hessian-vector product
    double predicted_reduction_hvp(const Vec& g,
                                   const std::function<Vec(const Vec&)>& Hv,
                                   const Vec& p) const;

    /// Find intersection of ray with trust region boundary
    double find_boundary_step(const Vec& p, const Vec& d, double delta) const;
};

} // namespace occ::mults
