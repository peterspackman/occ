#pragma once
#include <occ/ints/boys.h>
#include <Eigen/Core>
#include <array>
#include <memory>
#include <vector>

namespace occ::ints {

// Forward declarations - implementation details hidden
template <typename T> struct ShellPairData;

/// High-level ESP integral evaluator
///
/// This class manages precomputed shell pair data and workspace buffers
/// for efficient evaluation of ESP integrals over many grid points.
///
/// Usage:
///   ESPEvaluator<double> esp(boys_table);
///   esp.add_shell_pair(la, lb, na_prim, nb_prim, ...);
///   esp.evaluate(shell_idx, grid_points, integrals, workspace);
///
template <typename T>
class ESPEvaluator {
public:
    // Convenient type aliases
    using Mat3N = Eigen::Matrix<T, 3, Eigen::Dynamic>;
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /// Construct evaluator with Boys function table
    explicit ESPEvaluator(const T* boys_table);

    ~ESPEvaluator();

    // Non-copyable, movable
    ESPEvaluator(const ESPEvaluator&) = delete;
    ESPEvaluator& operator=(const ESPEvaluator&) = delete;
    ESPEvaluator(ESPEvaluator&&) noexcept;
    ESPEvaluator& operator=(ESPEvaluator&&) noexcept;

    /// Add a shell pair and precompute its E-matrices (Cartesian)
    /// Returns index for later evaluation
    size_t add_shell_pair(int la, int lb,
                          int na_prim, int nb_prim,
                          const T* exponents_a, const T* exponents_b,
                          const T* coeffs_a, const T* coeffs_b,
                          const T* A, const T* B);

    /// Add a shell pair with option for spherical harmonics
    /// @param spherical  If true, output integrals in spherical basis
    size_t add_shell_pair(int la, int lb,
                          int na_prim, int nb_prim,
                          const T* exponents_a, const T* exponents_b,
                          const T* coeffs_a, const T* coeffs_b,
                          const T* A, const T* B,
                          bool spherical);

    /// Evaluate ESP integrals for a precomputed shell pair (Eigen interface)
    /// @param shell_idx  Index returned by add_shell_pair
    /// @param C          Grid point coordinates [3 x npts] column-major
    /// @param integrals  Output buffer [npts x nab] row-major
    /// @param workspace  External workspace buffer [npts x nherm] row-major
    /// @param omega      Range-separation parameter: 0 = full Coulomb (1/r),
    ///                   omega > 0 = long-range erf(omega*r)/r operator
    void evaluate(size_t shell_idx,
                  const Eigen::Ref<const Mat3N>& C,
                  Eigen::Ref<MatRM> integrals,
                  Eigen::Ref<MatRM> workspace,
                  T omega = T(0));

    /// Structure-of-arrays evaluation, vectorised over points.
    /// @param x,y,z  Point coordinates; npts must be a multiple of
    ///               soa_padding, with padding entries holding any finite
    ///               coordinates (their outputs are computed and ignored)
    /// @param V      Output [nab x ldv], V[ab * ldv + pt] (point fastest)
    /// @param omega  As for evaluate()
    void evaluate_soa(size_t shell_idx, const T *x, const T *y, const T *z,
                      int npts, T *V, int ldv, T omega = T(0)) const;

    /// The pair's integrals contracted as they are formed, for
    /// seminumerical exchange: with V as from evaluate_soa,
    ///   G[(bf_q + b) * ld + i] -= sum_a F[(bf_p + a) * ld + i] V_ab(i)
    /// and, when both, G[(bf_p + a) * ld + i] -= sum_b F[(bf_q + b) * ld + i]
    /// V_ab(i) (a over the pair's first shell, b its second). V is never
    /// stored in full. npts as for evaluate_soa, ld >= npts.
    void evaluate_soa_contract(size_t shell_idx, const T *x, const T *y,
                               const T *z, int npts, const T *F, T *G, int ld,
                               int bf_p, int bf_q, bool both,
                               T omega = T(0)) const;

    /// Shell pairs for centre derivatives (seminumerical exchange
    /// gradients), the same shells as add_shell_pair: resize_derivative_pairs(n)
    /// once, then set_shell_pair_derivative for each idx < n (distinct indices
    /// may be set concurrently). Angular momenta up to 5.
    void resize_derivative_pairs(size_t n);
    void set_shell_pair_derivative(size_t idx, int la, int lb, int na_prim,
                                   int nb_prim, const T *exponents_a,
                                   const T *exponents_b, const T *coeffs_a,
                                   const T *coeffs_b, const T *A, const T *B,
                                   bool spherical);
    /// One pass over a derivative pair giving both its integrals and their
    /// centre derivatives contracted with per-point weights:
    ///   A[ab * lda + i] = (a b | 1/|r - x_i|)           (nab rows, as evaluate_soa)
    ///   out[c] += sum_i sum_ab W[ab * ldw + i] d/dc A_ab(x_i)
    /// c = A_x, A_y, A_z (and B_x, B_y, B_z when ncentres == 2). The derivative
    /// integrals are never stored. W must be zero on padding points.
    void evaluate_soa_value_and_derivative_contraction(
        size_t idx, const T *x, const T *y, const T *z, int npts, T *A,
        int lda, const T *W, int ldw, int ncentres, T *out,
        T omega = T(0)) const;

    /// Shell pairs for second derivatives (seminumerical exchange Hessians),
    /// set like the derivative pairs: resize_hessian_pairs(n) once, then
    /// set_shell_pair_hessian for each idx < n (distinct indices may be set
    /// concurrently). Angular momenta up to 4.
    void resize_hessian_pairs(size_t n);
    void set_shell_pair_hessian(size_t idx, int la, int lb, int na_prim,
                                int nb_prim, const T *exponents_a,
                                const T *exponents_b, const T *coeffs_a,
                                const T *coeffs_b, const T *A, const T *B,
                                bool spherical);
    /// One pass over a Hessian pair, point-fastest like evaluate_soa:
    ///   V[(k * nab + ab) * ldv + i], k = 0 the integrals (a b | 1/|r - x_i|),
    ///   1..3 their d/dA_{x,y,z}, 4..6 d/dB_{x,y,z};
    /// and, unless W is null, second derivatives contracted with per-point
    /// weights, out[k] += sum_i sum_ab W[ab * ldw + i] d2/dc_k (a b | ...),
    /// 21 values: AA (xx, xy, xz, yy, yz, zz), AB (A_d B_e at 3 d + e), BB
    /// (as AA). W must be zero on padding points.
    void evaluate_soa_hessian(size_t idx, const T *x, const T *y, const T *z,
                              int npts, T *V, int ldv, const T *W, int ldw,
                              T *out, T omega = T(0)) const;

    /// Point counts passed to evaluate_soa must be a multiple of this.
    static constexpr int soa_padding = 32;

    /// Get required workspace size for a shell pair
    size_t workspace_size(size_t shell_idx, int npts) const;

    /// Evaluate overlap integrals for a precomputed shell pair
    /// @param shell_idx  Index returned by add_shell_pair
    /// @param integrals  Output buffer [nab]
    void evaluate_overlap(size_t shell_idx, T* integrals);

    /// Get number of basis function pairs for a shell pair
    int nab(size_t shell_idx) const;

    /// Get number of Hermite Gaussians for a shell pair (workspace cols needed)
    int nherm(size_t shell_idx) const;

    /// Get angular momenta for a shell pair
    std::pair<int, int> angular_momenta(size_t shell_idx) const;

    /// Clear all precomputed shell pairs
    void clear();

    /// Number of precomputed shell pairs
    size_t num_shell_pairs() const;

    /// Check if a shell pair uses spherical harmonics
    bool is_spherical(size_t shell_idx) const;

    /// Get the effective center of a shell pair (overlap region)
    /// Based on the most diffuse primitive combination
    std::array<T, 3> pair_center(size_t shell_idx) const;

    /// Get the extent of a shell pair (radius where pair density < 1e-12)
    /// Based on the most diffuse primitive combination
    T pair_extent(size_t shell_idx) const;

    /// Grid-point-independent upper bound on this pair's ESP integrals,
    ///     sum_prim |2 pi / p * c_a * c_b * exp(-mu R_AB^2)|,
    /// i.e. the integral prefactor with the Boys factor F_0 <= 1 bounded out.
    /// Because it drops the only distance-to-grid-point dependence, it bounds
    /// the pair everywhere and is the right magnitude to screen a 1/R kernel
    /// on. Cheaper and tighter than the overlap, which is this times
    /// sqrt(pi/p)/2 and so understates compact pairs badly.
    T pair_estimate(size_t shell_idx) const;

    /// Far-field form of the same bound, {q1, spread}: at a point C with
    /// R = |C - pair_center| > spread the pair's ESP integrals are at most
    /// q1 / (R - spread) (each primitive's F_0(T) <= sqrt(pi/T)/2). The 1/R
    /// tail is kept: this only tightens the bound, it never cuts it off.
    std::array<T, 2> pair_far_estimate(size_t shell_idx) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations (definitions in esp.cpp)
extern template class ESPEvaluator<float>;
extern template class ESPEvaluator<double>;

} // namespace occ::ints
