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
    void evaluate(size_t shell_idx,
                  const Eigen::Ref<const Mat3N>& C,
                  Eigen::Ref<MatRM> integrals,
                  Eigen::Ref<MatRM> workspace);

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit instantiation declarations (definitions in esp.cpp)
extern template class ESPEvaluator<float>;
extern template class ESPEvaluator<double>;

} // namespace occ::ints
