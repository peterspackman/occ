#pragma once
#include <limits>
#include <occ/core/linear_algebra.h>
#include <occ/qm/coulomb_metric.h>
#include <occ/qm/integral_engine.h>
#include <vector>

namespace occ::qm {

/// Three-centre integrals (μν|P) over the screened canonical shell pairs
/// (q <= p, as `IntegralEngine::shellpairs()`), packed: one contiguous
/// np x nq block per stored shell pair, every auxiliary function a column.
/// Pairs the shell-pair screen drops are never stored, and only one triangle
/// is, so the store is a fraction of the dense nbf x nbf x naux one.
///
/// Under a byte budget only some pairs are stored -- those most expensive to
/// recompute per byte first -- and `unstored()` lists the rest for the caller
/// to compute directly. A complete store is folded with the Coulomb metric as
/// soon as it is built, B = (μν|P) L^-T, so exchange and the correlated
/// methods need no metric solve; a partial store keeps the raw integrals.
class PackedDFStore {
public:
  using ShellPairList = std::vector<std::vector<size_t>>;
  static constexpr size_t unlimited = std::numeric_limits<size_t>::max();
  struct Block {
    int p, q, fp, fq, np, nq;
    size_t row;
  };

  /// Bytes of a store holding every pair of `shellpairs`.
  static size_t bytes_for(const gto::AOBasis &basis,
                          const ShellPairList &shellpairs, size_t naux);

  /// Stores the pairs that fit `max_bytes`, from `engine` at its current
  /// operator, and folds `metric` in when all of them fit.
  void build(IntegralEngine &engine, const CoulombMetric &metric,
             size_t max_bytes = unlimited);
  void clear();

  inline bool built() const { return m_built; }
  inline bool empty() const { return m_blocks.empty(); }
  /// Every screened pair is stored, and the metric is folded in.
  inline bool complete() const { return m_complete; }
  inline size_t bytes() const { return m_ints.size() * sizeof(double); }
  /// Auxiliary columns: fewer than the auxiliary basis once folded with a
  /// metric whose eigendecomposition fallback dropped vectors.
  inline size_t num_aux() const { return m_naux; }
  inline const ShellPairList &unstored() const { return m_unstored; }

  /// Column c: sum over the stored pairs of D_c(μ,ν) B(μν,P), each D_c a
  /// symmetric nbf x nbf matrix.
  Mat contract(const std::vector<Mat> &densities) const;
  /// Matrix c: sum_P B(μν,P) x(P,c) over the stored pairs, symmetric.
  std::vector<Mat> expand(const Mat &x) const;
  /// sum_P (B_P C_s)(B_P C_t)^T for every requested (s, t), on a complete
  /// store. Returns one nbf x nbf matrix per pair.
  std::vector<Mat>
  exchange(const std::vector<Mat> &C,
           const std::vector<std::pair<int, int>> &pairs) const;
  /// B(i*nR + a, P) = sum_μν C_left(μ,i) C_right(ν,a) B_P(μν) on a complete
  /// store, i.e. the metric-folded MO three-index tensor (nL*nR x num_aux).
  Mat half_transform(Eigen::Ref<const Mat> C_left,
                     Eigen::Ref<const Mat> C_right) const;

private:
  void fold(const CoulombMetric &metric);
  /// Writes B_P into the stored blocks of the symmetric nbf x nbf `dense`;
  /// screened positions are never touched, so a zeroed scratch stays zero.
  void unpack(size_t P, Mat &dense) const;

  size_t m_nbf{0}, m_naux{0};
  bool m_built{false}, m_complete{false};
  std::vector<Block> m_blocks;
  ShellPairList m_unstored;
  Mat m_ints;
};

} // namespace occ::qm
