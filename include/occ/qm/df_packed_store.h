#pragma once
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
/// to compute directly. A complete store can be folded with the Coulomb
/// metric, B = (μν|P) L^-T, after which exchange needs no metric solve.
class PackedDFStore {
public:
  using ShellPairList = std::vector<std::vector<size_t>>;
  struct Block {
    int p, q, fp, fq, np, nq;
    size_t row;
  };

  /// Bytes of a store holding every pair of `shellpairs`.
  static size_t bytes_for(const gto::AOBasis &basis,
                          const ShellPairList &shellpairs, size_t naux);

  /// Stores the pairs that fit `max_bytes` (0: all of them).
  void build(IntegralEngine &engine, size_t max_bytes);
  /// B <- B L^-T. Only for a complete store.
  void fold(const CoulombMetric &metric);
  void clear();

  inline bool empty() const { return m_blocks.empty(); }
  inline bool complete() const { return m_complete; }
  inline bool folded() const { return m_folded; }
  inline size_t bytes() const { return m_ints.size() * sizeof(double); }
  inline size_t num_aux() const { return m_naux; }
  inline const ShellPairList &unstored() const { return m_unstored; }

  /// Column c: sum over the stored pairs of D_c(μ,ν) B(μν,P), each D_c a
  /// symmetric nbf x nbf matrix.
  Mat contract(const std::vector<Mat> &densities) const;
  /// Matrix c: sum_P B(μν,P) x(P,c) over the stored pairs, symmetric.
  std::vector<Mat> expand(const Mat &x) const;
  /// sum_P (B_P C_s)(B_P C_t)^T for every requested (s, t), on a folded
  /// store. Returns one nbf x nbf matrix per pair.
  std::vector<Mat>
  exchange(const std::vector<Mat> &C,
           const std::vector<std::pair<int, int>> &pairs) const;

private:
  size_t m_nbf{0}, m_naux{0};
  bool m_complete{false}, m_folded{false};
  std::vector<Block> m_blocks;
  ShellPairList m_unstored;
  Mat m_ints;
};

} // namespace occ::qm
