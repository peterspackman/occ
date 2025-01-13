#pragma once
#include <occ/qm/cint_interface.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <vector>

namespace occ::qm::detail {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

// Helper functions to make chained calls to std::max more clear
inline double max_of(double p, double q, double r) {
  return std::max(p, std::max(q, r));
}

inline double max_of(double p, double q, double r, double s) {
  return std::max(p, std::max(q, std::max(r, s)));
}

// Handles initialization of result matrices for different spin cases
template <SpinorbitalKind sk>
std::vector<MatTriple> initialize_result_matrices(size_t nbf, size_t nthreads) {
  auto [rows, cols] = occ::qm::matrix_dimensions<sk>(nbf);
  std::vector<MatTriple> results(nthreads);
  for (auto &r : results) {
    r.x = Mat::Zero(rows, cols);
    r.y = Mat::Zero(rows, cols);
    r.z = Mat::Zero(rows, cols);
  }
  return results;
}

template <SpinorbitalKind sk>
void accumulate_operator_symmetric(const Mat &source, Mat &dest) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    dest.noalias() += (source + source.transpose());
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    auto source_a = occ::qm::block::a(source);
    auto source_b = occ::qm::block::b(source);
    occ::qm::block::a(dest).noalias() += (source_a + source_a.transpose());
    occ::qm::block::b(dest).noalias() += (source_b + source_b.transpose());
  } else if constexpr (sk == SpinorbitalKind::General) {
    auto source_aa = occ::qm::block::aa(source);
    auto source_ab = occ::qm::block::ab(source);
    auto source_ba = occ::qm::block::ba(source);
    auto source_bb = occ::qm::block::bb(source);
    occ::qm::block::aa(dest).noalias() += (source_aa + source_aa.transpose());
    occ::qm::block::ab(dest).noalias() += (source_ab + source_ab.transpose());
    occ::qm::block::ba(dest).noalias() += (source_ba + source_ba.transpose());
    occ::qm::block::bb(dest).noalias() += (source_bb + source_bb.transpose());
  }
}

} // namespace occ::qm::detail
