#pragma once
#include <Eigen/Core>
#include <fmt/core.h>
#include <iterator>
#include <string>
#include <string_view>

namespace occ {

template <typename Derived>
std::string format_matrix(const Eigen::DenseBase<Derived> &matrix,
                          std::string_view fmt_str = "{:12.5f}") {
  const auto &derived = matrix.derived();
  fmt::memory_buffer out;

  // For vectors, always format as a row vector
  const Eigen::Index rows = derived.cols() == 1 ? 1 : derived.rows();
  const Eigen::Index cols =
      derived.cols() == 1 ? derived.rows() : derived.cols();

  // Pre-allocate with rough estimate of size needed
  out.reserve(rows * cols * (fmt_str.size() + 2));

  for (Eigen::Index i = 0; i < rows; ++i) {
    if (i != 0)
      fmt::format_to(std::back_inserter(out), "\n");
    for (Eigen::Index j = 0; j < cols; ++j) {
      if (j != 0)
        fmt::format_to(std::back_inserter(out), " ");
      // For vectors, transpose the access if it's a column vector
      const auto val = derived.cols() == 1 ? derived(j, 0) : derived(i, j);
      fmt::format_to(std::back_inserter(out), fmt::runtime(fmt_str), val);
    }
  }

  return fmt::to_string(out);
}

} // namespace occ
