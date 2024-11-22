#pragma once
#include <vector>

namespace occ::core {

/**
 * Class to generate all possible combinations of a integers.
 *
 * The role of the Combinations class is to serve as a generator
 * for all possible combinations (i.e. permutations where order is not
 * important) of integers up to some maximum, of a specific length.
 *
 */

class Combinations {
public:
  using Result = std::vector<int>;
  /**
   * Combinations constructor
   *
   * \param n maximum possible value for the integer (exclusive, i.e. range
   * will be [0,n))
   *
   * \param r length of the combination sequences
   *
   * Internally, combinations holds a std::vector<int> of length r,
   * and returns a const reference to this for each generated sequence.
   *
   * For example to generate all combinations of length 3
   * of containing integers in the range [0, 10) i.e. up to 9, you would call
   *
   * ```
   * Combinations com(10, 3);
   *
   * while(!com.is_completed()) {
   *
   *   const auto &c = com.next();
   *   // do something with c
   *
   * }
   *
   * // the total number of combinations generated
   * int count = com.num_generated();
   *
   * ```
   *
   */
  Combinations(int n, int r);

  /**
   * Check if the sequence of unique combinations is exhausted/completed.
   *
   * \returns True if there are no more combinations to generate
   */
  bool is_completed() const;

  /**
   * The current total count of unique combinations generated
   *
   * \returns an int representing the number of combinations generated.
   */
  int num_generated() const;

  /**
   * The next combination in the sequence
   *
   * \returns a const reference to a std::vector<int> containing the next
   * combination.
   */
  const Result &next();

private:
  int m_n{0};
  int m_r{0};
  bool m_completed{false};
  int m_count{0};
  int m_pivot{0};
  Result m_result;
};

} // namespace occ::core
