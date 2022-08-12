#pragma once
#include <vector>

namespace occ::core {

class Combinations {
  public:
    using Result = std::vector<int>;
    Combinations(int n, int r);

    bool is_completed() const;
    int num_generated() const;

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
