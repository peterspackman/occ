#include <occ/core/combinations.h>

namespace occ::core {

Combinations::Combinations(int n, int r) : m_n(n), m_r(r) {
    for (int x = 0; x < m_r; x++)
        m_result.push_back(x);
}

bool next_combination(int pivot, std::vector<int> &v, int k, int N) {
    ++v[pivot];
    for (int i = pivot + 1; i < k; ++i)
        v[i] = v[pivot] + i - pivot;
    return true;
}

int select_next_pivot(const std::vector<int> &v, int k, int N) {
    int pivot = k - 1;
    while (pivot >= 0 && v[pivot] == N - k + pivot)
        --pivot;
    return pivot;
}

const Combinations::Result &Combinations::next() {
    if (m_count > 0) {
        next_combination(m_pivot, m_result, m_r, m_n);
    }
    m_count++;
    m_pivot = select_next_pivot(m_result, m_r, m_n);
    if (m_pivot == -1)
        m_completed = true;
    return m_result;
}

bool Combinations::is_completed() const { return m_completed; }
int Combinations::num_generated() const { return m_count; }

} // namespace occ::core
