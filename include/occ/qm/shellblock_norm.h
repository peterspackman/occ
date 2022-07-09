#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

template <SpinorbitalKind sk, Shell::Kind kind>
inline Mat shellblock_norm(const AOBasis &basis, const Mat &matrix) noexcept {
    occ::timing::start(occ::timing::category::la);
    const auto nsh = basis.size();
    const auto &first_bf = basis.first_bf();
    Mat result(nsh, nsh);

    for (size_t s1 = 0; s1 < nsh; ++s1) {
        const auto &s1_first = first_bf[s1];
        const auto &s1_size = basis[s1].size();
        for (size_t s2 = 0; s2 < nsh; ++s2) {
            const auto &s2_first = first_bf[s2];
            const auto &s2_size = basis[s2].size();

            if constexpr (sk == SpinorbitalKind::Restricted) {
                result(s1, s2) =
                    matrix.block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                const auto alpha =
                    occ::qm::block::a(matrix)
                        .block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
                const auto beta =
                    occ::qm::block::b(matrix)
                        .block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
                result(s1, s2) = std::max(alpha, beta);
            } else if constexpr (sk == SpinorbitalKind::General) {
                const auto aa = occ::qm::block::aa(matrix)
                                    .block(s1_first, s2_first, s1_size, s2_size)
                                    .lpNorm<Eigen::Infinity>();
                const auto bb = occ::qm::block::bb(matrix)
                                    .block(s1_first, s2_first, s1_size, s2_size)
                                    .lpNorm<Eigen::Infinity>();
                const auto ab = occ::qm::block::ab(matrix)
                                    .block(s1_first, s2_first, s1_size, s2_size)
                                    .lpNorm<Eigen::Infinity>();
                const auto ba = occ::qm::block::ba(matrix)
                                    .block(s1_first, s2_first, s1_size, s2_size)
                                    .lpNorm<Eigen::Infinity>();
                result(s1, s2) = std::max(aa, std::max(ab, std::max(ba, bb)));
            }
        }
    }
    occ::timing::stop(occ::timing::category::la);
    return result;
}

} // namespace occ::qm
