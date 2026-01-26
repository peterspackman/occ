#pragma once
#include <occ/gto/gto.h>
#include <ankerl/unordered_dense.h>
#include <optional>

namespace occ::qm {

using gto::AOBasis;

/// Statistics about the auto auxiliary basis generation
struct AutoAuxResult {
    AOBasis aux_basis;
    ankerl::unordered_dense::map<int, int> candidates_per_l;
    ankerl::unordered_dense::map<int, int> selected_per_l;
    double time_seconds{0.0};
};

/// Generate automatic auxiliary basis using pivoted Cholesky decomposition
/// Based on Lehtola, J. Chem. Theory Comput. 2021, 17, 6886-6900
AutoAuxResult generate_auto_aux(
    const AOBasis& basis,
    double threshold = 1e-7,
    std::optional<int> max_l = std::nullopt);

} // namespace occ::qm
