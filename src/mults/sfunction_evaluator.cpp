#include <occ/mults/sfunction_evaluator.h>
#include <occ/mults/sfunctions.h>
#include <stdexcept>

namespace occ::mults {

SFunctionEvaluator::SFunctionEvaluator(int max_rank)
    : m_max_rank(max_rank),
      m_binomial(max_rank + 4)
{
    if (max_rank < 0 || max_rank > 5) {
        throw std::invalid_argument("max_rank must be in [0, 5]");
    }
}

void SFunctionEvaluator::set_coordinates(const Vec3& ra, const Vec3& rb) {
    m_coords = CoordinateSystem::from_points(ra, rb);

    if (m_coords.r < 1e-15) {
        throw std::runtime_error("Sites too close: r < 1e-15");
    }
}

void SFunctionEvaluator::set_coordinate_system(const CoordinateSystem& coords) {
    m_coords = coords;

    if (m_coords.r < 1e-15) {
        throw std::runtime_error("Sites too close: r < 1e-15");
    }
}

SFunctionResult SFunctionEvaluator::compute(int t1, int t2, int j, int deriv_level) {
    // For now, create temporary SFunctions instance and call it
    // Later we'll move the logic directly into this class
    SFunctions sf(m_max_rank);

    // CRITICAL: Use set_coordinate_system() to preserve body-frame transformations!
    // The m_coords may have been created with from_body_frame() which includes
    // orientation matrix information needed for S-function derivatives.
    sf.set_coordinate_system(m_coords);

    auto old_result = sf.compute_s_function(t1, t2, j, deriv_level);

    // Convert from old SFunctions::SFunctionResult format to new SFunctionResult format
    SFunctionResult result;
    result.s0 = old_result.s0;

    if (deriv_level >= 1) {
        // Copy ALL first derivatives (15 elements: 6 for unit vectors + 9 for orientation)
        for (int i = 0; i < 15; ++i) {
            result.s1[i] = old_result.s1[i];
        }
    }

    return result;
}

std::vector<SFunctionResult> SFunctionEvaluator::compute_batch(
    const SFunctionTermList& term_list,
    int deriv_level)
{
    std::vector<SFunctionResult> results;
    results.reserve(term_list.size());

    // Simple implementation: call compute() for each term
    // Later we can optimize this with vectorization
    for (const auto& term : term_list.terms) {
        results.push_back(compute(term.t1, term.t2, term.j, deriv_level));
    }

    return results;
}

std::pair<int, int> SFunctionEvaluator::index_to_lm(int index) {
    // Delegate to existing SFunctions implementation
    SFunctions sf_temp(0);  // Temporary instance just for static method access
    return sf_temp.index_to_lm(index);
}

int SFunctionEvaluator::lm_to_index(int l, int m) {
    // Orient's multipole index mapping
    // l=0 (charge): m=0 -> index=0
    // l=1 (dipole): m=-1,0,1 -> index=1,2,3 (y,z,x in Orient order)
    // l=2 (quadrupole): m=-2,-1,0,1,2 -> index=4,5,6,7,8
    // etc.

    // Start index for rank l: sum from i=0 to l-1 of (2i+1)
    int start_index = l * l;

    // Orient's m ordering: -l, -l+1, ..., -1, 0, 1, ..., l-1, l
    // But with special handling for dipole (y,z,x ordering)
    if (l == 1) {
        // Dipole special case: Orient uses y,z,x order (m=-1,0,1)
        if (m == -1) return start_index + 0; // y component
        if (m == 0) return start_index + 1;  // z component
        if (m == 1) return start_index + 2;  // x component
    }

    // Standard ordering: m goes from -l to l
    return start_index + (m + l);
}

SFunctionResult SFunctionEvaluator::compute_internal(int t1, int t2, int j, int deriv_level) {
    // This is a placeholder for future direct implementation
    // For now, just delegate to compute()
    return compute(t1, t2, j, deriv_level);
}

} // namespace occ::mults
