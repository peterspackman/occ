#pragma once
#include <ankerl/unordered_dense.h>
#include <fmt/format.h>
#include <vector>

namespace occ::core {

/**
 * Storage class for components of energy, separated by the dot character.
 *
 * Really just a convenience wrapper around a hash map.
 */
class EnergyComponents
    : public ankerl::unordered_dense::map<std::string, double> {
  public:
    /**
     * The categories in this set of energy components
     *
     * \returns a std::vector<std::string> of category names
     *
     * Category names are determined by finding keys separated by the `.`
     * character, for example:
     *
     * ```
     * EnergyComponents comps{
     *   {"electronic.1e", 0.5},
     *   {"electronic.2e", 0.25},
     *   {"nuclear.relativistic", 0.2},
     *   {"exchange", 0.3},
     * };
     *
     * // will return a vector of {"electronic", "nuclear"}
     * auto c = comps.categories();
     *
     *
     * ```
     */
    std::vector<std::string> categories() const;
    std::string to_string() const;
};

} // namespace occ::core
