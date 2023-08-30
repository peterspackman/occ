#pragma once
#include <fmt/format.h>
#include <ankerl/unordered_dense.h>
#include <vector>

namespace occ::core {

/**
 * Storage class for components of energy, separated by the dot character.
 *
 * Really just a convenience wrapper around a hash map.
 */
class EnergyComponents : public ankerl::unordered_dense::map<std::string, double> {
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
};

} // namespace occ::core

template <> struct fmt::formatter<occ::core::EnergyComponents> {
    char presentation{'f'};

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'e'))
            presentation = *it++;

        if (it != end && *it != '}')
            throw format_error("invalid format");

        return it;
    }

    template <typename FormatContext>
    auto format(const occ::core::EnergyComponents &e, FormatContext &ctx) {
        auto fmt_string = fmt::format("{{:32s}} {{:20.12{}}}\n", presentation);
        std::string result = fmt::format("\n{:32s} {:>20s}\n\n", "Component",
                                         "Energy (Hartree)");

        ankerl::unordered_dense::set<std::string> printed;

        auto cats = e.categories();
        for (const auto &c : cats) {
            result += fmt::format("{:-<72s}\n", c + "  ");
            for (const auto &component : e) {
                if (printed.find(component.first) != printed.end())
                    continue;

                if (component.first.rfind(c, 0) == 0) {
                    result += fmt::format(fmt_string, component.first,
                                          component.second);
                    printed.insert(component.first);
                }
            }
        }

        result += fmt::format("\n\n");
        for (const auto &component : e) {
            if (printed.find(component.first) == printed.end())
                result +=
                    fmt::format(fmt_string, component.first, component.second);
        }
        return format_to(ctx.out(), result);
    }
};
