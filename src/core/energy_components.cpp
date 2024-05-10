#include <occ/core/energy_components.h>
#include <occ/core/util.h>

namespace occ::core {

std::vector<std::string> EnergyComponents::categories() const {
    std::vector<std::string> result;
    for (const auto &term : *this) {
        auto tokens = occ::util::tokenize(term.first, ".");
        if (tokens.size() > 1 && std::find(result.begin(), result.end(),
                                           tokens[0]) == result.end()) {
            result.push_back(tokens[0]);
        }
    }
    return result;
}

std::string EnergyComponents::to_string() const {
    std::string result =
        fmt::format("\n{:32s} {:>20s}\n\n", "Component", "Energy (Hartree)");

    ankerl::unordered_dense::set<std::string> printed;

    auto cats = categories();
    for (const auto &c : cats) {
        result += fmt::format("{:-<72s}\n", c + "  ");
        for (const auto &component : *this) {
            if (printed.find(component.first) != printed.end())
                continue;

            if (component.first.rfind(c, 0) == 0) {
                result += fmt::format("{:32s} {:20.12f}\n", component.first,
                                      component.second);
                printed.insert(component.first);
            }
        }
    }

    result += fmt::format("\n\n");
    for (const auto &component : *this) {
        if (printed.find(component.first) == printed.end())
            result += fmt::format("{:32s} {:20.12f}\n", component.first,
                                  component.second);
    }
    return result;
}
} // namespace occ::core
