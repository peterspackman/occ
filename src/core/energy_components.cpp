#include <occ/core/energy_components.h>
#include <occ/core/util.h>

namespace occ::core {

std::vector<std::string> EnergyComponents::categories() const {
    std::vector<std::string> result;
    for (const auto term : *this) {
        auto tokens = occ::util::tokenize(term.first, ".");
        if (tokens.size() > 1 && std::find(result.begin(), result.end(),
                                           tokens[0]) == result.end()) {
            result.push_back(tokens[0]);
        }
    }
    return result;
}

} // namespace occ::core
