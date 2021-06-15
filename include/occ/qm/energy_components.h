#pragma once
#include <fmt/format.h>
#include <occ/3rdparty/robin_hood.h>

namespace occ::qm
{

class EnergyComponents : public robin_hood::unordered_map<std::string, double> 
{

};

}

template<>
struct fmt::formatter<occ::qm::EnergyComponents>
{
    char presentation{'f'};

    constexpr auto parse(format_parse_context& ctx)
    {
        auto it = ctx.begin(), end = ctx.end();
        if(it != end && (*it == 'f' || *it == 'e')) presentation = *it++;

        if(it != end && *it != '}')
            throw format_error("invalid format");

        return it;
    }

    template<typename FormatContext>
    auto format(const occ::qm::EnergyComponents &e, FormatContext &ctx)
    {
        auto fmt_string = fmt::format("{{:32s}} {{:20.12{}}}\n", presentation);
        std::string result = fmt::format("{:32s} {:>20s}\n\n", "Component", "Energy (Hartree)");

        for(const auto& component: e)
        {
            result += fmt::format(fmt_string, component.first, component.second);
        }
        return format_to(ctx.out(), result);
    }
};
