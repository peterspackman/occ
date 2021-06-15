#pragma once
#include <fmt/format.h>
#include <occ/3rdparty/robin_hood.h>

namespace occ::core
{

class EnergyComponents : public robin_hood::unordered_map<std::string, double> 
{
public:
    std::vector<std::string> categories() const;
};

}

template<>
struct fmt::formatter<occ::core::EnergyComponents>
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
    auto format(const occ::core::EnergyComponents &e, FormatContext &ctx)
    {
        auto fmt_string = fmt::format("{{:32s}} {{:20.12{}}}\n", presentation);
        std::string result = fmt::format("{:32s} {:>20s}\n\n", "Component", "Energy (Hartree)");

        robin_hood::unordered_map<std::string, bool> printed;

        auto cats = e.categories();
        for(const auto &c : cats)
        {
            result += fmt::format("{}\n-------------------\n", c);
            for(const auto& component: e)
            {
                if(printed[component.first]) continue;

                if(component.first.rfind(c, 0) == 0)
                {
                    result += fmt::format(fmt_string, component.first, component.second);
                    printed[component.first] = true;
                }
            }
            result += '\n';
        }

        for(const auto& component: e)
        {
            if(!printed[component.first])
                result += fmt::format(fmt_string, component.first, component.second);
        }
        return format_to(ctx.out(), result);
    }
};
