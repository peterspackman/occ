#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <fmt/format.h>

namespace occ::core {

inline constexpr unsigned int num_multipole_components(int L)
{
    unsigned int n{0};
    for(unsigned int i = 0; i <= L; i++)
    {
        n += (i + 1) * (i + 2) / 2;
    }
    return n;
}

inline constexpr std::array<const char *, 20> multipole_component_names{
    "chg", "Dx", "Dy", "Dz", "Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "Qzz",
    "Oxxx", "Oxxy", "Oxxz", "Oxyy", "Oxyz", "Oxzz", "Oyyy", "Oyyz", "Oyzz", "Ozzz"
};

template <unsigned int L>
struct Multipole
{
    static constexpr unsigned int num_components{num_multipole_components(L)};
    std::array<double, num_components> components;
    double charge() const { return components[0]; }

    std::array<double, 3> dipole() const
    {
        static_assert(L > 0, "No dipole for a multipole with angular momentum < 1");
        return {
            components[1],
            components[2],
            components[3]
        };
    }

    std::array<double, 6> quadrupole() const
    {
        static_assert(L > 1, "No quadrupole for a multipole with angular momentum < 2");
        return {
            components[4],
            components[5],
            components[6],
            components[7],
            components[8],
            components[9]
        };
    }

    std::array<double, 10> octupole() const
    {
        static_assert(L > 2, "No octupole for a multipole with angular momentum < 3");
        return {
            components[10],
            components[11],
            components[12],
            components[13],
            components[14],
            components[15],
            components[16],
            components[17],
            components[18],
            components[19]
        };
    }

    template<unsigned int L2>
    Multipole<std::max(L, L2)> operator +(const Multipole<L2> &rhs) const
    {
        constexpr unsigned int LM = std::max(L, L2);
        Multipole<LM> result;
        std::fill(result.components.begin(), result.components.end(), 0.0);
        for(unsigned int i = 0; i < num_components; i++)
        {
            result.components[i] += components[i];
        }

        for(unsigned int i = 0; i < rhs.num_components; i++)
        {
            result.components[i] += rhs.components[i];
        }
        return result;
    }
};

}

template<unsigned int L>
struct fmt::formatter<occ::core::Multipole<L>>
{
    char presentation = 'f';
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) 
    {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;

        if (it != end && *it != '}')
            throw format_error("invalid format");
        return it;
    }

    template <typename FormatContext>
    auto format(const occ::core::Multipole<L>& m, FormatContext& ctx) -> decltype(ctx.out())
    {
        constexpr auto& names = occ::core::multipole_component_names;
        if constexpr(L == 0)
        {
            return format_to(
                ctx.out(),
                presentation == 'f' ? 
                    "{:5s} = {:12.6f}\n" :
                    "{:5s} = {:12.6e}\n",
                names[0],
                m.components[0]
            );
        }
        else if constexpr(L == 1)
        {
            return format_to(
                ctx.out(),
                presentation == 'f' ? 
                    "{:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" :
                    "{:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n",
                names[0], m.components[0],
                names[1], m.components[1],
                names[2], m.components[2],
                names[3], m.components[3]
            );
        }
        else if constexpr(L == 2)
        {
            return format_to(
                ctx.out(),
                presentation == 'f' ? 
                    "{:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" : 
                    "{:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n",
                names[0], m.components[0],
                names[1], m.components[1],
                names[2], m.components[2],
                names[3], m.components[3],
                names[4], m.components[4],
                names[5], m.components[5],
                names[6], m.components[6],
                names[7], m.components[7],
                names[8], m.components[8],
                names[9], m.components[9]
            );
        }
        else if constexpr(L == 3)
        {
            return format_to(
                ctx.out(),
                presentation == 'f' ? 
                    "{:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n"
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" 
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" 
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" 
                    "{:5s} = {:12.6f} {:5s} = {:12.6f} {:5s} = {:12.6f}\n" 
                    "{:5s} = {:12.6f}\n" :
                    "{:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e} {:5s} = {:12.6e} {:5s} = {:12.6e}\n"
                    "{:5s} = {:12.6e}\n",
                names[0], m.components[0],
                names[1], m.components[1],
                names[2], m.components[2],
                names[3], m.components[3],
                names[4], m.components[4],
                names[5], m.components[5],
                names[6], m.components[6],
                names[7], m.components[7],
                names[8], m.components[8],
                names[9], m.components[9],
                names[10], m.components[10],
                names[11], m.components[11],
                names[12], m.components[12],
                names[13], m.components[13],
                names[14], m.components[14],
                names[15], m.components[15],
                names[16], m.components[16],
                names[17], m.components[17],
                names[18], m.components[18],
                names[19], m.components[19]
            );
        }

    }
};
