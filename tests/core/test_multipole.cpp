#include <occ/core/multipole.h>
#include "catch.hpp"
#include <fmt/core.h>

using occ::core::Multipole;

TEST_CASE("Multipole constructor", "[multipole]")
{
    auto c = Multipole<0>{};
    auto d = Multipole<1>{};
    auto q = Multipole<2>{};
    auto o = Multipole<3>{};
    fmt::print("Charge\n{}\n", c);
    fmt::print("Dipole\n{}\n", d);
    fmt::print("Quadrupole\n{}\n", q);
    fmt::print("Octupole\n{}\n", o);
    REQUIRE(c.charge() == 0.0);
}

TEST_CASE("Multipole addition", "[multipole]")
{
    auto o = Multipole<3>{
        {1.0,
         0.0, 0.0, 0.5,
         6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
         10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}
    };
    auto q = Multipole<2>{
        {1.0,
         0.0, 0.0, 0.5,
         6.0, 5.0, 4.0, 3.0, 2.0, 1.0}
    };
    auto sum_oq = o + q;
    fmt::print("Result\n{}\n", sum_oq);
    for(unsigned int i = 0; i < o.num_components; i++)
    {
        if(i < q.num_components)
            REQUIRE(sum_oq.components[i] == (o.components[i] + q.components[i]));
        else REQUIRE(sum_oq.components[i] == o.components[i]);
    }
}
