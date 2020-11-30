#include <tonto/core/timings.h>
#include <fmt/core.h>

namespace tonto::timing {

static StopWatch<static_cast<size_t>(category::_group_count)> sw{};

time_point_t start(category cat)
{
    return sw.start(static_cast<size_t>(cat));
}

duration_t stop(category cat)
{
    return sw.stop(static_cast<size_t>(cat));
}

double total(category cat)
{
    return sw.read(static_cast<size_t>(cat));
}

void clear_all()
{
    sw.clear_all();
}

std::string category_name(category cat)
{
    switch(cat) {
    case ints1e: return "integrals 1e";
    case ints2e: return "integrals 2e";
    case io: return "file i/o";
    case la: return "linear algebra";
    case grid_init: return "dft grid init";
    case grid_points: return "dft grid points";
    case dft: return "dft functional";
    case gto: return "gto evaluation";
    case fock: return "fock build";
    case df: return "density fitting";
    case global: return "global";
    default: return "other";
    }
}

void print_timings()
{
    const auto categories = {
        ints1e,
        ints2e,
        io,
        la,
        grid_init,
        grid_points,
        dft,
        gto,
        fock,
        df,
        global
    };
    fmt::print("Wall clock time by category\n");
    for(const auto& cat : categories)
    {
        auto t = total(cat);
        if(t > 0) {
            fmt::print("{:<20s} {:12.6f}s\n", category_name(cat), t);
        }
    }
}

}
