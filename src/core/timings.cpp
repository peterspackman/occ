#include <occ/core/timings.h>
#include <fmt/core.h>

namespace occ::timing {

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
    case gto_dist: return "gto dist evaluation";
    case gto_mask: return "gto mask evaluation";
    case gto_shell: return "gto shell evaluation";
    case gto_s: return "S shell evaluation";
    case gto_p: return "P shell evaluation";
    case gto_d: return "D shell evaluation";
    case gto_f: return "F shell evaluation";
    case gto_g: return "G shell evaluation";
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
        gto_dist,
        gto_mask,
        gto_shell,
        gto_s,
        gto_p,
        gto_d,
        gto_f,
        gto_g,
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
