#include <fmt/core.h>
#include <occ/core/timings.h>

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
    case ints1e: return "integrals (one-electron)";
    case ints2e: return "integrals (two-electron)";
    case io: return "file input/output";
    case la: return "linear algebra";
    case guess: return "Initial guess";
    case mo: return "MO update";
    case diis: return "DIIS extrapolation";
    case grid_init: return "DFT grid init";
    case grid_points: return "DFT grid points";
    case dft: return "DFT functional evaluation";
    case gto: return "GTO evaluation (overall)";
    case gto_dist: return "GTO dist evaluation";
    case gto_mask: return "GTO mask evaluation";
    case gto_shell: return "GTO evaluation";
    case gto_s: return "GTO S-function eval";
    case gto_p: return "GTO P-function eval";
    case gto_gen: return "GTO higher order eval";
    case fock: return "Fock build";
    case df: return "Density fitting";
    case solvent: return "Solvation";
    case global: return "Global (total time)";
    case jkmat: return "J+K matrix";
    case jmat: return "J matrix";
    case engine_construct: return "libint2 eng.";
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
        guess,
        mo,
        diis,
        grid_init,
        grid_points,
        dft,
        gto,
        gto_dist,
        gto_mask,
        gto_shell,
        gto_s,
        gto_p,
        gto_gen,
        engine_construct,
        jmat,
        jkmat,
        fock,
        df,
        solvent,
        global,
    };
    fmt::print("Wall clock time by category (s)\n");
    for(const auto& cat : categories)
    {
        auto t = total(cat);
        if(t > 0) {
            fmt::print("{:<30s} {:12.6f}\n", category_name(cat), t);
        }
    }
}

}
