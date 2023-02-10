#include <occ/core/log.h>
#include <occ/core/timings.h>

namespace occ::timing {

static StopWatch<static_cast<size_t>(category::_group_count)> sw{};

time_point_t start(category cat) { return sw.start(static_cast<size_t>(cat)); }

duration_t stop(category cat) { return sw.stop(static_cast<size_t>(cat)); }

double total(category cat) { return sw.read(static_cast<size_t>(cat)); }

void clear_all() { sw.clear_all(); }

const char *category_name(category cat) {
    switch (cat) {
    case ints1e:
        return "integrals (one-electron)";
    case ints4c2e:
        return "4-centre 2-electron integrals";
    case ints3c2e:
        return "3-centre 2-electron integrals";
    case io:
        return "file input/output";
    case la:
        return "linear algebra";
    case guess:
        return "Initial guess";
    case mo:
        return "MO update";
    case diis:
        return "DIIS extrapolation";
    case grid_init:
        return "DFT grid init";
    case grid_points:
        return "DFT grid points";
    case dft:
        return "DFT functional evaluation";
    case rho:
        return "density evaluation";
    case dft_xc:
        return "DFT XC total";
    case xc_func_init:
        return "DFT XC func init";
    case xc_func_end:
        return "DFT XC func close";
    case gto:
        return "GTO evaluation total";
    case gto_dist:
        return "GTO dist evaluation";
    case gto_mask:
        return "GTO mask evaluation";
    case gto_shell:
        return "GTO shell evaluation";
    case gto_s:
        return "GTO S-function eval";
    case gto_p:
        return "GTO P-function eval";
    case gto_gen:
        return "GTO higher order eval";
    case ecp:
        return "ECP integrals";
    case assoc_legendre:
        return "Assoc. Legendre Poly";
    case fft:
        return "FFT";
    case fock:
        return "Fock build";
    case df:
        return "Density fitting";
    case solvent:
        return "Solvation";
    case global:
        return "Global (total time)";
    case jkmat:
        return "J+K matrix";
    case jmat:
        return "J matrix";
    default:
        return "other";
    }
}

void print_timings() {
    const auto categories = {
        ints1e, ints4c2e, ints3c2e, io,     guess,          mo,  diis, dft,
        rho,    dft_xc,   gto,      ecp,    assoc_legendre, fft, jmat, jkmat,
        fock,   df,       solvent,  global,
    };
    log::debug("Wall clock time by category (s)");
    for (const auto &cat : categories) {
        auto t = total(cat);
        if (t > 0) {
            log::debug("{:<30s} {:12.6f}", category_name(cat), t);
        }
    }
}

} // namespace occ::timing
