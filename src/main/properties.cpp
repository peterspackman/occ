#include <fmt/core.h>
#include <occ/core/logger.h>
#include <occ/core/units.h>
#include <occ/dft/grid.h>
#include <occ/main/properties.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/hf.h>

namespace occ::main {

void calculate_properties(const OccInput &config, const Wavefunction &wfn) {
    fmt::print("\n{:=^72s}\n\n", "  Converged Properties  ");

    occ::qm::HartreeFock hf(wfn.basis);

    Vec3 com = hf.center_of_mass();

    fmt::print("Center of Mass {:12.6f} {:12.6f} {:12.6f}\n", com.x(), com.y(),
               com.z());
    auto mult = hf.template compute_multipoles<4>(wfn.mo, com);

    fmt::print("\n{:—<72s}\n\n{}\n", "Molecular Multipole Moments (au)  ",
               mult);

    Vec charges = wfn.mulliken_charges();
    fmt::print("\n{:—<72s}\n\n", "Mulliken Charges (au)  ");
    for (int i = 0; i < charges.rows(); i++) {
        fmt::print("{:<6s} {: 9.6f}\n",
                   core::Element(wfn.atoms[i].atomic_number).symbol(),
                   charges(i));
    }
    fmt::print("\n");

    bool do_chelpg = false;
    if (do_chelpg) {
        fmt::print("\n{:—<72s}\n\n", "CHELPG Charges (au)  ");
        Vec charges_chelpg = occ::qm::chelpg_charges(wfn);
        for (int i = 0; i < charges_chelpg.rows(); i++) {
            fmt::print("{:<6s} {: 9.6f}\n",
                       core::Element(wfn.atoms[i].atomic_number).symbol(),
                       charges_chelpg(i));
        }
        fmt::print("\n");
    }
}

} // namespace occ::main
