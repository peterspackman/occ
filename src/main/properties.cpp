#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/grid.h>
#include <occ/main/properties.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/hf.h>

namespace occ::main {

void calculate_properties(const OccInput &config, const Wavefunction &wfn) {
    log::info("{:=^72s}", "  Converged Properties  ");

    occ::qm::HartreeFock hf(wfn.basis);

    Vec3 com = hf.center_of_mass();

    log::info("Center of Mass {:12.6f} {:12.6f} {:12.6f}\n", com.x(), com.y(),
              com.z());
    auto mult = hf.template compute_multipoles<4>(wfn.mo, com);

    log::info("{:—<72s}", "Molecular Multipole Moments (au)  ");
    log::info("{}", mult);

    Vec charges = wfn.mulliken_charges();
    log::info("{:—<72s}", "Mulliken Charges (au)  ");
    for (int i = 0; i < charges.rows(); i++) {
        log::info("{:<6s} {: 9.6f}",
                  core::Element(wfn.atoms[i].atomic_number).symbol(),
                  charges(i));
    }

    bool do_chelpg = false;
    if (do_chelpg) {
        log::info("{:—<72s}", "CHELPG Charges (au)  ");
        Vec charges_chelpg = occ::qm::chelpg_charges(wfn);
        for (int i = 0; i < charges_chelpg.rows(); i++) {
            log::info("{:<6s} {: 9.6f}",
                      core::Element(wfn.atoms[i].atomic_number).symbol(),
                      charges_chelpg(i));
        }
    }
}

} // namespace occ::main
