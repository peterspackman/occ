#include <fmt/core.h>
#include <occ/main/properties.h>

namespace occ::main {

void calculate_properties(const OccInput &config, const Wavefunction &wfn) {

    fmt::print("\nproperties\n----------\n");
    /*
    occ::Vec3 origin = m.center_of_mass() * occ::units::ANGSTROM_TO_BOHR;
    fmt::print("center of mass (bohr)        {:12.6f} {:12.6f} {:12.6f}\n\n",
               origin(0), origin(1), origin(2));
    auto e_mult =
        proc.template compute_electronic_multipoles<3>(SK, wfn.D, origin);
    auto nuc_mult = proc.template compute_nuclear_multipoles<3>(origin);
    auto tot_mult = e_mult + nuc_mult;
    fmt::print("electronic multipole\n{}\n", e_mult);
    fmt::print("nuclear multipole\n{}\n", nuc_mult);
    fmt::print("total multipole\n{}\n", tot_mult);
    occ::Vec mulliken_charges =
        -2 * occ::qm::mulliken_partition<SK>(proc.basis(), proc.atoms(), wfn.D,
                                             proc.compute_overlap_matrix());
    fmt::print("Mulliken charges\n");
    for (size_t i = 0; i < wfn.atoms.size(); i++) {
        mulliken_charges(i) += wfn.atoms[i].atomic_number;
        fmt::print("Atom {}: {:12.6f}\n", i, mulliken_charges(i));
    }
    */
    Vec charges = wfn.mulliken_charges();
    fmt::print("Atomic charges:\n");
    for (int i = 0; i < charges.rows(); i++) {
        fmt::print("{:<6s} {: 9.6f}\n",
                   core::Element(wfn.atoms[i].atomic_number).symbol(),
                   charges(i));
    }
}

} // namespace occ::main
