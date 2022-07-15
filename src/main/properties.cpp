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

    occ::hf::HartreeFock hf(wfn.basis);

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

    fmt::print("\n{:—<72s}\n\n", "CHELPG Charges (au)  ");
    occ::dft::AtomGridSettings grid_settings{38, 38, 10, 1e-12};
    occ::dft::MolecularGrid grid(wfn.basis, grid_settings);
    std::vector<occ::dft::AtomGrid> atom_grids;
    for (size_t i = 0; i < wfn.basis.atoms().size(); i++) {
        atom_grids.push_back(grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        atom_grids.begin(), atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    occ::Mat3N grid_points(3, num_grid_points);
    size_t npt = 0;
    size_t n = 0;
    for (const auto &atom_grid : atom_grids) {
        double r_vdw = occ::units::ANGSTROM_TO_BOHR *
                       core::Element(wfn.atoms[n].atomic_number).vdw();
        const Vec3 pn = {wfn.atoms[n].x, wfn.atoms[n].y, wfn.atoms[n].z};
        for (int i = 0; i < atom_grid.num_points(); i++) {
            double r = (atom_grid.points.col(i) - pn).norm();
            if (r >= r_vdw) {
                grid_points.col(npt) = atom_grid.points.col(i);
                npt++;
            }
        }
        n++;
    }
    grid_points.resize(3, npt);
    occ::log::debug("CHELPG num grid points = {}", npt);
    Vec charges_chelpg = occ::qm::chelpg_charges(wfn, grid_points);
    for (int i = 0; i < charges_chelpg.rows(); i++) {
        fmt::print("{:<6s} {: 9.6f}\n",
                   core::Element(wfn.atoms[i].atomic_number).symbol(),
                   charges_chelpg(i));
    }
    fmt::print("\n");
}

} // namespace occ::main
