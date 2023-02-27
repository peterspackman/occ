#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/grid.h>
#include <occ/main/properties.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/hf.h>
#include <scn/scn.h>

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
    bool do_esp = false;
    // TODO make this flexible
    if (do_esp) {
        const std::string filename{"esp_points.txt"};
        const std::string destination{"esp.txt"};
        Mat3N points;
        {
            std::ifstream is(filename);
            std::string line;
            std::getline(is, line);
            int num{0}, idx{0};
            auto scan_result = scn::scan(line, "{}", num);
            points = Mat3N(3, num);
            // comment line
            double x, y, z;
            while (std::getline(is, line) && num > 0) {
                auto result = scn::scan(line, "{} {} {}", x, y, z);
                if (!result) {
                    occ::log::error("failed reading {}", result.error().msg());
                    continue;
                }
                points(0, idx) = x;
                points(1, idx) = y;
                points(2, idx) = z;
                num--;
                idx++;
            }
        }
        log::info("Computing electric potential at points from '{}'", filename);
        auto output = fmt::output_file(
            destination, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);

        Mat esp(points.cols(), 4);
        esp.leftCols(3) = points.transpose();
        esp.col(3) = wfn.electric_potential(points);
        output.print("{}\n", esp.rows());
        output.print("{}\n", esp);
        log::info("Finishing computing ESP");
    }
}

} // namespace occ::main
