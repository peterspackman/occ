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
#include <occ/xdm/xdm.h>
#include <scn/scan.h>

namespace occ::main {

void print_charges(const std::string &charge_type, const Vec &charges,
                   const std::vector<occ::core::Atom> &atoms) {
    log::info("{:-<72s}", fmt::format("{} charges (au)  ", charge_type));
    for (int i = 0; i < charges.rows(); i++) {
        log::info("{:<6s} {: 9.6f}",
                  core::Element(atoms[i].atomic_number).symbol(), charges(i));
    }
}

void calculate_dispersion(const OccInput &config, const Wavefunction &wfn) {
    if (!config.dispersion.evaluate_correction)
        return;
    log::info("{:=^72s}", "  Dispersion Correction  ");
    log::info("Method: {}", "XDM");
    occ::xdm::XDM xdm_calc(
        wfn.basis, wfn.charge(),
        {config.dispersion.xdm_a1, config.dispersion.xdm_a2});
    auto energy = xdm_calc.energy(wfn.mo);
    log::info("a1                           {:>20.12f}",
              xdm_calc.parameters().a1);
    log::info("a2                           {:>20.12f}",
              xdm_calc.parameters().a2);
    log::info("Energy            	    {:>20.12f} (Hartree)", energy);
    log::info("Corrected total energy      {:>20.12f} (Hartree)\n",
              energy + wfn.energy.total);
    Vec charges = xdm_calc.hirshfeld_charges();
    print_charges("Hirshfeld", charges, wfn.atoms);
}

void calculate_properties(const OccInput &config, const Wavefunction &wfn) {
    log::info("{:=^72s}", "  Converged Properties  ");

    occ::qm::HartreeFock hf(wfn.basis);

    Vec3 com = hf.center_of_mass();

    log::info("Center of Mass {:12.6f} {:12.6f} {:12.6f}\n", com.x(), com.y(),
              com.z());
    auto mult = hf.template compute_multipoles<4>(wfn.mo, com);

    log::info("{:-<72s}", "Molecular Multipole Moments (au)  ");
    log::info("{}", mult.to_string());

    Vec charges = wfn.mulliken_charges();
    print_charges("Mulliken", charges, wfn.atoms);

    bool do_chelpg = false;
    if (do_chelpg) {
        Vec charges_chelpg = occ::qm::chelpg_charges(wfn);
        print_charges("CHELPG", charges_chelpg, wfn.atoms);
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
            int idx{0};
            auto scan_result = scn::scan<int>(line, "{}");
	    auto & num = scan_result->value();
            points = Mat3N(3, num);
            // comment line
            while (std::getline(is, line) && num > 0) {
                auto result = scn::scan<double, double, double>(line, "{} {} {}");
		auto &[x, y, z] = result->values();
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
