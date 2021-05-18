#include <filesystem>
#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/eem.h>
#include <occ/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/os.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/surface.h>
#include <occ/core/units.h>
#include <occ/io/fchkwriter.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>
#include <occ/qm/ints.h>
#include <occ/io/fchkreader.h>
#include <occ/core/parallel.h>


namespace fs = std::filesystem;
using occ::chem::Molecule;
using occ::chem::Element;
using occ::timing::StopWatch;
using occ::solvent::COSMO;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::qm::SpinorbitalKind;


struct InputConfiguration {
    fs::path geometry_filename;
    std::string method{"b3lyp"};
    std::string basis_name{"3-21G"};
    int multiplicity = 1;
    int charge = 0;
};

Wavefunction run_from_xyz_file(const InputConfiguration &config)
{
    std::string filename = config.geometry_filename.string();
    Molecule m = occ::chem::read_xyz_file(filename);

    fmt::print("Method: '{}'\n", config.method);
    fmt::print("Basis set: '{}'\n", config.basis_name);

    occ::qm::BasisSet basis(config.basis_name, m.atoms());
    basis.set_pure(false);
    fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
    Wavefunction wfn;
    occ::dft::DFT rks(config.method, basis, m.atoms(), SpinorbitalKind::Restricted);
    rks.set_system_charge(config.charge);
    SCF<occ::dft::DFT, SpinorbitalKind::Restricted> scf(rks);
    scf.set_charge_multiplicity(config.charge, config.multiplicity);
    scf.start_incremental_F_threshold = 0.0;
    double e = scf.compute_scf_energy();
    wfn = scf.wavefunction();
    return wfn;
}

occ::Vec compute_esp(const Wavefunction &wfn, const occ::Mat3N &points)
{
    occ::ints::shellpair_list_t shellpair_list;
    occ::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = occ::ints::compute_shellpairs(wfn.basis);
    auto dout = fmt::output_file("D.txt");
    dout.print("{}", wfn.D);
    return occ::ints::compute_electric_potential(wfn.D, wfn.basis, shellpair_list, points);
}



int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("occ");
    parser.add_argument("input").help("Input file geometry");
    parser.add_argument("-c", "--charge")
            .help("Molecular charge")
            .default_value(0)
            .action([](const std::string& value) { return std::stoi(value); });

    parser.add_argument("-j", "--threads")
            .help("Number of threads")
            .default_value(2)
            .action([](const std::string& value) { return std::stoi(value); });
    occ::log::set_level(occ::log::level::debug);
    spdlog::set_level(spdlog::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        occ::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }

    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();
    InputConfiguration config;
    config.geometry_filename = parser.get<std::string>("input");
    using occ::parallel::nthreads;
    nthreads = parser.get<int>("--threads");
    config.charge = parser.get<int>("--charge");


    Molecule m = occ::chem::read_xyz_file(config.geometry_filename);
    fs::path fchk_path = config.geometry_filename;
    fchk_path.replace_extension(".fchk");


    fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n", config.geometry_filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    fmt::print("Charge: {}\n", config.charge);

    Wavefunction wfn;
    if(fs::exists(fchk_path)) {
        using occ::io::FchkReader;
        FchkReader fchk(fchk_path.string());
        wfn = Wavefunction(fchk);
    }
    else {
        wfn = run_from_xyz_file(config);
        occ::io::FchkWriter fchk(fchk_path.string());
        fchk.set_title(fmt::format("{} {}/{} generated by occ-ng", fchk_path.stem(), config.method, config.basis_name));
        fchk.set_method(config.method);
        fchk.set_basis_name(config.basis_name);
        wfn.save(fchk);
        fchk.write();
    }


    StopWatch<1> sw;
    occ::Mat3N pos = m.positions() / 0.52917749;
    occ::IVec nums = m.atomic_numbers();
    occ::Vec radii = occ::solvent::cosmo::solvation_radii(nums);
    fmt::print("van der Waals radii:\n{}\n", radii);
    radii.array() /= 0.52917749;

    sw.start(0);
    auto surface = occ::solvent::surface::solvent_surface(radii, nums, pos);
    sw.stop(0);
    fmt::print("Surface ({} atoms, {} points) calculated in {}\n", radii.rows(), surface.areas.rows(), sw.read(0));
    sw.clear_all();

    occ::Vec areas = surface.areas;
    occ::Mat3N points = surface.vertices;
    sw.start(0);
    occ::Vec charges = compute_esp(wfn, points);
    for(size_t i = 0; i < nums.rows(); i++)
    {
        auto p1 = pos.col(i);
        double q = nums(i);
        auto r = (points.colwise() - p1).colwise().norm();
        charges.array() += q / r.array();
    }
    sw.stop(0);
    fmt::print("ESP (range = {}, {}) calculated in {}\n", charges.minCoeff(), charges.maxCoeff(), sw.read(0)); 


    COSMO cosmo(78.40);
    cosmo.set_max_iterations(100);
    cosmo.set_x(0.0);
    auto result = cosmo(points, areas, charges);
    auto vout = fmt::output_file("points.txt");
    vout.print("{}", points.transpose());
    auto cout = fmt::output_file("charges.txt");
    cout.print("{}", charges);
    auto vaout = fmt::output_file("areas.txt");
    vaout.print("{}", areas);
    fmt::print("Surface area: {} angstrom**2\n", areas.sum() * 0.52917749 * 0.52917749);

    fmt::print("Total energy: {} kcal/mol\n", occ::units::AU_TO_KCAL_PER_MOL * result.energy);

    return 0;
}
