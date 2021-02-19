#include <filesystem>
#include <tonto/core/logger.h>
#include <tonto/core/molecule.h>
#include <tonto/core/timings.h>
#include <tonto/core/eem.h>
#include <tonto/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/os.h>
#include <tonto/solvent/cosmo.h>
#include <tonto/solvent/surface.h>
#include <tonto/core/units.h>
#include <tonto/io/fchkwriter.h>
#include <tonto/dft/dft.h>
#include <tonto/qm/scf.h>
#include <tonto/qm/ints.h>
#include <tonto/io/fchkreader.h>


namespace fs = std::filesystem;
using tonto::chem::Molecule;
using tonto::chem::Element;
using tonto::timing::StopWatch;
using tonto::solvent::COSMO;
using tonto::qm::Wavefunction;
using tonto::scf::SCF;
using tonto::qm::SpinorbitalKind;


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
    Molecule m = tonto::chem::read_xyz_file(filename);

    fmt::print("Method: '{}'\n", config.method);
    fmt::print("Basis set: '{}'\n", config.basis_name);

    tonto::qm::BasisSet basis(config.basis_name, m.atoms());
    basis.set_pure(false);
    fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
    Wavefunction wfn;
    tonto::dft::DFT rks(config.method, basis, m.atoms(), SpinorbitalKind::Restricted);
    SCF<tonto::dft::DFT, SpinorbitalKind::Restricted> scf(rks);
    scf.set_charge_multiplicity(config.charge, config.multiplicity);
    scf.start_incremental_F_threshold = 0.0;
    double e = scf.compute_scf_energy();
    wfn = scf.wavefunction();
    return wfn;
}

tonto::Vec compute_esp(const Wavefunction &wfn, const tonto::Mat3N &points)
{
    tonto::ints::shellpair_list_t shellpair_list;
    tonto::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = tonto::ints::compute_shellpairs(wfn.basis);
    return tonto::ints::compute_electric_potential(wfn.D, wfn.basis, shellpair_list, points);
}


int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("tonto");
    parser.add_argument("input").help("Input file geometry");
    tonto::log::set_level(tonto::log::level::debug);
    spdlog::set_level(spdlog::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }

    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();
    InputConfiguration config;
    config.geometry_filename = parser.get<std::string>("input");

    Molecule m = tonto::chem::read_xyz_file(config.geometry_filename);
    fs::path fchk_path = config.geometry_filename;
    fchk_path.replace_extension(".fchk");


    fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n", config.geometry_filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    Wavefunction wfn;
    if(fs::exists(fchk_path)) {
        using tonto::io::FchkReader;
        FchkReader fchk(fchk_path.string());
        wfn = Wavefunction(fchk);
    }
    else {
        wfn = run_from_xyz_file(config);
        tonto::io::FchkWriter fchk(fchk_path.string());
        fchk.set_title(fmt::format("{} {}/{} generated by tonto-ng", fchk_path.stem(), config.method, config.basis_name));
        fchk.set_method(config.method);
        fchk.set_basis_name(config.basis_name);
        wfn.save(fchk);
        fchk.write();
    }


    StopWatch<1> sw;
    tonto::Mat3N pos = m.positions();
    tonto::IVec nums = m.atomic_numbers();
    tonto::Vec radii = m.vdw_radii();
    radii.array() += 1.2;

    sw.start(0);
    auto surface = tonto::solvent::surface::solvent_surface(radii, nums, pos);
    sw.stop(0);
    fmt::print("Surface calculated in {}\n", sw.read(0));
    sw.clear_all();

    tonto::Vec areas = surface.areas;
    tonto::Mat3N points = surface.vertices;
    sw.start(0);
    tonto::Vec charges = compute_esp(wfn, points);
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
    auto result = cosmo(points, areas, charges);
    auto vout = fmt::output_file("points.txt");
    vout.print("{}", points.transpose());
    auto cout = fmt::output_file("charges.txt");
    cout.print("{}", charges);
    auto vaout = fmt::output_file("areas.txt");
    vaout.print("{}", areas);
    fmt::print("Surface area: {}\n", areas.sum());

    fmt::print("Total energy: {} kcal/mol\n", tonto::units::AU_TO_KCAL_PER_MOL * result.energy);

    return 0;
}
