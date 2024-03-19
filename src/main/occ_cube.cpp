#include <occ/core/log.h>
#include <occ/io/cube.h>
#include <occ/io/xyz.h>
#include <occ/core/units.h>
#include <occ/main/occ_cube.h>
#include <occ/main/point_functors.h>
#include <fmt/os.h>
#include <occ/core/eeq.h>
#include <occ/qm/wavefunction.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Element;
using occ::core::Molecule;
using occ::qm::Wavefunction;
using occ::io::Cube;


namespace occ::main {

CLI::App *add_cube_subcommand(CLI::App &app) {

    CLI::App *cube =
        app.add_subcommand("cube", "compute molecule/qm properties on points");
    auto config = std::make_shared<CubeConfig>();

    cube->add_option("input", config->input_filename,
                    "input geometry file (xyz, wavefunction ...)")
        ->required();

    cube->add_option("property", config->property,
                    "property to evaluate (default=density)");

    cube->add_option("spin", config->spin,
                    "spin (for e.g. electron density) [alpha,beta,default=both]");

    cube->add_option("--mo", config->mo_number,
                    "MO number (for e.g. electron density) [default=-1 i.e. all]");

    cube->add_option("-n,--divisions", config->divisions,
                    "how many times to divide space");
    cube->add_option("--points", config->points_filename,
                    "list of points/mesh file requesting points to evaluate the property");
    cube->add_option("--output,-o", config->output_filename,
                    "destination to write file");

    cube->fallthrough();
    cube->callback([config]() { run_cube_subcommand(*config); });
    return cube;

}

void run_cube_subcommand(CubeConfig const &config) {
    Wavefunction wfn;
    bool have_wfn{false};

    if(Wavefunction::is_likely_wavefunction_filename(config.input_filename)) {
	wfn = Wavefunction::load(config.input_filename);
	occ::log::info("Loaded wavefunction from {}", config.input_filename);
	occ::log::info("Spinorbital kind: {}", occ::qm::spinorbital_kind_to_string(wfn.mo.kind));
	occ::log::info("Num alpha:        {}", wfn.mo.n_alpha);
	occ::log::info("Num beta:         {}", wfn.mo.n_beta);
	occ::log::info("Num AOs:          {}", wfn.mo.n_ao);
	have_wfn = true;
    }
    else {
	Molecule m = occ::io::molecule_from_xyz_file(config.input_filename);
	wfn.atoms = m.atoms();
    }

    occ::log::info("Loaded molecule from xyz file, found {} atoms", wfn.atoms.size());

    Cube cube;
    cube.origin = Vec3(-5.0, -5.0, -5.0);
    cube.atoms = wfn.atoms;

    cube.basis(0, 0) = 10.0 / config.divisions;
    cube.basis(1, 1) = 10.0 / config.divisions;
    cube.basis(2, 2) = 10.0 / config.divisions;

    cube.divisions.setConstant(config.divisions);

    auto require_wfn = [&config, &have_wfn]() {
	if(!have_wfn) throw std::runtime_error("Property requires a wavefunction: " + config.property);
    };

    if(config.property == "eeqesp") {
	EEQEspFunctor func(wfn.atoms);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "promolecule") {
	PromolDensityFunctor func(wfn.atoms);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "deformation_density") {
	require_wfn();
	DeformationDensityFunctor func(wfn);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho") {
	require_wfn();
	ElectronDensityFunctor func(wfn);
	func.mo_index = config.mo_number;
	occ::log::info("MO number:    {}", config.mo_number);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho_alpha") {
	require_wfn();
	ElectronDensityFunctor func(wfn);
	func.spin = ElectronDensityFunctor::Spin::Alpha;
	func.mo_index = config.mo_number;
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho_beta") {
	require_wfn();
	ElectronDensityFunctor func(wfn);
	func.spin = ElectronDensityFunctor::Spin::Beta;
	func.mo_index = config.mo_number;
	cube.fill_data_from_function(func);
    }
    else if (config.property == "esp") {
	require_wfn();
	EspFunctor func(wfn);
	cube.fill_data_from_function(func);
    }


    cube.write_data_to_file(config.output_filename);

}

} // namespace occ::main
