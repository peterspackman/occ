#include <occ/core/log.h>
#include <occ/io/cube.h>
#include <occ/io/xyz.h>
#include <occ/core/units.h>
#include <occ/main/occ_cube.h>
#include <occ/main/point_functors.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/eeq.h>
#include <scn/scan.h>
#include <occ/qm/wavefunction.h>

namespace fs = std::filesystem;
using occ::Vec;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Element;
using occ::core::Molecule;
using occ::qm::Wavefunction;
using occ::io::Cube;


namespace occ::main {

void require_wfn(const CubeConfig &config, bool have_wfn) {
    if(have_wfn) return;
    throw std::runtime_error("Property requires a wavefunction: " + config.property);
}


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
    cube->add_option("--functional", config->functional,
		    "DFT functional for XC density [default=blyp]");


    cube->add_option("-n,--steps", config->steps, "Number of steps in each direction")->expected(0, 3)->default_val("{}");

    cube->add_option("--points", config->points_filename,
                    "list of points/mesh file requesting points to evaluate the property");
    cube->add_option("--output,-o", config->output_filename,
                    "destination to write file");

    cube->add_option("--origin", config->origin,
		     "origin/starting point for grid")->expected(0,3)->default_val("{}");

    cube->add_option("--direction-a,--direction_a,--da,-a", config->da,
                     "Translation in direction A")->expected(0,3)->default_val("{}");
    cube->add_option("--direction-b,--direction_b,--db,-b", config->db,
                     "Translation in direction B")->expected(0,3)->default_val("{}");
    cube->add_option("--direction-c,--direction_c,--dc,-c", config->dc,
                     "Translation in direction C")->expected(0,3)->default_val("{}");


    cube->fallthrough();
    cube->callback([config]() { run_cube_subcommand(*config); });
    return cube;

}

inline void fail_with_error(const std::string &msg, int line) {
    throw std::runtime_error(fmt::format(
        "Unable to parse points file, error: {}, line {}", msg, line));
}

Mat3N read_points_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
	fail_with_error(
	    fmt::format("Failed to open points file '{}'", filename),
	    0
	);
    }

    std::string line;
    std::getline(file, line);

    auto result = scn::scan<int>(line, "{}");
    if (!result) {
	fail_with_error(result.error().msg(), 0);
    }
    int n = result->value();


    Mat3N points(3, n);

    for (int i = 0; i < n; ++i) {
	std::getline(file, line);
	auto scan_result = scn::scan<double, double, double>(line, "{} {} {}");
	if (!scan_result) {
	    fail_with_error(scan_result.error().msg(), i + 1);
	}
	auto [px, py, pz] = scan_result->values();
        points(0, i) = px;
        points(1, i) = py;
        points(2, i) = pz;
    }

    return points;
}

Mat3N load_points_for_evaluation(const std::string &points_filename) {
    return read_points_file(points_filename);
}

void write_results(CubeConfig const &config, const Mat3N &points, const Vec& data) {
    auto output = fmt::output_file(config.output_filename, fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    output.print("{}\n", data.rows());
    for(int i = 0; i < data.rows(); i++) {
	output.print("{:20.12f} {:20.12f} {:20.12f} {:20.12f}\n",
		     points(0, i), points(1, i), points(2, i), data(i));
    }

}

void evaluate_custom_points(const Wavefunction &wfn, CubeConfig const &config, bool have_wfn) {

    auto points = load_points_for_evaluation(config.points_filename);
    Vec data = Vec::Zero(points.cols());

    if(config.mo_number > -1) {
	occ::log::info("Specified MO number:    {}", config.mo_number);
    }
    occ::timing::start(occ::timing::category::cube_evaluation);
    if(config.property == "eeqesp") {
	EEQEspFunctor func(wfn.atoms);
	func(points, data);
    }
    else if(config.property == "promolecule") {
	PromolDensityFunctor func(wfn.atoms);
	func(points, data);
    }
    else if(config.property == "deformation_density") {
	require_wfn(config, have_wfn);
	DeformationDensityFunctor func(wfn);
	func(points, data);
    }
    else if(config.property == "rho" || config.property == "density") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.mo_index = config.mo_number;
	func(points, data);
    }
    else if(config.property == "rho_alpha") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.spin = SpinConstraint::Alpha;
	func.mo_index = config.mo_number;
	func(points, data);
    }
    else if(config.property == "rho_beta") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.spin = SpinConstraint::Beta;
	func.mo_index = config.mo_number;
	func(points, data);
    }
    else if (config.property == "esp") {
	require_wfn(config, have_wfn);
	EspFunctor func(wfn);
	func(points, data);
    }
    else if (config.property == "xc") {
	require_wfn(config, have_wfn);
	XCDensityFunctor func(wfn, config.functional);
	func(points, data);
    }
    occ::timing::stop(occ::timing::category::cube_evaluation);

    write_results(config, points, data);
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

    occ::log::info("System has {} atoms", wfn.atoms.size());

    if(!config.points_filename.empty()) {
	evaluate_custom_points(wfn, config, have_wfn);
	return;
    }

    Cube cube;
    cube.origin = Vec3::Zero();

    auto fill_values = [](const auto &vector, auto &dest) {
	if(vector.size() == 1) {
	    dest.setConstant(vector[0]);
	}
	else {
	    for(int i = 0; i < std::min(size_t{3}, vector.size()); i++) {
		dest(i) = vector[i];
	    }
	}
    };
    auto fill_direction = [&cube](const auto &vector, int pos) {
	if(vector.size() == 1) {
	    cube.basis(pos, pos) = vector[0];
	}
	else {
	    for(int i = 0; i < std::min(size_t{3}, vector.size()); i++) {
		cube.basis(i, pos) = vector[i];
	    }
	}
    };
    cube.atoms = wfn.atoms;

    cube.basis = Mat3::Identity();

    fill_values(config.origin, cube.origin);
    fill_values(config.steps, cube.steps);

    fill_direction(config.da, 0);
    fill_direction(config.db, 1);
    fill_direction(config.dc, 2);

    cube.name = fmt::format("Generated by OCC from file: {}", config.input_filename);
    cube.description = fmt::format("Scalar values for property '{}'", config.property);

    occ::log::info("Cube geometry");

    occ::log::info("Origin: {:12.5f} {:12.5f} {:12.5f}",
		   cube.origin(0), cube.origin(1), cube.origin(2));
    occ::log::info("A:{: 5d} {:12.5f} {:12.5f} {:12.5f}",
		    cube.steps(0), 
		    cube.basis(0, 0), cube.basis(1, 0), cube.basis(2, 0));
    occ::log::info("B:{: 5d} {:12.5f} {:12.5f} {:12.5f}",
		    cube.steps(1), 
		    cube.basis(0, 1), cube.basis(1, 1), cube.basis(2, 1));
    occ::log::info("C:{: 5d} {:12.5f} {:12.5f} {:12.5f}",
		    cube.steps(2), 
		    cube.basis(0, 2), cube.basis(1, 2), cube.basis(2, 2));

    if(config.mo_number > -1) {
	occ::log::info("Specified MO number:    {}", config.mo_number);
    }
    occ::timing::start(occ::timing::category::cube_evaluation);
    if(config.property == "eeqesp") {
	EEQEspFunctor func(wfn.atoms);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "promolecule") {
	PromolDensityFunctor func(wfn.atoms);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "deformation_density") {
	require_wfn(config, have_wfn);
	DeformationDensityFunctor func(wfn);
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.mo_index = config.mo_number;
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho_alpha") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.spin = SpinConstraint::Alpha;
	func.mo_index = config.mo_number;
	cube.fill_data_from_function(func);
    }
    else if(config.property == "rho_beta") {
	require_wfn(config, have_wfn);
	ElectronDensityFunctor func(wfn);
	func.spin = SpinConstraint::Beta;
	func.mo_index = config.mo_number;
	cube.fill_data_from_function(func);
    }
    else if (config.property == "esp") {
	require_wfn(config, have_wfn);
	EspFunctor func(wfn);
	cube.fill_data_from_function(func);
    }
    else if (config.property == "xc") {
	require_wfn(config, have_wfn);
	XCDensityFunctor func(wfn, config.functional);
	cube.fill_data_from_function(func);
    }
    occ::timing::stop(occ::timing::category::cube_evaluation);


    occ::timing::start(occ::timing::category::io);
    cube.write_data_to_file(config.output_filename);
    occ::timing::stop(occ::timing::category::io);

}

} // namespace occ::main
