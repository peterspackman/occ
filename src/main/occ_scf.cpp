#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/io/cifparser.h>
#include <occ/io/crystal_json.h>
#include <occ/io/gaussian_input_file.h>
#include <occ/io/occ_input.h>
#include <occ/io/pc.h>
#include <occ/io/qcschema.h>
#include <occ/io/wavefunction_json.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/xyz.h>
#include <occ/main/cli_validators.h>
#include <occ/main/occ_scf.h>
#include <occ/main/pair_energy.h>
#include <occ/main/properties.h>
#include <occ/main/single_point.h>
#include <occ/main/geometry_optimization.h>
#include <occ/main/version.h>
#include <spdlog/cfg/env.h>
#include <xc.h>

namespace occ::main {

namespace fs = std::filesystem;
using occ::io::OccInput;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

void read_input_file(const std::string &filename, OccInput &config) {
    occ::timing::stop(occ::timing::category::io);
    auto path = fs::path(filename);
    std::string ext = path.extension().string();
    occ::log::debug("Attempting to read input from {}, file extension = {}",
                    filename, ext);
    if (!fs::exists(path))
        throw std::runtime_error("Input file does not exist.");
    if (ext == ".gjf" || ext == ".com") {
        occ::log::debug("Detected Gaussian input file");
        occ::io::GaussianInputFile g(filename);
        g.update_occ_input(config);
    } else if (ext == ".json") {
        occ::log::debug("Detected JSON input");
        occ::io::QCSchemaReader qcs(filename);
        qcs.update_occ_input(config);
    } else if (ext == ".xyz") {
        occ::log::debug("Detected xyz input");
        occ::io::XyzFileReader xyz(filename);
        xyz.update_occ_input(config);
    } else if (ext == ".cif") {
        occ::log::debug("Detected CIF input");
        occ::io::CifParser cif;
        auto crystal = cif.parse_crystal(filename);
        if (!crystal)
            throw std::runtime_error(fmt::format("Could not parse crystal: {}",
                                                 cif.failure_description()));
        config.crystal.asymmetric_unit = (*crystal).asymmetric_unit();
        config.crystal.unit_cell = (*crystal).unit_cell();
        config.crystal.space_group = (*crystal).space_group();
        config.driver.driver = "crystal";
    } else {
        throw std::runtime_error("unknown file type");
    }
}

void write_output_files(const OccInput &config, Wavefunction &wfn) {
    for(const auto &format: config.output.formats) {
	fs::path path = config.filename;
	if (!config.solvent.solvent_name.empty()) {
	    path.replace_extension(fmt::format(".solvated.owf.{}", format));
	} else {
	    path.replace_extension(fmt::format(".owf.{}", format));
	}
	wfn.save(path.string());
    }
}

CLI::App *add_scf_subcommand(CLI::App &app) {
    auto config = std::make_shared<occ::io::OccInput>();

    CLI::App *scf =
        app.add_subcommand("scf", "Perform an SCF on a molecular geometry");
    scf->fallthrough();

    CLI::Option *input_option =
        scf->add_option("input,--geometry-filename,--geometry_filename",
                        config->filename, "input file");
    input_option->check(CLI::ExistingFile);
    input_option->required();

    scf->add_option("method_name,--method", config->method.name, "method name");
    scf->add_option("basis_name,--basis", config->basis.name, "basis set name");
    // electronic
    scf->add_option("-c,--charge", config->electronic.charge,
                    "system net charge");

    scf->add_option("-o,--output", config->output.formats,
                    "output formats");

    auto *multiplicity_option =
        scf->add_option("--multiplicity", config->electronic.multiplicity,
                        "system multiplicity");
    multiplicity_option->check(occ::main::validator::Multiplicity);

    scf->add_flag_function(
        "-u,--unrestricted",
        [&config](int count) {
            config->electronic.spinorbital_kind = SpinorbitalKind::Unrestricted;
        },
        "use unrestricted SCF");

    scf->add_option("--driver",
                    config->driver.driver,
                    "override driver");

    scf->add_option("--basis-set-directory,--basis_set_directory",
                    config->basis.basis_set_directory,
                    "override basis set directory");

    scf->add_option("--integral-precision,--integral_precision",
                    config->method.integral_precision,
                    "cutoff for integral screening");
    // dft grid
    scf->add_option("--dft-grid-max-angular,--dft_grid_max_angular",
                    config->method.dft_grid.max_angular_points,
                    "maximum angular grid points for DFT integration");
    scf->add_option("--dft-grid-min-angular,--dft_grid_min_angular",
                    config->method.dft_grid.min_angular_points,
                    "minimum angular grid points for DFT integration");
    scf->add_option("--dft-grid-radial-precision,--dft_grid_radial_precision",
                    config->method.dft_grid.radial_precision,
                    "radial precision for DFT integration");
    scf->add_option(
        "--dft-grid-reduce-light-elements,--dft_grid_reduce_light_elements",
        config->method.dft_grid.reduced_first_row_element_grid,
        "radial precision for DFT integration");

    // basis set
    scf->add_option("-d,--df-basis,--density_fitting_basis",
                    config->basis.df_name, "basis set");
    scf->add_flag("--spherical", config->basis.spherical,
                  "use spherical basis sets");

    scf->add_option("--orbital-smearing-sigma,--orbital_smearing_sigma",
		    config->method.orbital_smearing_sigma, "Orbital smearing sigma");
    // point charges
    CLI::Option *pc_option = scf->add_option(
        "--point-charges,--point_charge_file",
        config->geometry.point_charge_filename, "file listing point charges");
    pc_option->check(CLI::ExistingFile);

    // Solvation
    scf->add_option("-s,--solvent,--solvent_name", config->solvent.solvent_name,
		    "Solvent name");
    scf->add_option("-f,--solvent-file,--solvent_file",
		    config->solvent.output_surface_filename,
		    "file to write solvent surface");
    scf->add_flag("--solvent-radii-scaling,--solvent_radii_scaling,--draco", config->solvent.radii_scaling,
                  "use DRACO for radii scaling");
    // XDM
    scf->add_flag("--xdm", config->dispersion.evaluate_correction,
                  "use XDM dispersion correction");
    scf->add_option("--xdm-a1,--xdm_a1", config->dispersion.xdm_a1,
                  "a1 parameter for XDM");
    scf->add_option("--xdm-a2,--xdm_a2", config->dispersion.xdm_a2,
                  "a2 parameter for XDM");

    scf->callback([config]() { run_scf_subcommand(*config); });
    return scf;
}

void run_scf_subcommand(occ::io::OccInput &config) {

    occ::main::print_header();

    occ::timing::start(occ::timing::category::io);

    config.name = config.filename;
    // read input file first so we can override with command line settings
    read_input_file(config.filename, config);
    if (config.filename.empty()) {
        config.filename = config.name;
    }
    occ::timing::stop(occ::timing::category::io);

    if (!config.geometry.point_charge_filename.empty()) {
        occ::io::PointChargeFileReader pc(
            config.geometry.point_charge_filename);
        pc.update_occ_input(config);
    }

    occ::log::info("Driver: {}", config.driver.driver);
    if(config.driver.driver == "opt") {
	Wavefunction wfn = optimization_calculation(config);
    }
    else {
	// store solvent name so we can do an unsolvated calculation first
	std::string stored_solvent_name = config.solvent.solvent_name;
	config.solvent.solvent_name = "";

	Wavefunction wfn = occ::main::single_point_calculation(config);
	write_output_files(config, wfn);
	occ::main::calculate_dispersion(config, wfn);
	occ::main::calculate_properties(config, wfn);

	config.solvent.solvent_name = stored_solvent_name;

	if (!config.solvent.solvent_name.empty()) {
	    Wavefunction wfn2 = occ::main::single_point_calculation(config, wfn);
	    double esolv = wfn2.energy.total - wfn.energy.total;

	    occ::log::info("{:-<72s}", "Solvation free energy (SMD)  ");
	    occ::log::info("dG(solv)    = {:20.12f} Hartree", esolv);
	    occ::log::info("            = {:20.12f} kJ/mol",
			   esolv * occ::units::AU_TO_KJ_PER_MOL);
	    occ::log::info("            = {:20.12f} kcal/mol",
			   esolv * occ::units::AU_TO_KCAL_PER_MOL);
	    write_output_files(config, wfn2);
	}
    }
}

} // namespace occ::main
