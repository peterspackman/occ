#include <CLI/App.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/parallel.h>
#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/main/pair_energy.h>
#include <occ/main/single_point.h>

occ::core::Molecule make_molecule(const std::string &element) {
    auto el = occ::core::Element(element);
    return occ::core::Molecule({{el.atomic_number(), 0.0, 0.0, 0.0}});
}

occ::qm::Wavefunction get_wavefunction(occ::core::Molecule molecule, int charge) {
    occ::io::OccInput config;
    config.method.name = "b3lyp";
    config.basis.name = "def2-svp";
    molecule.set_charge(charge);

    if(molecule.num_electrons() % 2 != 0) {
	molecule.set_multiplicity(2);
    }
    config.electronic.charge = molecule.charge();
    config.electronic.multiplicity = molecule.multiplicity();
    config.geometry.set_molecule(molecule);
    return occ::main::single_point_calculation(config);
}

int main(int argc, char *argv[]) {
    occ::timing::start(occ::timing::category::global);
    occ::timing::start(occ::timing::category::io);
    occ::log::setup_logging(2);
    std::string symbol_a, symbol_b;
    int min_charge_cation{0}, min_charge_anion{0};
    int max_charge_cation{0}, max_charge_anion{0};
    int alternative_multiplicity{2};

    CLI::App app("make_atomic_pair_potentials - generate (charged) pair potentials");
    app.allow_config_extras(CLI::config_extras_mode::error);
    app.set_config("--config", "occ_input.toml",
                   "Read configuration from an ini or TOML file", false);

    app.set_help_all_flag("--help-all", "Show help for all sub commands");
    app.add_option("cation", symbol_a, "Element A for cation")->required();
    app.add_option("anion", symbol_b, "Element B for anion")->required();
    app.add_option("--cation-max-charge", max_charge_cation, "Maximum (positive) charge for anion (provide a positive number)");
    app.add_option("--anion-max-charge", max_charge_anion, "Maximum (negative) charge for anion (provide a positive number)");

    auto *threads_option = app.add_flag_function(
        "--threads{1}",
        [](int num_threads) {
            occ::parallel::set_num_threads(std::max(1, num_threads));
        },
        "number of threads");
    threads_option->default_val(1);
    threads_option->run_callback_for_default();
    threads_option->force_callback();

    // logging verbosity
    auto *verbosity_option = app.add_flag_function(
        "--verbosity{2}",
        [](int verbosity) { occ::log::setup_logging(verbosity); },
        "logging verbosity {0=silent,1=minimal,2=normal,3=verbose,4=debug}");
    verbosity_option->default_val(2);
    verbosity_option->run_callback_for_default();
    verbosity_option->force_callback();

    constexpr auto *error_format = "exception:\n    {}\nterminating program.\n";
    try {
        CLI11_PARSE(app, argc, argv);
	occ::log::info("Pair potential: {} ... {}", symbol_a, symbol_b);
	auto cation = make_molecule(symbol_a);
	auto anion = make_molecule(symbol_b);

	std::vector<occ::qm::Wavefunction> cation_wfns, anion_wfns;

	for(int i = min_charge_cation; i <= max_charge_cation; i++) {
	    cation_wfns.push_back(get_wavefunction(cation, i));
	}

	for(int i = min_charge_anion; i <= max_charge_anion; i++) {
	    anion_wfns.push_back(get_wavefunction(anion, -i));
	}

	occ::log::info("have {} cation, {} anion wavefunctions", cation_wfns.size(), anion_wfns.size());

	std::vector<double> separations{1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0};

	for(double sep: separations) {
	    for(const auto &wfn_a: cation_wfns) {
		for(const auto &wfn_b: anion_wfns) {
		    occ::log::info("{:s}{:+d} {:s}{:+d} (sep: {:.3f})",
			symbol_a, wfn_a.charge(), 
			symbol_b, wfn_b.charge(),
			sep);
		}
	    }
	}


    } catch (const char *ex) {
        occ::log::error(error_format, ex);
        spdlog::dump_backtrace();
        return 1;
    } catch (std::string &ex) {
        occ::log::error(error_format, ex);
        spdlog::dump_backtrace();
        return 1;
    } catch (std::exception &ex) {
        occ::log::error(error_format, ex.what());
        spdlog::dump_backtrace();
        return 1;
    } catch (...) {
        occ::log::error("Exception:\n- Unknown...\n");
        spdlog::dump_backtrace();
        return 1;
    }

    occ::timing::stop(occ::timing::global);
    occ::timing::print_timings();
    occ::log::info("A job well done");
    return 0;
}
