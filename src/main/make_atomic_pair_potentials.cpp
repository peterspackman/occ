#include <CLI/App.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/parallel.h>
#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/main/pair_energy.h>
#include <occ/main/single_point.h>
#include <occ/core/progress.h>
#include <occ/qm/hf.h>
#include <fmt/os.h>

using occ::qm::Wavefunction;
using occ::core::Molecule;
using occ::interaction::CEEnergyComponents;
using occ::interaction::CEModelInteraction;
using occ::qm::HartreeFock;

Molecule make_molecule(const std::string &element) {
    auto el = occ::core::Element(element);
    return occ::core::Molecule({{el.atomic_number(), 0.0, 0.0, 0.0}});
}

// copy the molecule so we can modify it
Wavefunction get_wavefunction(Molecule molecule, int charge) {
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

void compute_wfn_monomer_energies(const std::string &basename, std::vector<Wavefunction> &wavefunctions) {
    size_t index = 0;
    for (auto &wfn : wavefunctions) {
	HartreeFock hf(wfn.basis);
	occ::interaction::CEMonomerCalculationParameters params;
	params.Schwarz = hf.compute_schwarz_ints();
	params.xdm = true;

	occ::log::info("Computing monomer energies for {} {}", basename, wfn.charge());
	occ::interaction::compute_ce_model_energies(wfn, hf, params);
	auto output = fmt::output_file(fmt::format("{}_{}.txt", basename, wfn.charge()));
	output.print("Monomer energy terms (Hartree)\n{}\n", wfn.energy.to_string());
	output.print("XDM Polarizability:\n{}\n\n", wfn.xdm_polarizabilities);
	output.print("XDM Moments:\n{}\n\n", wfn.xdm_moments);
	output.print("XDM Volumes:\n{}\n\n", wfn.xdm_volumes);
	output.print("XDM Free volumes:\n{}\n\n", wfn.xdm_free_volumes);
        index++;
    }
}


// copy the wavefunctions so we can modify them
CEEnergyComponents compute_pair_energy(Wavefunction wfn_a,
				       Wavefunction wfn_b,
				       double separation, bool dimer_xdm = true) {

    // assumes that B is at the origin (as is A)
    wfn_b.apply_transformation(occ::Mat3::Identity(), occ::Vec3(0.0, 0.0, separation * occ::units::ANGSTROM_TO_BOHR));

    auto model = occ::interaction::ce_model_from_string("ce-1p");
    CEModelInteraction interaction(model);
    interaction.set_use_xdm_dimer_parameters(dimer_xdm);
    return interaction(wfn_a, wfn_b);
}

int main(int argc, char *argv[]) {
    occ::timing::start(occ::timing::category::global);
    occ::timing::start(occ::timing::category::io);
    occ::log::setup_logging(2);
    std::string symbol_a, symbol_b;
    int min_charge_cation{0}, min_charge_anion{0};
    int max_charge_cation{0}, max_charge_anion{0};
    int alternative_multiplicity{2};
    bool use_xdm_dimer_parameters{true};

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
	    occ::log::info("Wavefunction for {}{:+d}", symbol_a, i);
	    cation_wfns.push_back(get_wavefunction(cation, i));
	}
	compute_wfn_monomer_energies(symbol_a, cation_wfns);

	for(int i = min_charge_anion; i <= max_charge_anion; i++) {
	    occ::log::info("Wavefunction for {}{:+d}", symbol_b, -i);
	    anion_wfns.push_back(get_wavefunction(anion, -i));
	}
	compute_wfn_monomer_energies(symbol_b, anion_wfns);

	occ::log::info("have {} cation, {} anion wavefunctions", cation_wfns.size(), anion_wfns.size());

	std::vector<double> separations{1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0};

	size_t dimers_to_compute = separations.size() * cation_wfns.size() * anion_wfns.size();
	occ::core::ProgressTracker progress(dimers_to_compute);
	size_t computed_dimers{0};
	for(const auto &wfn_a: cation_wfns) {
	    for(const auto &wfn_b: anion_wfns) {
		auto output = fmt::output_file(
		    fmt::format("{}_{}_{}_{}.txt", symbol_a, wfn_a.charge(), symbol_b, wfn_b.charge()));
		output.print("{:>6s}\t{:>20s}\t{:>20s}\t{:>20s}\t{:>20s}\t{:>20s}\t{:>20s}\n", "r_angs",
			     "coulomb", "exchange", "repulsion", "polarization", "dispersion", "total");
		for(double sep: separations) {
		    occ::log::debug("Dimer: {:s}{:+d} {:s}{:+d} (sep: {:.3f} Angs)",
			symbol_a, wfn_a.charge(), 
			symbol_b, wfn_b.charge(),
			sep);

		    auto energy = compute_pair_energy(wfn_a, wfn_b, sep);

		    double e_coul = energy.coulomb_kjmol();
		    double e_exch = energy.exchange_kjmol();
		    double e_rep = energy.repulsion_kjmol();
		    double e_pol = energy.polarization_kjmol();
		    double e_disp = energy.dispersion_kjmol();
		    double e_tot = energy.total_kjmol();

		    occ::log::debug("Component              Energy (kJ/mol)\n");
		    occ::log::debug("Coulomb               {: 12.6f}", e_coul);
		    occ::log::debug("Exchange              {: 12.6f}", e_exch);
		    occ::log::debug("Repulsion             {: 12.6f}", e_rep);
		    occ::log::debug("Polarization          {: 12.6f}", e_pol);
		    occ::log::debug("Dispersion            {: 12.6f}", e_disp);
		    occ::log::debug("__________________________________");
		    occ::log::debug("Total 		      {: 12.6f}", e_tot);
		    computed_dimers++;
		    output.print("{:6.1f}\t{:20.6f}\t{:20.6f}\t{:20.6f}\t{:20.6f}\t{:20.6f}\t{:20.6f}\n", sep, e_coul, e_exch, e_rep, e_pol, e_disp, e_tot);

		    progress.update(computed_dimers, dimers_to_compute, 
                        fmt::format("E[{}{:+d}|{}{:+d}] @ {:.3f}", symbol_a, wfn_a.charge(), symbol_b, wfn_b.charge(), sep));

		}
	    }
	}
	progress.clear();
	occ::log::info("Computed {} pairwise interactions", computed_dimers);

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
