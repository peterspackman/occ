#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/ostream.h>
#include <occ/core/element.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/fchkreader.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>
#include <nlohmann/json.hpp>

using occ::interaction::CEModelInteraction;
using occ::qm::Wavefunction;
using occ::Mat3;
using occ::Vec3;

struct Pair {
    struct Monomer {
	Wavefunction wfn;
	occ::Mat3 rotation{occ::Mat3::Identity()};
	occ::Vec3 translation{occ::Vec3::Zero()};
    };
    Monomer a;
    Monomer b;
};

Wavefunction load_wavefunction(const std::string &filename) {
    namespace fs = std::filesystem;
    using occ::util::to_lower;
    std::string ext = fs::path(filename).extension();
    to_lower(ext);
    if (ext == ".fchk") {
	using occ::io::FchkReader;
	FchkReader fchk(filename);
	return Wavefunction(fchk);
    }
    if (ext == ".molden") {
	using occ::io::MoldenReader;
	MoldenReader molden(filename);
	return Wavefunction(molden);
    }
    throw std::runtime_error(
	    "Unknown file extension when reading wavefunction: " + ext);
}

void load_matrix(const nlohmann::json &json, Mat3 &mat) {
    for(size_t i = 0; i < json.size(); i++) {
	const auto& row = json.at(i);
	for (size_t j = 0; j < row.size(); j++) {
	    mat(i, j) = row.at(j).get<double>();
	}
    }
}

void load_vector(const nlohmann::json &json, Vec3 &vec) {
    for(size_t i = 0; i < json.size(); i++) {
	vec(i) = json.at(i).get<double>();
    }
}

void print_mo_norms(const Wavefunction &wfn) {
    auto [shellpair_list, shellpair_data] = occ::ints::compute_shellpairs(wfn.basis);

    occ::Mat overlap = occ::ints::compute_1body_ints<occ::ints::Operator::overlap>(wfn.basis, shellpair_list)[0];
    for(size_t i = 0; i < wfn.mo.Cocc.cols(); i++) {
	auto v = wfn.mo.Cocc.col(i);
	double norm = v.transpose().dot(overlap * v);
	fmt::print("norm {} = {}\n", i, norm);
    }
}

Pair parse_input_file(const std::string &filename) {
    std::ifstream i(filename);
    nlohmann::json j;
    i >> j;
    const auto& monomers = j["monomers"];

    if(monomers.size() != 2) throw std::runtime_error("Require two monomers in input file");

    Pair p;
    std::string source_a = monomers[0]["source"];
    std::string source_b = monomers[1]["source"];
    p.a.wfn = load_wavefunction(source_a);
    print_mo_norms(p.a.wfn);
    load_matrix(monomers[0]["rotation"], p.a.rotation);
    load_vector(monomers[0]["translation"], p.a.translation);
    p.a.translation *= occ::units::ANGSTROM_TO_BOHR;
    fmt::print("Rotation A:\n{}\n", p.a.rotation);
    fmt::print("Translation A:\n{}\n", p.a.translation);

    p.b.wfn = load_wavefunction(source_b);
    print_mo_norms(p.b.wfn);
    load_matrix(monomers[1]["rotation"], p.b.rotation);
    load_vector(monomers[1]["translation"], p.b.translation);
    p.b.translation *= occ::units::ANGSTROM_TO_BOHR;
    fmt::print("Rotation B:\n{}\n", p.b.rotation);
    fmt::print("Translation B:\n{}\n", p.b.translation);

    p.a.wfn.apply_transformation(p.a.rotation, p.a.translation);
    p.b.wfn.apply_transformation(p.b.rotation, p.b.translation);
    return p;
}


/*
 *  Example input:
 *  {
 *      "name": "formamide",
 *      "monomers": [
 *          {
 *              "source": "formamide.fchk",
 *              "rotation": [
 *                  [1.0, 0.0, 0.0],
 *                  [0.0, 1.0, 0.0],
 *                  [0.0, 0.0, 1.0]
 *              ],
 *              "translation": [
 *                  0.0, 0.0, 0.0
 *              ]
 *          },
 *          {
 *              "source": "formamide.fchk",
 *              "rotation": [
 *                  [-1.0,  0.0,  0.0],
 *                  [ 0.0, -1.0,  0.0],
 *                  [ 0.0,  0.0, -1.0]
 *              ],
 *              "translation": [
 *                  3.604, 0.0, 0.0
 *              ]
 *          }
 *      ]
 *  }
 *
 */

int main(int argc, char *argv[]) {
    cxxopts::Options options("occ", "A program for quantum chemistry");
    options.positional_help("[input_file]")
	.show_positional_help();

    options.add_options()("h,help", "Print help")(
	    "i,input", "Input file", cxxopts::value<std::string>())(
	    "t,threads", "Number of threads",
	    cxxopts::value<int>()->default_value("1"))(
	    "m,model", "CE model",
	    cxxopts::value<std::string>()->default_value("ce-b3lyp"))(
	    "v,verbosity", "Logging verbosity",
	    cxxopts::value<std::string>()->default_value("WARN"));

    options.parse_positional({"input"});
    auto args = options.parse(argc, argv);

    auto level = occ::log::level::warn;
    if (args.count("verbosity")) {
        std::string level_lower =
            occ::util::to_lower_copy(args["verbosity"].as<std::string>());
        if (level_lower == "debug")
            level = occ::log::level::debug;
        else if (level_lower == "info")
            level = occ::log::level::info;
        else if (level_lower == "error")
            level = occ::log::level::err;
    }
    occ::log::set_level(level);
    spdlog::set_level(level);

    occ::timing::start(occ::timing::category::global);
    libint2::Shell::do_enforce_unit_normalization(true);
    libint2::initialize();

    auto pair = parse_input_file(args["input"].as<std::string>());

    using occ::parallel::nthreads;
    nthreads = args["threads"].as<int>();

    const std::string model_name = args["model"].as<std::string>();

    auto model = occ::interaction::ce_model_from_string(model_name);

    CEModelInteraction interaction(model);
    auto interaction_energy = interaction(pair.a.wfn, pair.b.wfn);
    occ::timing::stop(occ::timing::category::global);

    fmt::print("Component              Energy (kJ/mol)\n\n");
    fmt::print("Coulomb               {: 12.6f}\n",
	    interaction_energy.coulomb_kjmol());
    fmt::print("Exchange-repulsion    {: 12.6f}\n",
	    interaction_energy.exchange_kjmol());
    fmt::print("Polarization          {: 12.6f}\n",
	    interaction_energy.polarization_kjmol());
    fmt::print("Dispersion            {: 12.6f}\n",
	    interaction_energy.dispersion_kjmol());
    fmt::print("__________________________________\n");
    fmt::print("Total {:^8s}        {: 12.6f}\n", model_name,
	    interaction_energy.total_kjmol());

    //fmt::print("\n");
    fmt::print("D norm\n{}\n", pair.a.wfn.mo.D.norm());
    occ::timing::print_timings();
}
