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

constexpr double kjmol_per_hartree{2625.46};

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
    load_matrix(monomers[0]["rotation"], p.a.rotation);
    load_vector(monomers[0]["translation"], p.a.translation);

    p.b.wfn = load_wavefunction(source_b);
    load_matrix(monomers[1]["rotation"], p.b.rotation);
    load_vector(monomers[1]["translation"], p.b.translation);

    p.a.wfn.apply_transformation(p.a.rotation, p.a.translation);
    p.b.wfn.apply_transformation(p.b.rotation, p.b.translation);
    return p;
}

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
               interaction_energy.coulomb * kjmol_per_hartree);
    fmt::print("Exchange-repulsion    {: 12.6f}\n",
               interaction_energy.exchange_repulsion * kjmol_per_hartree);
    fmt::print("Polarization          {: 12.6f}\n",
               interaction_energy.polarization * kjmol_per_hartree);
    fmt::print("Dispersion            {: 12.6f}\n",
               interaction_energy.dispersion * kjmol_per_hartree);
    fmt::print("__________________________________\n");
    fmt::print("Total {:^8s}        {: 12.6f}\n", model_name,
               interaction_energy.total * kjmol_per_hartree);

    fmt::print("\n");
    occ::timing::print_timings();
}
