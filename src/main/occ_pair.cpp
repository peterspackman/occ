#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/eigen_json.h>
#include <occ/io/fchkreader.h>
#include <occ/io/moldenreader.h>
#include <occ/io/orca_json.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace fs = std::filesystem;

using occ::Mat3;
using occ::Vec3;
using occ::interaction::CEModelInteraction;
using occ::qm::Wavefunction;

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
        occ::log::debug("Loading Gaussian fchk file from {}", filename);
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    } else if (ext == ".molden" || ext == ".input") {
        occ::log::debug("Loading molden file from {}", filename);
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        occ::log::debug("Wavefunction has {} atoms", molden.atoms().size());
        return Wavefunction(molden);
    } else if (ext == ".json") {
        occ::log::debug("Loading Orca JSON file from {}", filename);
        occ::io::OrcaJSONReader json(filename);
        return Wavefunction(json);
    }
    throw std::runtime_error(
        "Unknown file extension when reading wavefunction: " + ext);
}

void save_xdm_parameters(const Wavefunction &wfn, const std::string &filename) {
    std::ofstream out(filename);
    nlohmann::json j;
    j["elements"] = {};
    j["positions"] = {};
    for (const auto &a : wfn.atoms) {
        j["elements"].push_back(a.atomic_number);
        j["positions"].push_back({a.x, a.y, a.z});
    }
    j["volume"] = wfn.xdm_volumes;
    j["moments"] = wfn.xdm_moments;
    j["volume_free"] = wfn.xdm_free_volumes;
    j["polarizabilities"] = wfn.xdm_polarizabilities;
    out << j;
}

void load_matrix(const nlohmann::json &json, Mat3 &mat) {
    for (size_t i = 0; i < json.size(); i++) {
        const auto &row = json.at(i);
        for (size_t j = 0; j < row.size(); j++) {
            mat(i, j) = row.at(j).get<double>();
        }
    }
}

void load_vector(const nlohmann::json &json, Vec3 &vec) {
    for (size_t i = 0; i < json.size(); i++) {
        vec(i) = json.at(i).get<double>();
    }
}

template <typename T>
std::vector<T> load_std_vector(const nlohmann::json &json) {
    std::vector<T> result;
    result.reserve(json.size());
    for (size_t i = 0; i < json.size(); i++) {
        result.push_back(json.at(i).get<double>());
    }
    return result;
}

Pair parse_input_file(const std::string &filename,
                      const std::string &monomer_directory) {
    std::ifstream i(filename);
    nlohmann::json j;
    i >> j;
    const auto &monomers = j["monomers"];

    if (monomers.size() != 2)
        throw std::runtime_error("Require two monomers in input file");

    Pair p;
    std::string source_a =
        (fs::path(monomer_directory) / fs::path(monomers[0]["source"]))
            .string();
    std::string source_b =
        (fs::path(monomer_directory) / fs::path(monomers[1]["source"]))
            .string();
    p.a.wfn = load_wavefunction(source_a);
    load_matrix(monomers[0]["rotation"], p.a.rotation);
    load_vector(monomers[0]["translation"], p.a.translation);
    p.a.translation *= occ::units::ANGSTROM_TO_BOHR;
    occ::log::debug("Rotation A (det = {}):\n{}", p.a.rotation.determinant(),
                    p.a.rotation);
    if (monomers[0].contains("ecp_electrons")) {
        // TODO no more hard coded ecp basis
        auto bs = occ::qm::AOBasis::load(p.a.wfn.atoms, "def2-svp");
        bs.set_pure(p.a.wfn.basis.is_pure());
        const auto &atoms = p.a.wfn.atoms;
        const auto &shells = p.a.wfn.basis.shells();
        const auto &name = p.a.wfn.basis.name();
        const auto &ecp_shells = bs.ecp_shells();
        p.a.wfn.basis = occ::qm::AOBasis(atoms, shells, name, ecp_shells);
        p.a.wfn.basis.set_ecp_electrons(bs.ecp_electrons());
    }
    occ::log::debug("Translation A: {}", p.a.translation.transpose());
    {
        std::ofstream out("a_mo.txt");
        out << p.a.wfn.mo.C;
    }
    {
        std::ofstream out("a_d.txt");
        out << p.a.wfn.mo.D;
    }

    p.b.wfn = load_wavefunction(source_b);
    load_matrix(monomers[1]["rotation"], p.b.rotation);
    load_vector(monomers[1]["translation"], p.b.translation);
    p.b.translation *= occ::units::ANGSTROM_TO_BOHR;
    occ::log::debug("Rotation B (det = {}):\n{}", p.b.rotation.determinant(),
                    p.b.rotation);
    if (monomers[1].contains("ecp_electrons")) {
        // hard coded ecp basis
        auto bs = occ::qm::AOBasis::load(p.b.wfn.atoms, "def2-svp");
        bs.set_pure(p.b.wfn.basis.is_pure());
        const auto &atoms = p.b.wfn.atoms;
        const auto &shells = p.b.wfn.basis.shells();
        const auto &name = p.b.wfn.basis.name();
        const auto &ecp_shells = bs.ecp_shells();
        p.b.wfn.basis = occ::qm::AOBasis(atoms, shells, name, ecp_shells);
        p.b.wfn.basis.set_ecp_electrons(bs.ecp_electrons());
    }
    occ::log::debug("Translation A: {}", p.b.translation.transpose());
    {
        std::ofstream out("b_mo.txt");
        out << p.b.wfn.mo.C;
    }
    {
        std::ofstream out("b_d.txt");
        out << p.b.wfn.mo.D;
    }

    p.a.wfn.apply_transformation(p.a.rotation, p.a.translation);
    p.b.wfn.apply_transformation(p.b.rotation, p.b.translation);
    fmt::print("Monomer A positions after rotation + translation:\n");
    for (const auto &a : p.a.wfn.atoms) {
        fmt::print("{} {:20.12f} {:20.12f} {:20.12f}\n",
                   occ::core::Element(a.atomic_number).symbol(),
                   a.x / occ::units::ANGSTROM_TO_BOHR,
                   a.y / occ::units::ANGSTROM_TO_BOHR,
                   a.z / occ::units::ANGSTROM_TO_BOHR);
    }
    fmt::print("Monomer B positions after rotation + translation:\n");
    for (const auto &a : p.b.wfn.atoms) {
        fmt::print("{} {:20.12f} {:20.12f} {:20.12f}\n",
                   occ::core::Element(a.atomic_number).symbol(),
                   a.x / occ::units::ANGSTROM_TO_BOHR,
                   a.y / occ::units::ANGSTROM_TO_BOHR,
                   a.z / occ::units::ANGSTROM_TO_BOHR);
    }

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
    CLI::App app("occ - A program for quantum chemistry");
    std::string input_file{""}, model_name{"ce-b3lyp"}, output_file{""},
        verbosity{"warn"};
    std::string monomer_directory(".");
    std::string dft_functional("");

    int threads{1};
    bool use_df{false};
    double xdm_a1{1.0};
    double xdm_a2{1.0};

    CLI::Option *input_option =
        app.add_option("input", input_file, "input file");
    input_option->required();
    app.add_option("output", output_file, "output file");
    app.add_option("-t,--threads", threads, "number of threads");
    app.add_option("-m,--model", model_name, "CE energy model");
    app.add_flag("-d,--with-density-fitting", use_df,
                 "Use density fitting (RI-JK)");
    app.add_flag("--dft-functional", dft_functional,
                 "Use density functional for exchange");
    app.add_option("-v,--verbosity", verbosity, "logging verbosity");
    app.add_option("--monomer-directory", monomer_directory,
                   "directory to find monomer wavefunctions");
    app.add_option("--xdm-a1", xdm_a1, "a1 parameter for XDM");
    app.add_option("--xdm-a2", xdm_a2, "a2 parameter for XDM");

    CLI11_PARSE(app, argc, argv);

    auto level = occ::log::level::warn;
    std::string level_lower = occ::util::to_lower_copy(verbosity);

    occ::log::setup_logging(level_lower);

    occ::timing::start(occ::timing::category::global);

    auto pair = parse_input_file(input_file, monomer_directory);

    occ::parallel::set_num_threads(threads);

    auto model = occ::interaction::ce_model_from_string(model_name);
    model.xdm_a1 = xdm_a1;
    model.xdm_a1 = xdm_a2;

    CEModelInteraction interaction(model);
    if (use_df) {
        interaction.set_use_density_fitting(true);
    }

    occ::interaction::CEEnergyComponents interaction_energy;
    if (dft_functional.empty()) {
        interaction_energy = interaction(pair.a.wfn, pair.b.wfn);
    } else {
        interaction_energy =
            interaction.dft_pair(dft_functional, pair.a.wfn, pair.b.wfn);
    }
    occ::timing::stop(occ::timing::category::global);

    fmt::print("Monomer A energies\n");
    pair.a.wfn.energy.print();
    save_xdm_parameters(pair.a.wfn, "a_xdm.json");

    fmt::print("Monomer B energies\n");
    pair.b.wfn.energy.print();
    save_xdm_parameters(pair.b.wfn, "b_xdm.json");

    fmt::print("\nDimer\n");

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

    occ::timing::print_timings();
    if (!output_file.empty()) {
        nlohmann::json j = {
            {"energies",
             {{"pair",
               {{"coulomb", interaction_energy.coulomb_kjmol()},
                {"exchange_repulsion", interaction_energy.exchange_kjmol()},
                {"polarization", interaction_energy.polarization_kjmol()},
                {"dispersion", interaction_energy.dispersion_kjmol()},
                {"scaled_total", interaction_energy.total_kjmol()},
                {"model", model_name},
                {"units", "kj/mol"}}},
              {"monomer_a",
               {{"coulomb", pair.a.wfn.energy.coulomb},
                {"exchange", pair.a.wfn.energy.exchange},
                {"nuclear_repulsion", pair.a.wfn.energy.nuclear_repulsion},
                {"nuclear_attraction", pair.a.wfn.energy.nuclear_repulsion},
                {"kinetic", pair.a.wfn.energy.kinetic},
                {"core", pair.a.wfn.energy.core},
                {"total", pair.a.wfn.energy.total}}},
              {"monomer_b",
               {{"coulomb", pair.b.wfn.energy.coulomb},
                {"exchange", pair.b.wfn.energy.exchange},
                {"nuclear_repulsion", pair.b.wfn.energy.nuclear_repulsion},
                {"nuclear_attraction", pair.b.wfn.energy.nuclear_repulsion},
                {"kinetic", pair.b.wfn.energy.kinetic},
                {"core", pair.b.wfn.energy.core},
                {"total", pair.b.wfn.energy.total}}}}}};
        std::ofstream output(output_file);
        output << std::setw(2) << j;
    }
}
