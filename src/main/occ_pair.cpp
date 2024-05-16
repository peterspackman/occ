#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Option.hpp>
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
#include <occ/io/occ_input.h>
#include <occ/io/orca_json.h>
#include <occ/io/wavefunction_json.h>
#include <occ/main/occ_pair.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace occ::interaction {

inline void to_json(nlohmann::json &j, const CEEnergyComponents &e) {
    j["coulomb"] = e.coulomb;
    j["exchange"] = e.exchange;
    j["repulsion"] = e.repulsion;
    j["polarization"] = e.polarization;
    j["dispersion"] = e.dispersion;
    j["total"] = e.total;
    j["exchange_repulsion"] = e.exchange_repulsion;
    j["orthogonal_term_extra"] = e.orthogonal_term;
    j["nonorthogonal_term_extra"] = e.nonorthogonal_term;
}

inline void to_json(nlohmann::json &j, const CEParameterizedModel &m) {
    auto &factors = j["scale_factors"];
    factors["coulomb"] = m.coulomb;
    factors["exchange"] = m.exchange;
    factors["repulsion"] = m.repulsion;
    factors["polarization"] = m.polarization;
    factors["dispersion"] = m.dispersion;
    j["name"] = m.name;
    j["method"] = m.method;
    j["basis"] = m.basis;
    j["xdm_dispersion"] = m.xdm;
    if(m.xdm) {
	j["xdm_a1"] = m.xdm_a1;
	j["xdm_a2"] = m.xdm_a2;
    }
}

}

namespace occ::main {
namespace fs = std::filesystem;

using occ::Mat3;
using occ::Vec3;
using occ::interaction::CEModelInteraction;
using occ::qm::Wavefunction;

void store_xdm_parameters(const Wavefunction &wfn, nlohmann::json &j) {
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
}

void write_json(const PairEnergy &pair, const std::string &filename) {
    std::ofstream out(filename);
    nlohmann::json j;
    auto &a_json = j["a"];
    auto &b_json = j["b"];
    auto &pair_energy = j["interaction_energy"];
    store_xdm_parameters(pair.a.wfn, a_json["xdm_parameters"]);
    store_xdm_parameters(pair.b.wfn, b_json["xdm_parameters"]);

    a_json["energy_components"] = pair.a.wfn.energy;
    b_json["energy_components"] = pair.b.wfn.energy;
    pair_energy = pair.energy;
    j["interaction_model"] = pair.model;
    out << j.dump(2);
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

void parse_input_file(occ::io::OccInput &input, const std::string &filename,
                      const std::string &monomer_directory) {
    std::ifstream i(filename);
    nlohmann::json j;
    i >> j;
    const auto &monomers = j["monomers"];

    if (monomers.size() != 2)
        throw std::runtime_error("Require two monomers in input file");

    input.pair.source_a =
        (fs::path(monomer_directory) / fs::path(monomers[0]["source"]))
            .string();
    input.pair.source_b =
        (fs::path(monomer_directory) / fs::path(monomers[1]["source"]))
            .string();

    load_matrix(monomers[0]["rotation"], input.pair.rotation_a);
    load_vector(monomers[0]["translation"], input.pair.translation_a);
    if (monomers[0].contains("ecp_electrons")) {
        monomers[0]["ecp_electrons"].get_to(input.pair.ecp_electrons_a);
    }

    load_matrix(monomers[1]["rotation"], input.pair.rotation_b);
    load_vector(monomers[1]["translation"], input.pair.translation_b);

    if (monomers[1].contains("ecp_electrons")) {
        monomers[1]["ecp_electrons"].get_to(input.pair.ecp_electrons_b);
    }
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

/*
    } else if (config.driver.driver == "crystal") {
        occ::log::info("Using crystal driver\n");
        occ::crystal::Crystal crystal(config.crystal.asymmetric_unit,
                                      config.crystal.space_group,
                                      config.crystal.unit_cell);
        nlohmann::json json_data = crystal;
        auto path = fs::path(config.filename);
        path.replace_extension(".cxc.json");
        occ::log::info("Writing crystal data to {}\n", path.string());
        std::ofstream of(path.string());
        of << std::setw(2) << json_data << std::endl;

    } else {
        throw std::runtime_error(
            fmt::format("Unknown driver: {}", config.driver.driver));
    }
    */

CLI::App *add_pair_subcommand(CLI::App &app) {

    CLI::App *pair = app.add_subcommand("pair", "compute pair energy");
    auto config = std::make_shared<OccPairInput>();

    pair->add_option("-m,--model", config->model_name, "Energy model");
    pair->add_option("-a,--monomer_a,--monomer-a", config->monomer_a,
                     "Monomer wavefunction source A")
        ->required();
    pair->add_option("-b,--monomer_b,--monomer-b", config->monomer_b,
                     "Monomer wavefunction source B")
        ->required();
    pair->add_option("--rotation-a,--rotation_a", config->rotation_a,
                     "Rotation for monomer A (row major order)")
        ->expected(9);
    pair->add_option("--rotation-b,--rotation_b", config->rotation_b,
                     "Rotation for monomer B (row major order)")
        ->expected(9);

    pair->add_option("--translation-a,--translation_a", config->translation_a,
                     "Translation for monomer A")
        ->expected(3);
    pair->add_option("--translation-b,--translation_b", config->translation_b,
                     "Translation for monomer B")
        ->expected(3);
    pair->add_option("--json", config->output_json_filename,
                     "JSON filename for output");

    pair->fallthrough();
    pair->callback([config]() { run_pair_subcommand(*config); });
    return pair;
}

void run_pair_subcommand(OccPairInput const &input) {
    occ::io::OccInput config;
    auto &pair = config.pair;
    pair.model_name = input.model_name;
    pair.source_a = input.monomer_a;
    pair.source_b = input.monomer_b;
    pair.rotation_a = Eigen::Map<const Mat3RM>(input.rotation_a.data());
    pair.rotation_b = Eigen::Map<const Mat3RM>(input.rotation_b.data());

    pair.translation_a = Eigen::Map<const Vec3>(input.translation_a.data());
    pair.translation_b = Eigen::Map<const Vec3>(input.translation_b.data());

    if (!input.input_json_filename.empty()) {
        parse_input_file(config, input.input_json_filename,
                         input.monomer_directory);
    }
    occ::log::debug("Rotation A (det = {}):\n{}",
                    config.pair.rotation_a.determinant(),
                    config.pair.rotation_a);
    occ::log::debug("Translation A: {}", config.pair.translation_a.transpose());

    occ::log::debug("Rotation B (det = {}):\n{}",
                    config.pair.rotation_b.determinant(),
                    config.pair.rotation_b);
    occ::log::debug("Translation B: {}", config.pair.translation_b.transpose());

    PairEnergy pair_energy(config);

    pair_energy.compute();


    if (!input.output_json_filename.empty()) {
	occ::log::debug("Writing JSON output to {}", input.output_json_filename);
        write_json(pair_energy, input.output_json_filename);
    }

}

} // namespace occ::main
