#include <filesystem>
#include <occ/core/log.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/io/occ_input.h>
#include <occ/io/wavefunction_json.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/single_point.h>

namespace fs = std::filesystem;

namespace occ::main {

using occ::core::Element;
using occ::core::Molecule;
using occ::qm::Wavefunction;

void compute_monomer_energies(const std::string &basename,
                              WavefunctionList &wavefunctions,
                              const std::string &model_name) {
    size_t idx = 0;

    auto model = occ::interaction::ce_model_from_string(model_name);
    occ::interaction::CEModelInteraction interaction(model);
    for (auto &wfn : wavefunctions) {
        fs::path monomer_energies_path(
            fmt::format("{}_{}_monomer_energies.json", basename, idx));
        if (fs::exists(monomer_energies_path)) {
            occ::log::info("Loading monomer {} energies from {}", idx,
                           monomer_energies_path.string());
            std::ifstream ifs(monomer_energies_path.string());
            wfn.energy = nlohmann::json::parse(ifs).get<occ::qm::Energy>();
        } else {
            occ::log::info("Computing monomer {} energies", idx);
            interaction.compute_monomer_energies(wfn);
            occ::log::info("Writing monomer energies to {}",
                           monomer_energies_path.string());
            std::ofstream ofs(monomer_energies_path.string());
            nlohmann::json j = wfn.energy;
            ofs << j;
        }
        idx++;
    }
}

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name,
                                    const std::string &energy_model,
				    bool spherical) {
    fs::path json_path(fmt::format("{}.owf.json", name));
    if (fs::exists(json_path)) {
        occ::log::info("Loading gas phase wavefunction from {}",
                       json_path.string());
        using occ::io::JsonWavefunctionReader;
        JsonWavefunctionReader json_wfn_reader(json_path.string());
        return json_wfn_reader.wavefunction();
    }

    auto parameterized_model =
        occ::interaction::ce_model_from_string(energy_model);

    occ::io::OccInput input;
    input.method.name = parameterized_model.method;
    input.basis.name = parameterized_model.basis;
    input.basis.spherical = spherical;
    input.geometry.set_molecule(mol);
    input.electronic.charge = mol.charge();
    input.electronic.multiplicity = mol.multiplicity();
    auto wfn = occ::main::single_point_calculation(input);

    occ::io::JsonWavefunctionWriter writer;
    writer.write(wfn, json_path.string());
    return wfn;
}

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const std::vector<Molecule> &molecules,
                                         const std::string &energy_model,
					 bool spherical) {
    WavefunctionList wavefunctions;
    size_t index = 0;
    for (const auto &m : molecules) {
        occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}", index,
                       "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                           Element(atom.atomic_number).symbol(), atom.x, atom.y,
                           atom.z);
        }
        std::string name = fmt::format("{}_{}", basename, index);
        wavefunctions.emplace_back(
            calculate_wavefunction(m, name, energy_model, spherical));
        index++;
    }
    return wavefunctions;
}
} // namespace occ::main
