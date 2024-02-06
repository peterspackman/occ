#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/io/cifparser.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/main/occ_dimers.h>

namespace fs = std::filesystem;
using occ::crystal::Crystal;

inline Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

namespace occ::main {

CLI::App *add_dimers_subcommand(CLI::App &app) {

    CLI::App *dimers =
        app.add_subcommand("dimers", "compute dimers in crystal");
    auto config = std::make_shared<DimerGenerationSettings>();

    dimers
        ->add_option("crystal", config->crystal_filename,
                     "input crystal structure (CIF)")
        ->required();
    dimers->add_option("--json", config->output_json_filename,
                       "JSON filename for output");
    dimers->add_option("-r,--radius", config->max_radius,
                       "maximum radius (Angstroms) for neighbours");
    dimers->add_flag("--xyz", config->generate_xyz_files, "Generate xyz files");
    dimers->fallthrough();
    dimers->callback([config]() { run_dimers_subcommand(*config); });
    return dimers;
}

void run_dimers_subcommand(const DimerGenerationSettings &settings) {
    std::string filename = settings.crystal_filename;
    std::string basename = fs::path(filename).stem().string();

    Crystal c = read_crystal(filename);
    occ::log::info("Loaded crystal from {}", filename);
    auto molecules = c.symmetry_unique_molecules();
    auto uc_molecules = c.unit_cell_molecules();
    nlohmann::json json_obj;
    json_obj["symmetry unique molecules"] = molecules;
    json_obj["unit cell molecules"] = uc_molecules;
    occ::log::info("Symmetry unique molecules in {}: {}", filename,
                   molecules.size());

    occ::log::info("Calculating symmetry unique dimers");
    auto dimers = c.symmetry_unique_dimers(settings.max_radius);

    json_obj["symmetry unique dimers"] = dimers;

    occ::log::info("Writing dimers to '{}'", settings.output_json_filename);
    std::ofstream output(settings.output_json_filename);
    output << json_obj.dump();
}

} // namespace occ::main
