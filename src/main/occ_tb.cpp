#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/cifparser.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/load_geometry.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_tb.h>
#include <occ/main/version.h>
#include <occ/xtb/native_calculator.h>

namespace occ::main {

namespace {

// Decide whether `filename` resolves to a Crystal or a Molecule.
// CIF → always crystal. .gen → either (parser detects). .xyz → molecular.
enum class Kind { Molecule, Crystal };

Kind classify(const std::string &filename) {
  if (occ::io::CifParser::is_likely_cif_filename(filename))
    return Kind::Crystal;
  if (occ::io::DftbGenFormat::is_likely_gen_filename(filename)) {
    // DftbGenFormat::is_periodic() is declared but undefined; probe by
    // attempting to extract a crystal — gen-as-cluster yields nullopt.
    occ::io::DftbGenFormat g;
    g.parse(filename);
    return g.crystal().has_value() ? Kind::Crystal : Kind::Molecule;
  }
  if (occ::io::XyzFileReader::is_likely_xyz_filename(filename))
    return Kind::Molecule;
  throw std::runtime_error(fmt::format(
      "occ tb: unsupported input '{}' (expect .xyz / .cif / .gen)",
      filename));
}

void run_periodic(const TbConfig &cfg) {
  auto crystal = occ::io::load_crystal(cfg.filename);
  occ::xtb::NativeCalculator calc(crystal);
  calc.set_charge(cfg.charge);
  calc.set_include_multipoles(cfg.include_multipoles);
  calc.set_multipole_ewald(cfg.multipole_ewald);
  if (cfg.kpoints[0] != 1 || cfg.kpoints[1] != 1 || cfg.kpoints[2] != 1) {
    calc.set_kpoints(cfg.kpoints[0], cfg.kpoints[1], cfg.kpoints[2]);
  }
  occ::log::info("{:-<72s}", "GFN2-xTB periodic ");
  occ::log::info("input          : {}", cfg.filename);
  occ::log::info("atoms          : {}", calc.num_atoms());
  occ::log::info("k-points       : {} × {} × {}", cfg.kpoints[0],
                  cfg.kpoints[1], cfg.kpoints[2]);
  occ::log::info("multipoles     : {}", cfg.include_multipoles ? "on" : "off");
  occ::log::info("dispersion     : {}", cfg.include_dispersion ? "on" : "off");
  occ::log::info("charge         : {:+.3f} e", cfg.charge);

  const double e_total = calc.single_point_energy();
  if (!calc.last_result().converged) {
    occ::log::error("SCC did not converge.");
    return;
  }
  occ::log::info("");
  calc.print_summary();
  occ::log::info("");
  occ::log::info("Energy decomposition  ----------------");
  occ::log::info("  SCC                : {:>20.12f} Ha", calc.scc_energy());
  occ::log::info("  Repulsion          : {:>20.12f} Ha",
                  calc.repulsion_energy());
  occ::log::info("  Dispersion         : {:>20.12f} Ha",
                  calc.dispersion_energy());
  occ::log::info("  Total              : {:>20.12f} Ha", e_total);
  if (cfg.print_charges) {
    occ::log::info("");
    occ::log::info("Atomic charges (Mulliken)  -----------");
    const auto q = calc.charges();
    for (int i = 0; i < q.size(); ++i) {
      occ::log::info("  atom {:>4d}  : {:+.6f}", i + 1, q(i));
    }
  }
}

void run_molecular(const TbConfig &cfg) {
  auto mol = occ::io::load_molecule(cfg.filename);
  occ::xtb::NativeCalculator calc(mol);
  calc.set_charge(cfg.charge);
  calc.set_include_multipoles(cfg.include_multipoles);
  occ::log::info("{:-<72s}", "GFN2-xTB molecular ");
  occ::log::info("input          : {}", cfg.filename);
  occ::log::info("atoms          : {}", calc.num_atoms());
  occ::log::info("multipoles     : {}", cfg.include_multipoles ? "on" : "off");
  occ::log::info("charge         : {:+.3f} e", cfg.charge);

  const double e_total = calc.single_point_energy();
  if (!calc.last_result().converged) {
    occ::log::error("SCC did not converge.");
    return;
  }
  occ::log::info("");
  calc.print_summary();
  occ::log::info("");
  occ::log::info("Total energy        : {:>20.12f} Ha", e_total);
}

} // namespace

CLI::App *add_tb_subcommand(CLI::App &app) {
  auto cfg = std::make_shared<TbConfig>();
  CLI::App *tb = app.add_subcommand(
      "tb", "Run a tight-binding (GFN2-xTB) single point on .xyz / .cif / .gen");
  tb->fallthrough();

  auto *input_opt = tb->add_option("input", cfg->filename,
                                     "Geometry file (.xyz / .cif / .gen)");
  input_opt->required()->check(CLI::ExistingFile);
  tb->add_option("-c,--charge", cfg->charge, "Net charge (e)");
  tb->add_flag("--no-multipoles{false}", cfg->include_multipoles,
                "Disable CAMM multipoles + anisotropic ES (charge-only SCC)");
  tb->add_flag("--no-multipole-ewald{false}", cfg->multipole_ewald,
                "Use real-space-only multipole pair sum (no Ewald split)");
  tb->add_flag("--no-dispersion{false}", cfg->include_dispersion,
                "Disable D4 dispersion");
  tb->add_option("-k,--kpoints", cfg->kpoints, "k-mesh (n1 n2 n3, default Γ-only)")
      ->expected(3);
  tb->add_flag("--no-charges{false}", cfg->print_charges,
                "Suppress per-atom charge listing");

  tb->callback([cfg]() { run_tb_subcommand(*cfg); });
  return tb;
}

void run_tb_subcommand(const TbConfig &config) {
  occ::main::print_header();
  occ::log::info("");
  const Kind kind = classify(config.filename);
  if (kind == Kind::Crystal) {
    run_periodic(config);
  } else {
    run_molecular(config);
  }
}

} // namespace occ::main
