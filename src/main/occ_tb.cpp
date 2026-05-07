#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/io/cifparser.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/load_geometry.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_tb.h>
#include <occ/main/version.h>
#include <occ/xtb/xtb_calculator.h>

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
  occ::xtb::XtbCalculator calc(crystal);
  calc.set_charge(cfg.charge);
  calc.set_include_multipoles(cfg.include_multipoles);
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

  if (cfg.lattice_energy) {
    // Lattice energy = E_crystal − Σ_i E_mol_i over the unit cell, where
    // each unit-cell molecule's energy is taken from its symmetry-unique
    // representative. Reported per molecule in kJ/mol.
    const auto &uc_mols = crystal.unit_cell_molecules();
    const auto &uniq_mols = crystal.symmetry_unique_molecules();
    const int n_uniq = static_cast<int>(uniq_mols.size());
    const int n_uc = static_cast<int>(uc_mols.size());
    occ::log::info("");
    occ::log::info("{:-<72s}", "Lattice energy via molecular GFN2 ");
    occ::log::info("unit-cell molecules : {}", n_uc);
    occ::log::info("symmetry-unique     : {}", n_uniq);

    std::vector<double> e_unique(n_uniq, 0.0);
    for (int i = 0; i < n_uniq; ++i) {
      occ::xtb::XtbCalculator mol_calc(uniq_mols[i]);
      mol_calc.set_charge(cfg.charge);
      mol_calc.set_include_multipoles(cfg.include_multipoles);
      const double e_mol = mol_calc.single_point_energy();
      if (!mol_calc.last_result().converged) {
        occ::log::error("Molecular SCC for unique mol {} did not converge.", i);
        return;
      }
      e_unique[i] = e_mol;
      occ::log::info("  unique mol {:>2d}  ({:>3d} atoms)  E = {:>20.12f} Ha",
                      i, mol_calc.num_atoms(), e_mol);
    }

    double e_mol_sum = 0.0;
    for (const auto &m : uc_mols) {
      const int idx = m.asymmetric_molecule_idx();
      e_mol_sum += e_unique[idx];
    }
    const double e_lat_cell = e_total - e_mol_sum;
    const double e_lat_per_mol_ha = e_lat_cell / std::max(n_uc, 1);
    const double e_lat_per_mol_kjmol =
        e_lat_per_mol_ha * occ::units::AU_TO_KJ_PER_MOL;
    occ::log::info("  Σ molecular        = {:>20.12f} Ha", e_mol_sum);
    occ::log::info("  E_lattice / cell   = {:>20.12f} Ha", e_lat_cell);
    occ::log::info("  E_lattice / mol    = {:>20.12f} Ha   ({:>+8.2f} kJ/mol)",
                    e_lat_per_mol_ha, e_lat_per_mol_kjmol);
  }
}

void run_molecular(const TbConfig &cfg) {
  auto mol = occ::io::load_molecule(cfg.filename);
  occ::xtb::XtbCalculator calc(mol);
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
  tb->add_flag("--no-dispersion{false}", cfg->include_dispersion,
                "Disable D4 dispersion");
  tb->add_option("-k,--kpoints", cfg->kpoints, "k-mesh (n1 n2 n3, default Γ-only)")
      ->expected(3);
  tb->add_flag("-L,--lattice-energy", cfg->lattice_energy,
                "After periodic SCC, compute molecular SCC for each "
                "symmetry-unique molecule and report lattice energy per "
                "molecule (kJ/mol). Crystal input only.");

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
