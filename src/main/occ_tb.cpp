#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/core/vibration.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/io/cifparser.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/load_geometry.h>
#include <occ/io/occ_input.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_tb.h>
#include <occ/main/version.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/xtb_calculator.h>

namespace occ::main {

namespace {

namespace fs = std::filesystem;

// Save a converged GFN2 wavefunction next to the input geometry, one file
// per requested format. `<stem><suffix>.owf.<fmt>` (e.g. water.owf.json or
// water_opt.owf.json). Empty entries in `formats` are skipped; formats not
// understood by `Wavefunction::save` log a warning from save() itself.
void write_wavefunction(const occ::qm::Wavefunction &wfn,
                        const std::string &input_path,
                        const std::vector<std::string> &formats,
                        const std::string &suffix = "") {
  if (formats.empty())
    return;
  fs::path base(input_path);
  fs::path parent = base.parent_path();
  const std::string stem = base.stem().string();
  for (const auto &fmt_ext : formats) {
    if (fmt_ext.empty())
      continue;
    fs::path path =
        parent / fmt::format("{}{}.owf.{}", stem, suffix, fmt_ext);
    occ::qm::Wavefunction copy = wfn; // save() is non-const
    copy.save(path.string());          // save() logs the path on success
  }
}

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

  // Periodic GFN2 wavefunction: Γ-only central-cell snapshot, suitable
  // for cube / isosurface visualisation. `--lattice-energy` adds per-
  // monomer SCCs below but their wavefunctions are not persisted to
  // keep the output set predictable — run them as standalone `occ tb`
  // jobs if you need their files.
  write_wavefunction(calc.to_wavefunction(), cfg.filename, cfg.formats);

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

void run_frequencies(occ::xtb::XtbCalculator &calc, const TbConfig &cfg) {
  // Numerical Hessian via FD of the (multipole-on) analytical gradient,
  // then mass-weighted diagonalisation with optional t/r projection. Cost
  // ≈ 6N analytical-gradient evaluations.
  occ::log::info("");
  occ::log::info("{:-<72s}", "Vibrational analysis ");
  occ::log::info("Hessian step    : {:.4f} Bohr", cfg.freq_step_bohr);
  occ::log::info("Project T/R     : {}",
                  cfg.freq_project_tr_rot ? "yes (ORCA-style)" : "no");
  auto modes = calc.compute_vibrational_modes(cfg.freq_step_bohr,
                                               cfg.freq_project_tr_rot);
  occ::log::info("");
  // Sorted ascending — translations + rotations come first (≈0 with t/r
  // projection on, otherwise some imaginary "soft" modes).
  auto sorted = modes.get_all_frequencies();
  occ::log::info("{:>6s} {:>14s} {:>14s}", "Mode", "Freq (cm⁻¹)",
                  "Freq (meV)");
  occ::log::info("{:-<38s}", "");
  constexpr double cm_to_meV = 0.1239841974;
  for (Eigen::Index i = 0; i < sorted.size(); ++i) {
    occ::log::info("{:6d} {:14.2f} {:14.2f}", i + 1, sorted(i),
                    sorted(i) * cm_to_meV);
  }
  // Pull out the (likely) vibrational modes — the largest 3N-6 (or 3N-5
  // for linear). Reported separately for convenience.
  const auto N = static_cast<int>(modes.n_atoms());
  const int n_vib = std::max(0, 3 * N - 6);
  if (n_vib > 0) {
    occ::log::info("");
    occ::log::info("Vibrational modes (top {}):", n_vib);
    for (Eigen::Index i = sorted.size() - n_vib; i < sorted.size(); ++i) {
      occ::log::info("  {:14.2f} cm⁻¹", sorted(i));
    }
  }
}

void run_molecular(const TbConfig &cfg) {
  auto mol = occ::io::load_molecule(cfg.filename);

  if (cfg.optimize) {
    // Build a minimal OccInput and route through the existing
    // `geometry_optimization` driver. The driver's MethodKind::GFN2 branch
    // calls `XtbCalculator::compute_energy_and_gradient(numerical=false)`,
    // which is the analytical multipole-on gradient validated in
    // tests/xtb_native_tests.cpp.
    occ::io::OccInput input;
    input.method.name = "gfn2";
    input.geometry.set_molecule(mol);
    input.electronic.charge = cfg.charge;
    input.filename = cfg.filename;
    occ::log::info("{:-<72s}", "GFN2-xTB molecular optimization ");
    occ::log::info("input          : {}", cfg.filename);
    occ::log::info("atoms          : {}", mol.size());
    occ::log::info("multipoles     : on (analytical gradient)");
    occ::log::info("charge         : {:+.3f} e", cfg.charge);
    occ::log::info("");
    auto wfn = occ::driver::geometry_optimization(input);
    // Save the converged wavefunction at the optimised geometry. The
    // driver already returned it from `to_wavefunction()`, so no extra
    // SCC is needed.
    write_wavefunction(wfn, cfg.filename, cfg.formats, "_opt");
    if (cfg.frequencies) {
      // Vibrational analysis on the optimized geometry. The optimizer
      // returns a Wavefunction with the converged atoms baked in; rebuild
      // an XtbCalculator from those atoms.
      occ::core::Molecule opt_mol(wfn.atoms);
      occ::xtb::XtbCalculator opt_calc(opt_mol);
      opt_calc.set_charge(cfg.charge);
      opt_calc.set_include_multipoles(cfg.include_multipoles);
      run_frequencies(opt_calc, cfg);
    }
    return;
  }

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

  write_wavefunction(calc.to_wavefunction(), cfg.filename, cfg.formats);

  if (cfg.frequencies) {
    run_frequencies(calc, cfg);
  }
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
  tb->add_flag("--opt", cfg->optimize,
                "Geometry optimization (Berny / internal coords). "
                "Molecular only. Writes <input>_opt.xyz on convergence and "
                "<input>_trj.xyz with the trajectory.");
  tb->add_flag("--freq,--frequencies", cfg->frequencies,
                "After SCC, compute the numerical Hessian (FD of the "
                "multipole-on analytical gradient, ~6N gradient calls) and "
                "report vibrational frequencies. Molecular only.");
  tb->add_option("--freq-step", cfg->freq_step_bohr,
                  "Hessian FD step size in Bohr (default 0.005).");
  tb->add_flag("--no-project-tr-rot{false}", cfg->freq_project_tr_rot,
                "Disable translation/rotation projection of the mass-"
                "weighted Hessian (default on).");
  tb->add_option(
      "-o,--output", cfg->formats,
      "Wavefunction output formats (json, fchk; default json). Writes "
      "`<input>.owf.<fmt>` (and `<input>_opt.owf.<fmt>` with --opt). "
      "Periodic inputs save a Γ-only central-cell snapshot. Pass an "
      "empty value to disable.");

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
