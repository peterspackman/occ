#include <occ/driver/vibrational_analysis.h>
#include <occ/driver/acceleration.h>
#include <occ/driver/method_parser.h>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/dft/dft.h>
#include <occ/qm/hf.h>
#include <occ/qm/hessians.h>
#include <occ/qm/scf.h>
#include <occ/solvent/solvation_correction.h>
#include <fstream>
#include <occ/io/core_json.h>
#include <occ/io/eigen_json.h>
#include <nlohmann/json.hpp>

using occ::core::Molecule;
using occ::core::VibrationalModes;
using occ::dft::DFT;
using occ::io::OccInput;
using occ::qm::HartreeFock;
using occ::qm::HessianEvaluator;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

namespace occ::driver {

/**
 * @brief Load basis set for vibrational analysis
 */
occ::gto::AOBasis load_basis_set_vib(const Molecule &m, const std::string &name,
                                    bool spherical) {
  auto basis = occ::gto::AOBasis::load(m.atoms(), name);
  basis.set_pure(spherical);
  occ::log::debug("Loaded basis set: {}", spherical ? "spherical" : "cartesian");
  occ::log::debug("Number of shells:            {}", basis.size());
  occ::log::debug("Number of  basis functions:  {}", basis.nbf());
  occ::log::debug("Maximum angular momentum:    {}", basis.l_max());
  return basis;
}

/// Run `hessian` on `proc`, or on `proc` in the configured solvent.
template <typename T, typename F>
Mat in_solvent(T &proc, const OccInput &config, F &&hessian) {
  if (config.solvent.solvent_name.empty())
    return hessian(proc);
  occ::solvent::SolvationCorrectedProcedure<T> solvated(
      proc, config.solvent.solvent_name, config.solvent.radii_scaling);
  return hessian(solvated);
}

/**
 * @brief Template function to compute Hessian for a given method
 */
template <typename T, SpinorbitalKind SK>
Mat compute_hessian_for_method(const Molecule &m, const occ::gto::AOBasis &basis,
                              const OccInput &config, const Wavefunction &wfn,
                              const VibrationalAnalysisConfig &vib_config) {

  auto make_proc = [&]() {
    if constexpr (std::is_same<T, DFT>::value)
      return T(config.method.name, basis, config.method.dft_grid);
    else
      return T(basis);
  };
  T proc = make_proc();

  occ::log::debug("Spinorbital kind: {}", spinorbital_kind_to_string(SK));
  occ::log::trace("Setting integral precision: {}", config.method.integral_precision);
  proc.set_precision(config.method.integral_precision);

  // Analytic (CPHF/CPKS) Hessian where supported.
  using HM = VibrationalAnalysisConfig::HessianMethod;
  if (vib_config.hessian_method != HM::FiniteDifferences) {
    // Every term must differentiate one energy, so the SCF is reconverged
    // (from the given wavefunction) with the integrals the Hessian uses: the
    // energy's own density fitting and COSX where the analytic Hessian
    // supports them, otherwise exact integrals. COSX converges on the grid
    // its gradient uses (GRIDX3), so the Hessian is that gradient's
    // derivative.
    T accel = make_proc();
    accel.set_precision(config.method.integral_precision);
    apply_acceleration(accel, basis.nbf(), config, AccelerationUse::Gradient);
    const std::string why_smeared =
        wfn.mo.smearing.kind != occ::qm::OrbitalSmearing::Kind::None
            ? "smeared (fractional) occupations are not supported"
            : "";
    const std::string why_accel = why_smeared.empty()
                                      ? accel.analytic_hessian_unsupported(SK)
                                      : why_smeared;
    const std::string why = why_accel.empty() || !why_smeared.empty()
                                ? why_smeared
                                : proc.analytic_hessian_unsupported(SK);
    T &hproc = why_accel.empty() ? accel : proc;
    if (why.empty()) {
      occ::log::info("Computing molecular Hessian...");
      occ::log::info("  Method: analytic (coupled-perturbed SCF)");
      if (!why_accel.empty() && (accel.using_density_fitting() ||
                                 accel.using_cosx()))
        occ::log::info("  Using exact integrals: {}", why_accel);
      return in_solvent(hproc, config, [&](auto &p) {
        using P = std::remove_reference_t<decltype(p)>;
        HessianEvaluator<P> hess_eval(p);
        hess_eval.set_method(HessianEvaluator<P>::Method::Analytical);
        SCF<P> scf(p, SK);
        scf.set_charge_multiplicity(wfn.charge(), wfn.multiplicity());
        scf.set_initial_guess_from_wfn(wfn);
        scf.convergence_settings = hess_eval.scf_convergence_settings();
        scf.compute_scf_energy();
        Mat hessian = hess_eval(scf.wavefunction());
        occ::log::info("Hessian computation completed");
        return hessian;
      });
    }
    if (vib_config.hessian_method == HM::Analytic)
      throw std::runtime_error("Analytic Hessian not available: " + why);
    occ::log::info("Analytic Hessian not available ({}); using finite "
                   "differences",
                   why);
  }

  // Density fitting (same policy as the single-point energy driver), but not
  // COSX: the finite differences are of gradients, and the COSX energy
  // schedule's gradient is not that energy's derivative.
  apply_acceleration(proc, basis.nbf(), config, AccelerationUse::NoCOSX);

  Mat hessian = in_solvent(proc, config, [&](auto &p) {
    HessianEvaluator<std::remove_reference_t<decltype(p)>> hess_eval(p);
    hess_eval.set_step_size(vib_config.step_size);
    hess_eval.set_use_acoustic_sum_rule(vib_config.use_acoustic_sum_rule);

    occ::log::info("Computing molecular Hessian...");
    occ::log::info("  Method: Finite differences");
    occ::log::info("  Step size: {:.3e} Bohr", hess_eval.step_size());
    occ::log::info("  Acoustic sum rule: {}",
                   hess_eval.use_acoustic_sum_rule() ? "enabled" : "disabled");
    return hess_eval(wfn);
  });
  
  occ::log::info("Hessian computation completed");
  occ::log::debug("Hessian matrix dimensions: {}x{}", hessian.rows(), hessian.cols());
  
  return hessian;
}

/**
 * @brief Dispatch Hessian computation based on method type
 */
Mat compute_hessian_driver(const OccInput &config, const Molecule &m,
                          const Wavefunction &wfn,
                          const VibrationalAnalysisConfig &vib_config) {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;

  if (!config.basis.basis_set_directory.empty()) {
    occ::log::debug("Overriding environment basis set directory with: '{}'",
                   config.basis.basis_set_directory);
    occ::set_data_directory(config.basis.basis_set_directory);
  }
  
  auto basis = load_basis_set_vib(m, config.basis.name, config.basis.spherical);
  auto method_kind = method_kind_from_string(config.method.name);
  auto guess_sk = determine_spinorbital_kind(
      config.method.name, config.electronic.multiplicity, method_kind);
  auto conf_sk = config.electronic.spinorbital_kind;

  if (!config.solvent.solvent_name.empty() && config.solvent.radii_scaling)
    throw std::runtime_error("Solvated Hessians are not available with "
                             "DRACO-scaled cavity radii");
  switch (method_kind) {
  case MethodKind::HF: {
    if (guess_sk == U || conf_sk == U)
      return compute_hessian_for_method<HartreeFock, U>(m, basis, config, wfn, vib_config);
    else if (guess_sk == G || conf_sk == G)
      return compute_hessian_for_method<HartreeFock, G>(m, basis, config, wfn, vib_config);
    else
      return compute_hessian_for_method<HartreeFock, R>(m, basis, config, wfn, vib_config);
    break;
  }
  case MethodKind::DFT: {
    if (guess_sk == U || conf_sk == U)
      return compute_hessian_for_method<DFT, U>(m, basis, config, wfn, vib_config);
    else if (guess_sk == G || conf_sk == G)
      throw std::runtime_error("Not implemented: DFT general spinorbital Hessians");
    else
      return compute_hessian_for_method<DFT, R>(m, basis, config, wfn, vib_config);
    break;
  }
  case MethodKind::MP2: {
    throw std::runtime_error("Not implemented: MP2 Hessians for vibrational analysis");
    break;
  }
  default: {
    throw std::runtime_error("Unknown method kind for Hessian computation");
  }
  }
}

VibrationalModes vibrational_analysis(const OccInput &config,
                                     const Wavefunction &wfn,
                                     const VibrationalAnalysisConfig &vib_config) {
  
  occ::log::info("{:=^72s}", "  Vibrational Frequency Analysis  ");
  
  // Use the optimized geometry from the wavefunction, not the input config
  Molecule m(wfn.atoms);
  
  occ::log::info("Molecule:");
  occ::log::info("  Number of atoms: {}", m.size());
  occ::log::info("  Total degrees of freedom: {}", 3 * m.size());
  
  if (!vib_config.compute_frequencies) {
    occ::log::info("Frequency computation disabled by configuration");
    return VibrationalModes{};
  }
  
  // Compute molecular Hessian
  Mat hessian = compute_hessian_driver(config, m, wfn, vib_config);
  
  // Save Hessian if requested
  if (vib_config.save_hessian) {
    occ::log::info("Saving Hessian matrix to {}", vib_config.hessian_filename);
    nlohmann::json hess_json;
    hess_json["hessian"] = hessian;
    hess_json["units"] = "Hartree/Bohr^2";
    hess_json["method"] = config.method.name;
    hess_json["basis"] = config.basis.name;
    
    std::ofstream hess_file(vib_config.hessian_filename);
    hess_file << hess_json.dump(4);
    hess_file.close();
  }
  
  // Perform vibrational analysis
  occ::log::info("Performing normal mode analysis...");
  VibrationalModes vib_modes = occ::core::compute_vibrational_modes(
      hessian, m, vib_config.project_tr_rot);
  
  // Print detailed results
  occ::log::info("\n{}", vib_modes.frequencies_string());
  
  // Check for imaginary frequencies
  auto sorted_freqs = vib_modes.get_all_frequencies();
  int n_imaginary = 0;
  for (int i = 0; i < sorted_freqs.size(); i++) {
    if (sorted_freqs[i] < -50.0) {  // Consider frequencies < -50 cm⁻¹ as truly imaginary
      n_imaginary++;
    }
  }
  
  if (n_imaginary > 0) {
    occ::log::warn("Found {} imaginary frequencies", n_imaginary);
    occ::log::warn("This may indicate the structure is not a minimum on the potential energy surface");
  } else {
    // Count near-zero frequencies (should be 6 for non-linear, 5 for linear molecules)
    int n_near_zero = 0;
    for (int i = 0; i < sorted_freqs.size(); i++) {
      if (std::abs(sorted_freqs[i]) < 50.0) {
        n_near_zero++;
      }
    }
    occ::log::info("Found {} low-frequency modes (< 50 cm⁻¹)", n_near_zero);
    if (n_near_zero == 6) {
      occ::log::info("This is consistent with a non-linear molecule minimum");
    } else if (n_near_zero == 5) {
      occ::log::info("This is consistent with a linear molecule minimum");
    } else {
      occ::log::warn("Unexpected number of low-frequency modes (expected 5-6)");
    }
  }
  
  // Save results if requested
  if (vib_config.save_results) {
    occ::log::info("Saving vibrational analysis results to {}", vib_config.results_filename);
    nlohmann::json results_json;
    occ::core::to_json(results_json, vib_modes);
    results_json["method"] = config.method.name;
    results_json["basis"] = config.basis.name;
    results_json["step_size_bohr"] = vib_config.step_size;
    results_json["acoustic_sum_rule"] = vib_config.use_acoustic_sum_rule;
    results_json["projected_tr_rot"] = vib_config.project_tr_rot;
    
    std::ofstream results_file(vib_config.results_filename);
    results_file << results_json.dump(4);
    results_file.close();
  }
  
  return vib_modes;
}

VibrationalModes vibrational_analysis(const OccInput &config,
                                     const Wavefunction &wfn) {
  // Use default configuration
  VibrationalAnalysisConfig default_config;
  return vibrational_analysis(config, wfn, default_config);
}

} // namespace occ::driver