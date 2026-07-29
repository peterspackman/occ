#include <fmt/core.h>
#include <occ/core/constants.h>
#include <occ/core/data_directory.h>
#include <occ/dft/dft.h>
#include <occ/disp/d4.h>
#include <occ/driver/acceleration.h>
#include <occ/driver/correlation.h>
#include <occ/driver/method_parser.h>
#include <occ/xtb/xtb_calculator.h>
#include <occ/driver/single_point.h>
#include <occ/io/occ_input.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/thc_mp2.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/gradients.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/xdm/xdm.h>

namespace occ::driver {

using occ::core::Element;
using occ::core::Molecule;
using occ::dft::DFT;
using occ::io::OccInput;
using occ::qm::HartreeFock;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

void print_matrix_xyz(const Mat &m) {
  for (size_t i = 0; i < 3; i++) {
    log::info("{: 12.6f} {: 12.6f} {: 12.6f}", m(i, 0), m(i, 1), m(i, 2));
  }
}

void print_vector(const Vec3 &m) {
  log::info("{: 12.6f} {: 12.6f} {: 12.6f}", m(0), m(1), m(2));
}

occ::gto::AOBasis load_basis_set(const Molecule &m, const std::string &name,
                                bool spherical) {
  auto basis = occ::gto::AOBasis::load(m.atoms(), name);
  basis.set_pure(spherical);
  log::info("Loaded basis set: {}", spherical ? "spherical" : "cartesian");
  log::info("Number of shells:            {}", basis.size());
  log::info("Number of  basis functions:  {}", basis.nbf());
  log::info("Maximum angular momentum:    {}", basis.l_max());
  return basis;
}

void print_configuration(const Molecule &m, const OccInput &config) {
  log::info("{:=^72s}", "  Input  ");

  log::info("{: <20s} {: >20s}", "Method string", config.method.name);
  log::info("{: <20s} {: >20s}", "Basis name", config.basis.name);
  log::info("{: <20s} {: >20s}", "Shell kind",
            config.basis.spherical ? "spherical" : "Cartesian");
  log::info("{: <20s} {: >20d}", "Net charge",
            static_cast<int>(config.electronic.charge));
  log::info("{: <20s} {: >20d}", "Multiplicity",
            config.electronic.multiplicity);

  if (config.method.orbital_smearing_sigma != 0.0) {
    log::info("{: <20s} {: >12.5f}", "Orbital smearing sigma",
              config.method.orbital_smearing_sigma);
  }

  log::info("{:-<72s}", fmt::format("Geometry '{}' (au)  ", config.filename));
  for (const auto &atom : m.atoms()) {
    log::info("{:^3s} {:12.6f} {:12.6f} {:12.6f}",
              Element(atom.atomic_number).symbol(), atom.x, atom.y, atom.z);
  }

  double temperature = occ::constants::celsius<double> + 25;

  log::info("{:-<72s}", "Inertia tensor (x 10e-46 kg m^2)  ");
  print_matrix_xyz(m.inertia_tensor());
  log::info("{:-<72s}", "Principal moments of inertia  ");
  print_vector(m.principal_moments_of_inertia());
  log::info("{:-<72s}", "Rotational constants (GHz)  ");
  print_vector(m.rotational_constants());
  log::info("\n");

  log::info("{:-<72s}",
            fmt::format("Gas-phase properties (at {} K)  ", temperature));
  log::info("Rotational free energy      {: 12.6f} kJ/mol",
            m.rotational_free_energy(temperature));
  log::info("Translational free energy   {: 12.6f} kJ/mol",
            m.translational_free_energy(temperature));
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_method(Molecule &m, const occ::gto::AOBasis &basis,
                        const OccInput &config) {

  // Parse method name to extract dispersion correction
  auto method_spec = parse_method_string(config.method.name);

  T proc = [&]() {
    if constexpr (std::is_same<T, DFT>::value)
      return T(method_spec.base_method, basis, config.method.dft_grid);
    else
      return T(basis);
  }();

  apply_acceleration(proc, basis.nbf(), config);

  if constexpr (std::is_same<T, DFT>::value) {
    proc.set_xc_screening_threshold(config.method.dft_xc_screening_threshold);
  }

  occ::log::info("Spinorbital kind: {}", spinorbital_kind_to_string(SK));

  occ::log::trace("Setting integral precision: {}",
                  config.method.integral_precision);
  proc.set_precision(config.method.integral_precision);

  SCF<T> scf(proc, SK);
  scf.maxiter = config.method.scf_maxiter;
  occ::log::trace("Setting system charge: {}", config.electronic.charge);
  occ::log::trace("Setting system multiplicity: {}",
                  config.electronic.multiplicity);
  scf.set_charge_multiplicity(config.electronic.charge,
                              config.electronic.multiplicity);
  if (!config.geometry.point_charges.empty()) {
    scf.set_external_potential(
        occ::qm::PointChargePotential{config.geometry.point_charges});
  }

  if (config.method.orbital_smearing_sigma != 0.0) {
    scf.ctx.mo.smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;
    scf.ctx.mo.smearing.sigma = config.method.orbital_smearing_sigma;
  }

  double e = scf.compute_scf_energy();
  if constexpr (std::is_same<T, DFT>::value) {
    double enlc = proc.post_scf_nlc_correction(scf.ctx.mo);
    if (enlc != 0.0) {
      log::info("Post SCF NLC correction:         {: 20.12f}", enlc);
      e += enlc;
      log::info("Corrected total energy:          {: 20.12f}", e);
    }
  }

  // Add dispersion correction if specified via method string or --xdm flag
  bool use_xdm = (method_spec.dispersion == "xdm") || config.dispersion.evaluate_correction;
  bool use_d4 = (method_spec.dispersion == "d4");

  if (use_d4 || use_xdm) {
    if (use_d4) {
      occ::disp::D4Dispersion disp(m.atoms(), occ::disp::RefqMode::DFT);
      try {
        disp.set_functional(method_spec.base_method);
      } catch (const std::exception &ex) {
        log::warn("D4 parameters not found for functional '{}' ({}), "
                  "using default PBE parameters",
                  method_spec.base_method, ex.what());
        disp.set_functional("pbe");
      }
      disp.set_charges_eeq(static_cast<double>(config.electronic.charge));
      double e_disp = disp.energy();
      log::info("D4 dispersion correction:        {: 20.12f}", e_disp);
      e += e_disp;
      log::info("Dispersion-corrected energy:     {: 20.12f}", e);
    } else if (use_xdm) {
      auto wfn = scf.wavefunction();

      // Check if user specified custom XDM parameters via flags
      std::optional<xdm::XDM::Parameters> xdm_params;
      if (config.dispersion.xdm_a1 != 1.0 || config.dispersion.xdm_a2 != 1.0) {
        xdm_params = xdm::XDM::Parameters{config.dispersion.xdm_a1, config.dispersion.xdm_a2};
      }

      auto [e_xdm, grad_xdm] = xdm::xdm_dispersion_gradient(
          wfn.basis, wfn.mo, config.electronic.charge, method_spec.base_method, xdm_params);

      log::info("XDM dispersion correction:       {: 20.12f}", e_xdm);
      e += e_xdm;
      log::info("Dispersion-corrected energy:     {: 20.12f}", e);
    } else {
      log::warn("Unsupported dispersion type '{}' - ignoring", method_spec.dispersion);
    }
  }

  if (config.method.orbital_smearing_sigma != 0.0) {
    log::info("Correlation entropy approx.      {: 20.12f}",
              scf.ctx.mo.smearing.ec_entropy());
    log::info("Free energy                      {: 20.12f}",
              e + scf.ctx.mo.smearing.ec_entropy());
    log::info("Energy (zero point)              {: 20.12f}",
              e + 0.5 * scf.ctx.mo.smearing.ec_entropy());
  }

  return scf.wavefunction();
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_solvated_method(const Wavefunction &wfn,
                                 const OccInput &config) {
  using occ::solvent::SolvationCorrectedProcedure;

  if constexpr (std::is_same<T, DFT>::value) {
    DFT ks(config.method.name, wfn.basis, config.method.dft_grid);
    ks.set_xc_screening_threshold(config.method.dft_xc_screening_threshold);
    apply_acceleration(ks, wfn.basis.nbf(), config);
    ks.set_system_charge(config.electronic.charge);
    SolvationCorrectedProcedure<DFT> proc_solv(ks, config.solvent.solvent_name,
                                               config.solvent.radii_scaling);
    SCF<SolvationCorrectedProcedure<DFT>> scf(proc_solv, SK);
    scf.maxiter = config.method.scf_maxiter;
    scf.set_charge_multiplicity(config.electronic.charge,
                                config.electronic.multiplicity);
    scf.set_initial_guess_from_wfn(wfn);
    double e = scf.compute_scf_energy();
    if (!config.solvent.output_surface_filename.empty())
      proc_solv.write_surface_file(config.solvent.output_surface_filename);
    return scf.wavefunction();
  } else {
    T proc(wfn.basis);
    proc.set_system_charge(config.electronic.charge);
    apply_acceleration(proc, wfn.basis.nbf(), config);
    SolvationCorrectedProcedure<T> proc_solv(proc, config.solvent.solvent_name,
                                             config.solvent.radii_scaling);
    SCF<SolvationCorrectedProcedure<T>> scf(proc_solv, SK);
    scf.maxiter = config.method.scf_maxiter;
    scf.set_charge_multiplicity(config.electronic.charge,
                                config.electronic.multiplicity);
    scf.set_initial_guess_from_wfn(wfn);
    double e = scf.compute_scf_energy();
    if (!config.solvent.output_surface_filename.empty())
      proc_solv.write_surface_file(config.solvent.output_surface_filename);
    return scf.wavefunction();
  }
}

Wavefunction run_mp2_method(const Wavefunction &scf_wfn,
                            const OccInput &config) {
  CorrelationOptions opts;
  opts.method = config.method.name;
  opts.backend = config.method.mp2_backend;
  opts.aux_basis = config.basis.ri_basis;
  opts.spin_scaling = config.method.mp2_spin_scaling;
  opts.max_memory_gb = config.method.mp2_max_memory_gb;
  opts.thc_c_isdf = config.method.mp2_thc_c_isdf;
  opts.thc_isdf_method = config.method.mp2_thc_method;
  opts.laplace_points = config.method.mp2_laplace_points;

  const auto result = run_correlation(scf_wfn, opts);

  Wavefunction mp2_wfn = scf_wfn;
  mp2_wfn.energy.total = result.total_energy;
  mp2_wfn.method = result.method;
  return mp2_wfn;
}

Wavefunction run_ccsd_method(const Wavefunction &scf_wfn,
                             const OccInput &config, bool with_triples) {
  CorrelationOptions opts;
  opts.method = config.method.name;
  // "exact" is the flag default and is indistinguishable from an explicit
  // --ccsd-backend exact; treat it as unset so a method-name prefix
  // ("thc-ccsd(t)") can select the backend, matching the previous behaviour.
  const auto backend = occ::util::to_lower_copy(config.method.ccsd_backend);
  opts.backend = (backend.empty() || backend == "exact") ? "auto" : backend;
  opts.aux_basis = config.basis.ri_basis;
  opts.n_frozen = config.method.ccsd_frozen_core;
  opts.max_memory_gb = config.method.ccsd_max_memory_gb;
  opts.thc_c_isdf = config.method.ccsd_thc_c_isdf;
  opts.thc_isdf_method = config.method.ccsd_thc_method;
  opts.thc_grid_angular = config.method.ccsd_thc_grid_angular;
  opts.thc_grid_radial_precision = config.method.ccsd_thc_grid_radial;

  auto method_opts = opts;
  if (!with_triples && parse_method_string(opts.method).kind ==
                           MethodKind::CCSD_T)
    method_opts.method = "ccsd";

  const auto result = run_correlation(scf_wfn, method_opts);

  Wavefunction cc_wfn = scf_wfn;
  cc_wfn.energy.total = result.total_energy;
  cc_wfn.method = result.method;
  return cc_wfn;
}


Wavefunction
single_point_driver(const OccInput &config,
                    const std::optional<Wavefunction> &guess = {}) {
  Molecule m = config.geometry.molecule();
  print_configuration(m, config);
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;

  if (!config.basis.basis_set_directory.empty()) {
    occ::log::info("Overriding environment basis set directory with: '{}'",
                   config.basis.basis_set_directory);
    occ::set_data_directory(config.basis.basis_set_directory);
  }

  auto method_kind = method_kind_from_string(config.method.name);

  // Methods with their own internal basis (GFN2-xTB) skip the AO basis load.
  if (method_kind == MethodKind::GFN2) {
    if (!config.solvent.solvent_name.empty()) {
      throw std::runtime_error("GFN2-xTB solvation is not yet wired into the "
                               "native backend; build with WITH_TBLITE=ON to "
                               "use the tblite path with solvation.");
    }
    occ::xtb::XtbCalculator calc(m);
    if (config.electronic.charge != 0.0)
      calc.set_charge(config.electronic.charge);
    // multiplicity > 1 selects the spin-unrestricted (spin-polarized) SCC.
    calc.set_num_unpaired_electrons(config.electronic.multiplicity - 1);
    calc.set_spin_polarization(config.electronic.spin_polarization);
    calc.set_temperature(config.electronic.electronic_temperature);
    (void)calc.single_point_energy();
    calc.print_summary();
    return calc.to_wavefunction();
  }

  auto basis = load_basis_set(m, config.basis.name, config.basis.spherical);
  auto guess_sk = determine_spinorbital_kind(
      config.method.name, config.electronic.multiplicity, method_kind);
  auto conf_sk = config.electronic.spinorbital_kind;

  if (config.solvent.solvent_name.empty()) {
    switch (method_kind) {
    case MethodKind::HF: {
      if (guess_sk == U || conf_sk == U)
        return run_method<HartreeFock, U>(m, basis, config);
      else if (guess_sk == G || conf_sk == G)
        return run_method<HartreeFock, G>(m, basis, config);
      else
        return run_method<HartreeFock, R>(m, basis, config);
      break;
    }
    case MethodKind::DFT: {
      if (guess_sk == U || conf_sk == U)
        return run_method<DFT, U>(m, basis, config);
      else
        return run_method<DFT, R>(m, basis, config);
      break;
    }
    case MethodKind::MP2: {
      // MP2 requires SCF first
      Wavefunction scf_wfn;
      if (guess_sk == U || conf_sk == U)
        scf_wfn = run_method<HartreeFock, U>(m, basis, config);
      else if (guess_sk == G || conf_sk == G)
        scf_wfn = run_method<HartreeFock, G>(m, basis, config);
      else
        scf_wfn = run_method<HartreeFock, R>(m, basis, config);

      // Run MP2 calculation
      return run_mp2_method(scf_wfn, config);
    }
    case MethodKind::CCSD:
    case MethodKind::CCSD_T: {
      // CCSD(T) requires an SCF reference first (restricted -> spin-adapted
      // backends; unrestricted -> spin-orbital path).
      Wavefunction scf_wfn = (guess_sk == U || conf_sk == U)
                                 ? run_method<HartreeFock, U>(m, basis, config)
                                 : run_method<HartreeFock, R>(m, basis, config);
      return run_ccsd_method(scf_wfn, config,
                             method_kind == MethodKind::CCSD_T);
    }
    default: {
      throw std::runtime_error("Unknown method kind");
    }
    }
  } else {
    switch (method_kind) {
    case MethodKind::HF: {
      if (guess_sk == U || conf_sk == U)
        return run_solvated_method<HartreeFock, U>(*guess, config);
      else if (guess_sk == G || conf_sk == G)
        return run_solvated_method<HartreeFock, G>(*guess, config);
      else
        return run_solvated_method<HartreeFock, R>(*guess, config);
      break;
    }
    case MethodKind::DFT: {
      if (guess_sk == U || conf_sk == U)
        return run_solvated_method<DFT, U>(*guess, config);
      else
        return run_solvated_method<DFT, R>(*guess, config);
      break;
    }
    case MethodKind::MP2: {
      // MP2 with solvation: run solvated SCF first, then MP2
      Wavefunction scf_wfn;
      if (guess_sk == U || conf_sk == U)
        scf_wfn = run_solvated_method<HartreeFock, U>(*guess, config);
      else if (guess_sk == G || conf_sk == G)
        scf_wfn = run_solvated_method<HartreeFock, G>(*guess, config);
      else
        scf_wfn = run_solvated_method<HartreeFock, R>(*guess, config);

      // Run MP2 calculation
      return run_mp2_method(scf_wfn, config);
    }
    case MethodKind::CCSD:
    case MethodKind::CCSD_T: {
      // CCSD(T) on a solvated SCF reference (spin-orbital path if unrestricted).
      Wavefunction scf_wfn =
          (guess_sk == U || conf_sk == U)
              ? run_solvated_method<HartreeFock, U>(*guess, config)
              : run_solvated_method<HartreeFock, R>(*guess, config);
      return run_ccsd_method(scf_wfn, config,
                             method_kind == MethodKind::CCSD_T);
    }
    default: {
      throw std::runtime_error("Unknown method kind");
    }
    }
  }
}

Wavefunction single_point(const OccInput &config) {
  return single_point_driver(config);
}

Wavefunction single_point(const OccInput &config, const Wavefunction &wfn) {
  return single_point_driver(config, wfn);
}

} // namespace occ::driver
