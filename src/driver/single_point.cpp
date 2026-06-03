#include <fmt/core.h>
#include <occ/core/constants.h>
#include <occ/core/data_directory.h>
#include <occ/dft/dft.h>
#include <occ/disp/d4.h>
#include <occ/driver/acceleration.h>
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
  occ::log::trace("Setting system charge: {}", config.electronic.charge);
  occ::log::trace("Setting system multiplicity: {}",
                  config.electronic.multiplicity);
  scf.set_charge_multiplicity(config.electronic.charge,
                              config.electronic.multiplicity);
  if (!config.geometry.point_charges.empty()) {
    scf.set_external_potential(
        occ::qm::PointChargePotential{config.geometry.point_charges});
  }
  if (!config.basis.df_name.empty()) {
    scf.convergence_settings.incremental_fock_threshold = 0.0;
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

      auto [e_xdm, grad_xdm] = qm::impl::compute_xdm_dispersion(
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
    scf.set_charge_multiplicity(config.electronic.charge,
                                config.electronic.multiplicity);
    scf.convergence_settings.incremental_fock_threshold = 0.0;
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
    scf.set_charge_multiplicity(config.electronic.charge,
                                config.electronic.multiplicity);
    scf.set_initial_guess_from_wfn(wfn);
    scf.convergence_settings.incremental_fock_threshold = 0.0;
    double e = scf.compute_scf_energy();
    if (!config.solvent.output_surface_filename.empty())
      proc_solv.write_surface_file(config.solvent.output_surface_filename);
    return scf.wavefunction();
  }
}

Wavefunction run_mp2_method(const Wavefunction &scf_wfn,
                            const OccInput &config) {
  using occ::qm::MP2;
  namespace cc = occ::qm::cc;

  occ::log::info("{:=^72s}", "  MP2 Calculation  ");

  // Resolve the effective MP2 backend. Method-name prefixes ("ri-mp2",
  // "thc-mp2") select a backend when --mp2-backend is left at "auto"; "ri"/"df"
  // are synonyms for RI-MP2. An RI aux basis is resolved when RI is requested
  // without an explicit --ri-basis.
  const auto mspec = parse_method_string(config.method.name);
  std::string mp2_backend = occ::util::to_lower_copy(config.method.mp2_backend);
  if (mp2_backend == "auto" && !mspec.backend.empty())
    mp2_backend = mspec.backend;
  if (mp2_backend == "ri")
    mp2_backend = "df";
  std::string ri_basis = config.basis.ri_basis;
  if (ri_basis.empty() && mp2_backend == "df")
    ri_basis = occ::qm::resolve_fitting_basis(
        config.basis.name, occ::qm::FittingKind::Correlation);

  // THC-MP2 backend (LS-THC: Laplace denominator + THC factors). Lives in
  // occ_cc as a free function (the MP2 class is in occ_correlation, which occ_cc
  // depends on, so it can't host THC). Fills same/opposite spin so SCS/SOS work.
  if (mp2_backend == "thc") {
    const std::string auxname =
        config.basis.ri_basis.empty()
            ? occ::qm::resolve_fitting_basis(config.basis.name,
                                             occ::qm::FittingKind::Correlation)
            : config.basis.ri_basis;
    occ::log::info("Method: THC-MP2 (auxiliary basis: {})", auxname);
    auto aux = load_basis_set(config.geometry.molecule(), auxname,
                              config.basis.spherical);

    cc::ThcMP2Options opts;
    opts.thc.c_isdf = config.method.mp2_thc_c_isdf;
    const std::string sel = occ::util::to_lower_copy(config.method.mp2_thc_method);
    opts.thc.method =
        (sel == "qr") ? cc::IsdfMethod::QR : cc::IsdfMethod::Cholesky;
    opts.n_laplace = config.method.mp2_laplace_points;
    opts.memory_budget = static_cast<size_t>(
        config.method.mp2_max_memory_gb * 1024.0 * 1024.0 * 1024.0);
    opts.thc.memory_budget = opts.memory_budget;
    int n_frozen = cc::num_frozen_core(scf_wfn.basis);
    n_frozen = std::max(
        0, std::min(n_frozen, static_cast<int>(scf_wfn.mo.n_alpha) - 1));
    opts.n_frozen = n_frozen;

    // SOS-MP2 uses only the opposite-spin energy, so skip the same-spin
    // exchange entirely -> the whole calculation is the O(P^3) cubic Coulomb
    // path (the genuinely fast, large-system THC win).
    const std::string &scaling = config.method.mp2_spin_scaling;
    const bool scaled = (scaling == "scs" || scaling == "sos");
    double c_ss = 1.0, c_os = 1.0;
    if (scaling == "scs") {
      c_ss = 1.0 / 3.0;
      c_os = 6.0 / 5.0;
    } else if (scaling == "sos") {
      c_ss = 0.0;
      c_os = 1.3;
      opts.opposite_spin_only = true;
    } else if (scaling != "none" && !scaling.empty()) {
      occ::log::warn("Unknown --mp2-spin-scaling '{}', using unscaled MP2",
                     scaling);
    }
    occ::log::info("THC rank c = {}, ISDF selector = {}, Laplace points = {}, "
                   "frozen core = {}{}",
                   opts.thc.c_isdf, sel == "qr" ? "qr" : "cholesky",
                   opts.n_laplace, n_frozen,
                   opts.opposite_spin_only ? " [opposite-spin only]" : "");

    const auto r = cc::thc_mp2(scf_wfn.basis, aux, scf_wfn.mo, opts);

    const double scaled_corr = c_ss * r.same_spin + c_os * r.opposite_spin;
    const double used_corr = scaled ? scaled_corr : r.total;
    const double total_energy = scf_wfn.energy.total + used_corr;

    occ::log::info(
        "THC-MP2: {} interpolation points, {} Laplace points (max rel err "
        "{:.2e})",
        r.n_isdf, r.n_laplace, r.laplace_max_rel_error);
    occ::log::info("SCF energy:                       {: 20.12f}",
                   scf_wfn.energy.total);
    if (!opts.opposite_spin_only) {
      occ::log::info("MP2 correlation energy:           {: 20.12f}", r.total);
      occ::log::info("  same-spin:                      {: 20.12f}",
                     r.same_spin);
    }
    occ::log::info("  opposite-spin:                  {: 20.12f}",
                   r.opposite_spin);
    if (scaled)
      occ::log::info("{}-MP2 correlation energy:        {: 20.12f}",
                     scaling == "scs" ? "SCS" : "SOS", scaled_corr);
    occ::log::info("MP2 total energy:                 {: 20.12f}", total_energy);

    Wavefunction mp2_wfn = scf_wfn;
    mp2_wfn.energy.total = total_energy;
    mp2_wfn.method = scaling == "scs"   ? "SCS-THC-MP2"
                     : scaling == "sos" ? "SOS-THC-MP2"
                                        : "THC-MP2";
    return mp2_wfn;
  }

  MP2 mp2 = [&]() {
    if (!ri_basis.empty()) {
      occ::log::info("Method: RI-MP2 (auxiliary basis: {})", ri_basis);
      auto aux_basis = load_basis_set(config.geometry.molecule(), ri_basis,
                                      config.basis.spherical);
      return MP2(scf_wfn.basis, aux_basis, scf_wfn.mo, scf_wfn.energy.total);
    } else {
      occ::log::info("Method: Conventional MP2");
      return MP2(scf_wfn.basis, scf_wfn.mo, scf_wfn.energy.total);
    }
  }();

  // Set automatic frozen core
  mp2.set_frozen_core_auto();

  // Memory budget (GiB -> bytes) controls occupied blocking and whether the
  // dense 3-center store is used.
  mp2.set_memory_budget(static_cast<size_t>(config.method.mp2_max_memory_gb *
                                            1024.0 * 1024.0 * 1024.0));

  // Optional spin-component scaling.
  const std::string &scaling = config.method.mp2_spin_scaling;
  const bool scaled = (scaling == "scs" || scaling == "sos");
  if (scaling == "scs") {
    mp2.set_scs_parameters(1.0 / 3.0, 6.0 / 5.0); // Grimme SCS-MP2
  } else if (scaling == "sos") {
    mp2.set_scs_parameters(0.0, 1.3); // Grimme SOS-MP2
  } else if (scaling != "none" && !scaling.empty()) {
    occ::log::warn("Unknown --mp2-spin-scaling '{}', using unscaled MP2",
                   scaling);
  }

  // Compute MP2 correlation energy
  double correlation_energy = mp2.compute_correlation_energy();
  const auto &results = mp2.results();
  const double used_corr =
      scaled ? results.scs_mp2_correlation : correlation_energy;
  double total_mp2_energy = scf_wfn.energy.total + used_corr;

  occ::log::info("SCF energy:                       {: 20.12f}",
                 scf_wfn.energy.total);
  occ::log::info("MP2 correlation energy:           {: 20.12f}",
                 correlation_energy);
  occ::log::info("  same-spin:                      {: 20.12f}",
                 results.same_spin_correlation);
  occ::log::info("  opposite-spin:                  {: 20.12f}",
                 results.opposite_spin_correlation);
  if (scaled) {
    occ::log::info("{}-MP2 correlation energy:        {: 20.12f}",
                   scaling == "scs" ? "SCS" : "SOS",
                   results.scs_mp2_correlation);
  }
  occ::log::info("MP2 total energy:                 {: 20.12f}",
                 total_mp2_energy);

  // Create modified wavefunction with MP2 energy
  Wavefunction mp2_wfn = scf_wfn;
  mp2_wfn.energy.total = total_mp2_energy;
  mp2_wfn.method = scaling == "scs"   ? "SCS-MP2"
                   : scaling == "sos" ? "SOS-MP2"
                                      : "MP2";

  return mp2_wfn;
}

Wavefunction run_ccsd_method(const Wavefunction &scf_wfn, const OccInput &config,
                             bool with_triples) {
  namespace cc = occ::qm::cc;
  const bool open_shell = scf_wfn.mo.kind != SpinorbitalKind::Restricted;

  // Frozen core: -1 = auto (chemical core, the standard CCSD(T) default), 0 =
  // none (all-electron), N = freeze N lowest occupied orbitals.
  const bool fc_auto0 = config.method.ccsd_frozen_core < 0;
  int n_frozen0 = fc_auto0 ? cc::num_frozen_core(scf_wfn.basis)
                           : config.method.ccsd_frozen_core;

  // Resolve the effective CCSD backend. Method-name prefixes ("ri-ccsd(t)",
  // "thc-ccsd(t)") select a backend when --ccsd-backend is left at its default
  // ("exact"); "ri" and "df" are synonyms for density fitting.
  const auto mspec = parse_method_string(config.method.name);
  std::string backend = occ::util::to_lower_copy(config.method.ccsd_backend);
  if (backend.empty())
    backend = "exact";
  if (backend == "ri")
    backend = "df";
  if (backend == "exact" && !mspec.backend.empty())
    backend = mspec.backend;

  if (open_shell) {
    // Open shell uses the spin-adapted unrestricted CCSD(T) (exact / df / thc).
    occ::log::info("{:=^72s}",
                   with_triples ? "  UHF CCSD(T) Calculation  "
                                : "  UHF CCSD Calculation  ");
    const std::string &be = backend;
    const size_t nocc_min = std::min(scf_wfn.mo.n_alpha, scf_wfn.mo.n_beta);
    n_frozen0 = std::max(0, std::min(n_frozen0, static_cast<int>(nocc_min) - 1));
    occ::log::info("Backend: {}", be);
    occ::log::info("Frozen core: {} orbitals ({} electrons){}", n_frozen0,
                   2 * n_frozen0, fc_auto0 ? " [auto]" : "");

    cc::UCCSDOptions uopts;
    uopts.backend = be;
    uopts.n_frozen = n_frozen0;
    uopts.with_triples = with_triples;
    uopts.memory_budget = static_cast<size_t>(
        config.method.ccsd_max_memory_gb * 1024.0 * 1024.0 * 1024.0);
    uopts.thc.c_isdf = config.method.ccsd_thc_c_isdf;
    uopts.thc.memory_budget = uopts.memory_budget;
    uopts.thc.grid_max_angular = config.method.ccsd_thc_grid_angular;
    uopts.thc.grid_radial_precision = config.method.ccsd_thc_grid_radial;
    uopts.thc.method =
        (occ::util::to_lower_copy(config.method.ccsd_thc_method) == "qr")
            ? cc::IsdfMethod::QR
            : cc::IsdfMethod::Cholesky;

    cc::UCCSDResult r;
    if (be == "exact") {
      occ::log::info("Note: the exact backend stores the O(V^4) vvvv block; "
                     "use df or thc for larger systems.");
      r = cc::uccsd(scf_wfn.basis, scf_wfn.mo, uopts);
    } else {
      const std::string auxname =
          config.basis.ri_basis.empty()
              ? occ::qm::resolve_fitting_basis(config.basis.name,
                                               occ::qm::FittingKind::Correlation)
              : config.basis.ri_basis;
      occ::log::info("Auxiliary basis: {}", auxname);
      auto aux = load_basis_set(config.geometry.molecule(), auxname,
                                config.basis.spherical);
      r = cc::uccsd(scf_wfn.basis, aux, scf_wfn.mo, uopts);
    }
    if (!r.converged)
      occ::log::warn("CCSD did not converge in {} iterations", r.iterations);
    const double total = scf_wfn.energy.total + r.e_corr + r.e_triples;
    occ::log::info("SCF energy:                       {: 20.12f}",
                   scf_wfn.energy.total);
    occ::log::info("CCSD correlation energy:          {: 20.12f}", r.e_corr);
    if (with_triples) {
      occ::log::info("(T) correction:                   {: 20.12f}",
                     r.e_triples);
      occ::log::info("CCSD(T) correlation energy:       {: 20.12f}",
                     r.e_corr + r.e_triples);
    }
    occ::log::info("{:<33s} {: 20.12f}",
                   with_triples ? "CCSD(T) total energy:" : "CCSD total energy:",
                   total);
    Wavefunction cc_wfn = scf_wfn;
    cc_wfn.energy.total = total;
    cc_wfn.method = with_triples ? "CCSD(T)" : "CCSD";
    return cc_wfn;
  }

  const std::string label = with_triples ? "  CCSD(T) Calculation  "
                                          : "  CCSD Calculation  ";
  occ::log::info("{:=^72s}", label);

  const size_t budget = static_cast<size_t>(config.method.ccsd_max_memory_gb *
                                            1024.0 * 1024.0 * 1024.0);
  occ::log::info("Backend: {}", backend);

  // Frozen core: -1 = auto (chemical core, the standard CCSD(T) default), 0 =
  // none (all-electron), N = freeze N lowest occupied orbitals.
  const bool fc_auto = config.method.ccsd_frozen_core < 0;
  int n_frozen = fc_auto ? cc::num_frozen_core(scf_wfn.basis)
                         : config.method.ccsd_frozen_core;
  n_frozen = std::max(0, std::min(n_frozen,
                                  static_cast<int>(scf_wfn.mo.n_alpha) - 1));
  occ::log::info("Frozen core: {} orbitals ({} electrons){}", n_frozen,
                 2 * n_frozen, fc_auto ? " [auto]" : "");

  cc::CCIntegrals eris = [&]() -> cc::CCIntegrals {
    if (backend == "exact") {
      occ::log::info("Note: the exact backend stores the O(V^4) vvvv block; "
                     "use df or thc for larger systems.");
      return cc::exact_eris(scf_wfn.basis, scf_wfn.mo, n_frozen, budget);
    }
    const std::string auxname =
        config.basis.ri_basis.empty()
            ? occ::qm::resolve_fitting_basis(config.basis.name,
                                             occ::qm::FittingKind::Correlation)
            : config.basis.ri_basis;
    occ::log::info("Auxiliary basis: {}", auxname);
    auto aux = load_basis_set(config.geometry.molecule(), auxname,
                              config.basis.spherical);
    if (backend == "df")
      return cc::df_eris(scf_wfn.basis, aux, scf_wfn.mo, n_frozen, budget);
    if (backend == "thc") {
      cc::ThcOptions opts;
      opts.c_isdf = config.method.ccsd_thc_c_isdf;
      opts.memory_budget = budget;
      opts.grid_max_angular = config.method.ccsd_thc_grid_angular;
      opts.grid_radial_precision = config.method.ccsd_thc_grid_radial;
      const std::string sel =
          occ::util::to_lower_copy(config.method.ccsd_thc_method);
      opts.method = (sel == "qr") ? cc::IsdfMethod::QR : cc::IsdfMethod::Cholesky;
      occ::log::info("THC rank c = {}, ISDF selector = {}", opts.c_isdf,
                     sel == "qr" ? "qr" : "cholesky");
      return cc::thc_eris(scf_wfn.basis, aux, scf_wfn.mo, opts, n_frozen, budget);
    }
    throw std::runtime_error("Unknown CCSD backend '" + backend +
                             "' (expected exact | df | thc)");
  }();

  const cc::CCSDResult res = cc::ccsd(eris);
  if (!res.converged)
    occ::log::warn("CCSD did not converge in {} iterations", res.iterations);

  const double et = with_triples ? cc::ccsd_t(res.t1, res.t2, eris) : 0.0;
  const double total = scf_wfn.energy.total + res.e_corr + et;

  occ::log::info("SCF energy:                       {: 20.12f}",
                 scf_wfn.energy.total);
  occ::log::info("CCSD correlation energy:          {: 20.12f}", res.e_corr);
  if (with_triples) {
    occ::log::info("(T) correction:                   {: 20.12f}", et);
    occ::log::info("CCSD(T) correlation energy:       {: 20.12f}",
                   res.e_corr + et);
  }
  occ::log::info("{:<33s} {: 20.12f}",
                 with_triples ? "CCSD(T) total energy:" : "CCSD total energy:",
                 total);

  Wavefunction cc_wfn = scf_wfn;
  cc_wfn.energy.total = total;
  cc_wfn.method = with_triples ? "CCSD(T)" : "CCSD";
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
