#include <LBFGS.h>
#include <fmt/os.h>
#include <occ/dft/dft.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/method_parser.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>

using occ::core::Molecule;
using occ::dft::DFT;
using occ::io::OccInput;
using occ::qm::HartreeFock;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

namespace occ::driver {

occ::qm::AOBasis load_basis_set_opt(const Molecule &m, const std::string &name,
                                    bool spherical) {
  auto basis = occ::qm::AOBasis::load(m.atoms(), name);
  basis.set_pure(spherical);
  log::info("Loaded basis set: {}", spherical ? "spherical" : "cartesian");
  log::info("Number of shells:            {}", basis.size());
  log::info("Number of  basis functions:  {}", basis.nbf());
  log::info("Maximum angular momentum:    {}", basis.l_max());
  return basis;
}

template <typename T, SpinorbitalKind SK>
std::pair<Wavefunction, Mat3N>
run_method_for_optimization(const Molecule &m, const occ::qm::AOBasis &basis,
                            const OccInput &config) {

  T proc = [&]() {
    if constexpr (std::is_same<T, DFT>::value)
      return T(config.method.name, basis, config.method.dft_grid);
    else
      return T(basis);
  }();

  if (!config.basis.df_name.empty())
    proc.set_density_fitting_basis(config.basis.df_name);

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
  scf.set_point_charges(config.geometry.point_charges);
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

  if (config.method.orbital_smearing_sigma != 0.0) {
    log::info("Correlation entropy approx.      {: 20.12f}",
              scf.ctx.mo.smearing.ec_entropy());
    log::info("Free energy                      {: 20.12f}",
              e + scf.ctx.mo.smearing.ec_entropy());
    log::info("Energy (zero point)              {: 20.12f}",
              e + 0.5 * scf.ctx.mo.smearing.ec_entropy());
  }

  auto wfn = scf.wavefunction();
  occ::qm::GradientEvaluator eval(scf.m_procedure);
  return {wfn, eval(wfn.mo)};
}

std::pair<Wavefunction, Mat3N> optimization_step_driver(const OccInput &config,
                                                        const Molecule &m) {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;

  if (!config.basis.basis_set_directory.empty()) {
    occ::log::info("Overriding environment basis set directory with: '{}'",
                   config.basis.basis_set_directory);
    occ::qm::override_basis_set_directory(config.basis.basis_set_directory);
  }
  auto basis = load_basis_set_opt(m, config.basis.name, config.basis.spherical);
  auto method_kind = method_kind_from_string(config.method.name);
  auto guess_sk = determine_spinorbital_kind(
      config.method.name, config.electronic.multiplicity, method_kind);
  auto conf_sk = config.electronic.spinorbital_kind;

  if (config.solvent.solvent_name.empty()) {
    switch (method_kind) {
    case MethodKind::HF: {
      if (guess_sk == U || conf_sk == U)
        return run_method_for_optimization<HartreeFock, U>(m, basis, config);
      else if (guess_sk == G || conf_sk == G)
        return run_method_for_optimization<HartreeFock, G>(m, basis, config);
      else
        return run_method_for_optimization<HartreeFock, R>(m, basis, config);
      break;
    }
    case MethodKind::DFT: {
      throw std::runtime_error("Not implemented: DFT gradients");
      break;
    }
    }
  } else {
    throw std::runtime_error("Not implemented: Solvated gradients");
  }
}

Wavefunction geometry_optimization(const OccInput &config) {

  Wavefunction wfn;
  Mat3N gradient;
  Molecule m = config.geometry.molecule();
  int current_step{0};
  auto traj = fmt::output_file("opt_traj.xyz");

  auto step = [&](const Vec &x, Vec &grad) {
    double eprev = wfn.energy.total;
    Eigen::Map<const Mat3N> pos(x.data(), 3, x.rows() / 3);
    Molecule m_step(m.atomic_numbers(), pos * occ::units::BOHR_TO_ANGSTROM);
    const auto &el = m_step.elements();
    const auto &positions = m_step.positions();
    occ::log::info("Computing wavefunction for optimization step {}",
                   current_step++);
    std::tie(wfn, gradient) = optimization_step_driver(config, m_step);

    traj.print("{}\nEnergy={:.9f}\n", el.size(), wfn.energy.total);
    for (int i = 0; i < el.size(); i++) {
      traj.print("{} {:12.6f} {:12.6f} {:12.6f}\n", el[i].symbol(),
                 positions(0, i), positions(1, i), positions(2, i));
    }

    Eigen::Map<Mat3N> grad_out(grad.data(), 3, grad.rows() / 3);
    occ::log::info("RMS gradient: {:20.12e}", gradient.norm());
    occ::log::info("Total energy: {:20.12f}", wfn.energy.total);
    occ::log::info("dE:           {:20.12e}", wfn.energy.total - eprev);
    grad_out = gradient;
    return wfn.energy.total;
  };

  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-6;
  param.max_iterations = 100;

  LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBracketing> solver(param);
  Vec x(3 * m.positions().cols());
  Eigen::Map<Mat3N>(x.data(), 3, m.positions().cols()) =
      m.positions() * occ::units::ANGSTROM_TO_BOHR;

  double energy;

  int niter = solver.minimize(step, x, energy);
  occ::log::info("Optimization complete after {} iterations", niter);
  occ::log::info("Final energy: {:20.12f}", energy);

  return wfn;
}

} // namespace occ::driver
