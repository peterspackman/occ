#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/disp/d4.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_calculator.h>
#include <occ/xtb/h0_gradient.h>
#include <occ/xtb/kpoint_grid.h>
#include <occ/xtb/native_calculator.h>
#include <occ/xtb/repulsion.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

std::vector<core::Atom> make_atoms(const Mat3N &positions_bohr,
                                   const IVec &atomic_numbers) {
  std::vector<core::Atom> atoms;
  atoms.reserve(atomic_numbers.size());
  for (Eigen::Index i = 0; i < atomic_numbers.size(); ++i) {
    atoms.push_back({atomic_numbers(i), positions_bohr(0, i),
                     positions_bohr(1, i), positions_bohr(2, i)});
  }
  return atoms;
}

// Wiberg bond orders: Σ_{μ∈A, ν∈B} (P·S)_μν · (P·S)_νμ.
Mat compute_wiberg_bond_orders(const Mat &P, const Mat &S,
                               const std::vector<int> &bf_to_atom,
                               int n_atoms) {
  Mat PS = P * S;
  Mat wb = Mat::Zero(n_atoms, n_atoms);
  for (Eigen::Index mu = 0; mu < PS.rows(); ++mu) {
    const int ai = bf_to_atom[mu];
    for (Eigen::Index nu = 0; nu < PS.cols(); ++nu) {
      const int aj = bf_to_atom[nu];
      if (ai == aj)
        continue;
      wb(ai, aj) += PS(mu, nu) * PS(nu, mu);
    }
  }
  return wb;
}

} // namespace

NativeCalculator::NativeCalculator(const core::Molecule &mol)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_charge(mol.charge()) {
  initialize_calculator();
}

NativeCalculator::NativeCalculator(const core::Dimer &dimer)
    : m_positions_bohr(dimer.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(dimer.atomic_numbers()), m_charge(dimer.charge()) {
  initialize_calculator();
}

NativeCalculator::NativeCalculator(const crystal::Crystal &crystal) {
  m_periodic = true;
  m_periodic_sys = PeriodicSystem::from_crystal(crystal);
  // Mirror molecular positions / atomic numbers so positions(), num_atoms(),
  // and to_wavefunction() can act on the central-cell atoms.
  const int n = m_periodic_sys.num_atoms();
  m_positions_bohr.resize(3, n);
  m_atomic_numbers.resize(n);
  for (int i = 0; i < n; ++i) {
    m_positions_bohr(0, i) = m_periodic_sys.atoms[i].x;
    m_positions_bohr(1, i) = m_periodic_sys.atoms[i].y;
    m_positions_bohr(2, i) = m_periodic_sys.atoms[i].z;
    m_atomic_numbers(i) = m_periodic_sys.atoms[i].atomic_number;
  }
  m_params = std::make_shared<Gfn2Parameters>(Gfn2Parameters::load_default());
  m_periodic_opts.total_charge = m_charge;
}

NativeCalculator::~NativeCalculator() = default;

void NativeCalculator::set_kpoints(int n1, int n2, int n3) {
  m_kpoints[0] = n1;
  m_kpoints[1] = n2;
  m_kpoints[2] = n3;
}

void NativeCalculator::set_include_multipoles(bool on) {
  m_periodic_opts.include_multipoles = on;
  // The molecular path always honors multipoles via single_point(opts, true).
  // We stash the flag for periodic dispatch; molecular ignores this setter.
}

void NativeCalculator::initialize_calculator() {
  m_params = std::make_shared<Gfn2Parameters>(Gfn2Parameters::load_default());
  m_calc = std::make_unique<Gfn2Calculator>(
      make_atoms(m_positions_bohr, m_atomic_numbers), *m_params);
  m_opts.total_charge = m_charge;
}

double NativeCalculator::single_point_energy() {
  if (m_periodic) {
    m_periodic_opts.total_charge = m_charge;
    if (m_kpoints[0] == 1 && m_kpoints[1] == 1 && m_kpoints[2] == 1) {
      m_periodic_result = run_charge_only_periodic_scc(
          m_periodic_sys, *m_params, m_periodic_opts);
    } else {
      auto kpts = monkhorst_pack_grid(m_periodic_sys.reciprocal_bohr(),
                                      m_kpoints[0], m_kpoints[1],
                                      m_kpoints[2]);
      m_periodic_result =
          run_periodic_scc_kpoints(m_periodic_sys, *m_params, kpts,
                                    m_periodic_opts);
    }
    // Mirror into m_last_result so the molecular accessors (charges,
    // bond_orders) keep working on the central-cell density.
    m_last_result.scc_energy = m_periodic_result.scc_energy;
    m_last_result.repulsion_energy = m_periodic_result.repulsion_energy;
    m_last_result.dispersion_energy = 0.0;
    m_last_result.total_energy = m_periodic_result.total_energy;
    m_last_result.shell_charges = m_periodic_result.shell_charges;
    m_last_result.atomic_charges = m_periodic_result.atomic_charges;
    m_last_result.orbital_energies = m_periodic_result.orbital_energies;
    m_last_result.orbital_occupations = m_periodic_result.orbital_occupations;
    m_last_result.density_matrix = m_periodic_result.density_matrix;
    m_last_result.overlap_matrix = m_periodic_result.overlap_matrix;
    m_last_result.orbital_coefficients =
        m_periodic_result.orbital_coefficients;
    m_last_result.n_iterations = m_periodic_result.n_iterations;
    m_last_result.converged = m_periodic_result.converged;
    return m_periodic_result.total_energy;
  }
  m_opts.total_charge = m_charge;
  m_last_result = m_calc->single_point(m_opts, /*include_multipoles=*/true);
  return m_last_result.total_energy;
}

Vec NativeCalculator::charges() const { return m_last_result.atomic_charges; }

Mat NativeCalculator::bond_orders() const {
  if (m_last_result.density_matrix.size() == 0) {
    throw std::runtime_error(
        "NativeCalculator::bond_orders: call single_point_energy() first");
  }
  return compute_wiberg_bond_orders(m_last_result.density_matrix,
                                    m_last_result.overlap_matrix,
                                    m_calc->bf_to_atom(),
                                    static_cast<int>(num_atoms()));
}

void NativeCalculator::set_charge(double c) { m_charge = c; }
void NativeCalculator::set_max_iterations(int n) { m_opts.max_iterations = n; }
void NativeCalculator::set_temperature(double t) {
  m_opts.electronic_temperature = t;
}
void NativeCalculator::set_mixer_damping(double f) {
  m_opts.damping_factor = f;
}

void NativeCalculator::update_structure(const Mat3N &positions) {
  if (positions.cols() != num_atoms()) {
    throw std::runtime_error(
        "NativeCalculator::update_structure: column count mismatch");
  }
  m_positions_bohr = positions;
  m_calc->update_positions(make_atoms(m_positions_bohr, m_atomic_numbers));
}

core::Molecule NativeCalculator::to_molecule() const {
  return core::Molecule(m_atomic_numbers,
                        m_positions_bohr / occ::units::BOHR_TO_ANGSTROM);
}

Mat3N NativeCalculator::compute_gradient_numerical(double step) {
  const int n = num_atoms();
  Mat3N grad = Mat3N::Zero(3, n);

  // Snapshot the original geometry so we can restore exactly.
  Mat3N original = m_positions_bohr;

  // Suppress per-iteration SCC chatter and crank max iterations to be safe
  // for displaced geometries that may converge slower.
  auto prev_max_iter = m_opts.max_iterations;
  m_opts.max_iterations = std::max(prev_max_iter, 250);

  for (int a = 0; a < n; ++a) {
    for (int k = 0; k < 3; ++k) {
      Mat3N pos_p = original;
      pos_p(k, a) += step;
      update_structure(pos_p);
      const double e_plus = single_point_energy();

      Mat3N pos_m = original;
      pos_m(k, a) -= step;
      update_structure(pos_m);
      const double e_minus = single_point_energy();

      grad(k, a) = (e_plus - e_minus) / (2.0 * step);
    }
  }

  // Restore the original geometry and recompute so subsequent queries
  // (charges(), bond_orders(), to_wavefunction()) reflect the input point.
  update_structure(original);
  (void)single_point_energy();
  m_opts.max_iterations = prev_max_iter;
  return grad;
}

Mat3N NativeCalculator::compute_gradient_analytical() {
  // Run a charge-only SCC to get a converged density consistent with the
  // pieces we differentiate analytically. Multipole anisotropic terms are
  // NOT in this energy expression — see header docstring.
  m_opts.total_charge = m_charge;
  SccOptions opts = m_opts;
  // Dispersion is included in the analytical pipeline via native D4 below;
  // disable it inside the SCC so we don't double-count the energy and so we
  // get the *raw* SCC charges for the dispersion piece.
  const bool wanted_disp = opts.include_dispersion;
  opts.include_dispersion = false;
  m_last_result = m_calc->run_charge_only(opts);
  if (!m_last_result.converged) {
    throw std::runtime_error(
        "NativeCalculator::compute_gradient_analytical: charge-only SCC did "
        "not converge");
  }

  // Build the energy-weighted density W = 2 Σ_i^occ ε_i C_i C_i^T
  // (closed shell — sum over both spins folded in via the factor 2).
  const auto &C = m_last_result.orbital_coefficients;
  const auto &eps = m_last_result.orbital_energies;
  const int n_occ = static_cast<int>(eps.size()) / 2; // closed-shell SCC
  Mat W = Mat::Zero(C.rows(), C.rows());
  for (int i = 0; i < n_occ; ++i) {
    W.noalias() += 2.0 * eps(i) * C.col(i) * C.col(i).transpose();
  }

  // Coordination numbers + ∂CN/∂R for the H0+self-energy chain.
  auto cn_g = gfn_coordination_numbers_with_gradient(m_calc->atoms());

  // Shell shift potential V = J·qsh + Γ_3 q² (matches the SCC's F = H0 - V).
  Vec V_shell = m_calc->gamma() * m_last_result.shell_charges;
  const auto &shells = m_calc->shell_table();
  for (Eigen::Index s = 0; s < V_shell.size(); ++s) {
    V_shell(s) +=
        shells.third_order(s) * m_last_result.shell_charges(s) *
        m_last_result.shell_charges(s);
  }

  // (1) H0 + Pulay + V_q-via-S + ∂Π/∂R + dE/dCN chain through self-energy.
  Mat3N grad = h0_scc_gradient(
      m_calc->atoms(), m_calc->parameters(), shells, m_calc->basis(),
      m_calc->engine(), m_last_result.overlap_matrix,
      m_last_result.density_matrix, W, V_shell, cn_g.cn, cn_g.dcn);

  // (2) ½ q^T (∂γ/∂R) q  (analytical Klopman-Ohno γ derivative).
  grad += klopman_ohno_gamma_energy_gradient(
      m_calc->atoms(), shells, m_calc->parameters(), m_calc->gamma(),
      m_last_result.shell_charges);

  // (3) Repulsion derivative (closed form).
  auto rep = repulsion_energy_and_gradient(m_calc->atoms(), m_calc->parameters());
  grad += rep.gradient;

  // (4) Native D4 dispersion. SCC-coupled (atomic Mulliken charges as fixed
  // input — variational q ⇒ ∂q/∂R chain vanishes by Hellmann-Feynman; this
  // matches xtb's d4_gradient convention).
  double e_disp = 0.0;
  if (wanted_disp) {
    const auto &g = m_calc->parameters().globals();
    occ::disp::Dispersion d4(m_calc->atoms());
    d4.set_damping(occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
    Vec atom_q = Vec::Zero(m_calc->atoms().size());
    for (Eigen::Index s = 0; s < m_last_result.shell_charges.size(); ++s) {
      atom_q(shells.atom[s]) += m_last_result.shell_charges(s);
    }
    d4.set_charges(atom_q);
    auto [ed, gd] = d4.energy_and_gradient();
    e_disp = ed;
    grad += gd;
  }

  // Update last_result so callers see the energy that matches the gradient
  // (charge-only SCC + native dispersion).
  m_last_result.dispersion_energy = e_disp;
  m_last_result.total_energy = m_last_result.scc_energy +
                                m_last_result.repulsion_energy + e_disp;
  occ::log::debug("analytical gradient: scc={:.10f} rep={:.10f} disp={:.10f} total={:.10f}",
                   m_last_result.scc_energy, m_last_result.repulsion_energy,
                   e_disp, m_last_result.total_energy);
  return grad;
}

std::pair<double, Mat3N>
NativeCalculator::compute_energy_and_gradient(bool numerical, double step) {
  if (numerical) {
    Mat3N g = compute_gradient_numerical(step);
    return {m_last_result.total_energy, g};
  }
  Mat3N g = compute_gradient_analytical();
  return {m_last_result.total_energy, g};
}

void NativeCalculator::print_summary() const {
  if (m_last_result.density_matrix.size() == 0) {
    occ::log::warn("NativeCalculator::print_summary: nothing to print "
                   "(call single_point_energy() first)");
    return;
  }
  const auto &r = m_last_result;
  occ::log::info("{:=^72s}", "  GFN2-xTB results  ");
  occ::log::info("{:<32s} {:>20.12f} Ha", "Total energy", r.total_energy);
  occ::log::info("{:<32s} {:>20.12f} Ha", "  SCC (electronic + ES)",
                 r.scc_energy);
  occ::log::info("{:<32s} {:>20.12f} Ha", "  Repulsion", r.repulsion_energy);
  occ::log::info("{:<32s} {:>20.12f} Ha", "  Dispersion (D4)",
                 r.dispersion_energy);
  if (r.converged) {
    occ::log::info("Converged in {} SCC iterations", r.n_iterations);
  } else {
    occ::log::warn("Did NOT converge ({} iterations)", r.n_iterations);
  }

  // HOMO / LUMO from the orbital energies.
  Eigen::Index n_occ = static_cast<Eigen::Index>(num_atoms()); // placeholder
  // Find n_occ from orbital_occupations.
  n_occ = 0;
  for (Eigen::Index i = 0; i < r.orbital_occupations.size(); ++i)
    if (r.orbital_occupations(i) > 1e-6) ++n_occ;
  if (n_occ > 0 && n_occ < r.orbital_energies.size()) {
    const double homo = r.orbital_energies(n_occ - 1);
    const double lumo = r.orbital_energies(n_occ);
    const double gap = lumo - homo;
    occ::log::info("HOMO = {:>10.4f} Ha ({:>8.3f} eV)", homo,
                   homo * occ::units::AU_TO_EV);
    occ::log::info("LUMO = {:>10.4f} Ha ({:>8.3f} eV)", lumo,
                   lumo * occ::units::AU_TO_EV);
    occ::log::info("Gap  = {:>10.4f} Ha ({:>8.3f} eV)", gap,
                   gap * occ::units::AU_TO_EV);
  }

  occ::log::info("{:-<72s}", "Atomic Mulliken charges  ");
  occ::log::info("  {:>3s}  {:>4s}  {:>14s}", "idx", "Z", "q (e)");
  for (int i = 0; i < num_atoms(); ++i) {
    occ::log::info("  {:>3d}  {:>4d}  {:>14.6f}", i, m_atomic_numbers(i),
                   r.atomic_charges(i));
  }
}

occ::qm::Wavefunction NativeCalculator::to_wavefunction() const {
  if (m_last_result.density_matrix.size() == 0) {
    throw std::runtime_error(
        "NativeCalculator::to_wavefunction: call single_point_energy() first");
  }
  occ::qm::Wavefunction wfn;
  wfn.method = "GFN2-xTB";
  wfn.basis = m_calc->basis();
  wfn.nbf = wfn.basis.nbf();
  wfn.atoms = m_calc->atoms();

  // GFN2 only carries valence electrons in its basis. Tell the basis to
  // treat the rest as ECP-like core electrons so downstream analyses
  // (mulliken_charges, etc.) compute Z_eff − pop instead of Z − pop.
  std::vector<int> core_electrons(wfn.atoms.size(), 0);
  for (size_t a = 0; a < wfn.atoms.size(); ++a) {
    const auto *e = m_params->element(wfn.atoms[a].atomic_number);
    double valence = 0.0;
    for (const auto &s : e->shells)
      valence += s.ref_occ;
    core_electrons[a] = wfn.atoms[a].atomic_number -
                        static_cast<int>(std::round(valence));
  }
  wfn.basis.set_ecp_electrons(core_electrons);

  // Closed-shell: total electrons = sum of reference shell occupations - charge.
  double n_elec = 0.0;
  for (Eigen::Index i = 0; i < m_calc->shell_table().ref_occ.size(); ++i)
    n_elec += m_calc->shell_table().ref_occ(i);
  n_elec -= m_charge;
  const int n_alpha =
      static_cast<int>(std::round(n_elec)) / 2; // restricted, so n_alpha = n_occ
  wfn.num_electrons = static_cast<int>(std::round(n_elec));

  auto &mo = wfn.mo;
  mo.kind = occ::qm::SpinorbitalKind::Restricted;
  mo.n_ao = wfn.nbf;
  mo.n_alpha = n_alpha;
  mo.n_beta = n_alpha; // closed-shell
  mo.C = m_last_result.orbital_coefficients;
  mo.energies = m_last_result.orbital_energies;
  mo.D = 0.5 * m_last_result.density_matrix; // Wavefunction stores α-only D
  mo.Cocc = mo.C.leftCols(n_alpha);

  // Energies — total only. Decomposition fields aren't part of `Energy`'s
  // standard set; fold the extras into nuclear/repulsion slots informally.
  wfn.energy.total = m_last_result.total_energy;
  wfn.energy.nuclear_repulsion = m_last_result.repulsion_energy;
  wfn.have_energies = true;

  // Cache the overlap matrix on T (commonly used by downstream).
  wfn.T = m_last_result.overlap_matrix;
  return wfn;
}

} // namespace occ::xtb
