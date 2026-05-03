#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/disp/d4.h>
#include <optional>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/gfn2_calculator.h>
#include <occ/xtb/h0.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/repulsion.h>
#include <memory>
#include <stdexcept>


namespace occ::xtb {

Gfn2Calculator::Gfn2Calculator(std::vector<core::Atom> atoms,
                               Gfn2Parameters params)
    : m_atoms(std::move(atoms)), m_params(std::move(params)),
      m_basis(build_aobasis(m_atoms, m_params)),
      m_shells(build_shell_table(m_atoms, m_params)), m_engine(m_basis) {
  m_nbf = static_cast<int>(m_basis.nbf());
  m_n_shells = static_cast<int>(m_shells.atom.size());
  m_bf_to_atom = m_basis.bf_to_atom();
  m_bf_to_shell = m_basis.bf_to_shell();
  m_z_sh = m_shells.ref_occ;
  recompute_geometry_caches();
}

void Gfn2Calculator::update_positions(const std::vector<core::Atom> &atoms) {
  if (atoms.size() != m_atoms.size()) {
    throw std::runtime_error(
        "Gfn2Calculator::update_positions: atom count changed (" +
        std::to_string(atoms.size()) + " vs " + std::to_string(m_atoms.size()) +
        ")");
  }
  // Verify atomic numbers are unchanged — basis would otherwise be invalid.
  for (size_t i = 0; i < atoms.size(); ++i) {
    if (atoms[i].atomic_number != m_atoms[i].atomic_number) {
      throw std::runtime_error(
          "Gfn2Calculator::update_positions: atomic number of atom " +
          std::to_string(i) + " changed");
    }
  }
  m_atoms = atoms;

  // Rebuild basis at the new positions (cheap — just shell origin updates).
  m_basis = build_aobasis(m_atoms, m_params);
  m_engine = qm::IntegralEngine(m_basis);
  m_have_multipole_ints = false;
  recompute_geometry_caches();
}

void Gfn2Calculator::recompute_geometry_caches() {
  m_S = m_engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  m_cn = gfn_coordination_numbers(m_atoms);
  m_e_rep = ::occ::xtb::repulsion_energy(m_atoms, m_params);
  m_J = klopman_ohno_gamma(m_atoms, m_shells, m_params);
  m_H0 = build_h0(m_atoms, m_params, m_shells, m_basis, m_S, m_cn);
  m_mp_radii = multipole_radii(m_atoms, m_cn, m_params);
  m_damped = damped_multipole_coulomb(m_atoms, m_mp_radii, m_params);
  m_have_multipole_ints = false; // built on demand
}

namespace {

// Mulliken populations per shell from PS = P · S.
Vec shell_populations(const Mat &PS, const std::vector<int> &bf_to_shell,
                      int n_shells) {
  Vec pop = Vec::Zero(n_shells);
  for (Eigen::Index mu = 0; mu < PS.rows(); ++mu)
    pop(bf_to_shell[mu]) += PS(mu, mu);
  return pop;
}

} // namespace

SccResult Gfn2Calculator::single_point(const SccOptions &opts,
                                       bool include_multipoles) {
  if (opts.unpaired_electrons != 0) {
    throw std::runtime_error(
        "Gfn2Calculator: open-shell case not yet supported");
  }

  // Build dipole / quadrupole AO matrices on first multipole-enabled call.
  if (include_multipoles && !m_have_multipole_ints) {
    m_D_ao = dipole_ao_matrices(m_engine);
    m_Q_ao = quadrupole_ao_matrices(m_engine);
    m_have_multipole_ints = true;
  }

  // Closed-shell electron count.
  double n_elec_total = 0.0;
  for (Eigen::Index i = 0; i < m_z_sh.size(); ++i)
    n_elec_total += m_z_sh(i);
  n_elec_total -= opts.total_charge;
  if (std::abs(std::round(n_elec_total) - n_elec_total) > 1e-6) {
    throw std::runtime_error(
        "Gfn2Calculator: non-integer electron count not supported");
  }
  const int n_elec = static_cast<int>(std::round(n_elec_total));
  if (n_elec % 2 != 0) {
    throw std::runtime_error(
        "Gfn2Calculator: open-shell n_elec=" + std::to_string(n_elec));
  }
  const int n_occ = n_elec / 2;

  // For SCC-D4, set up the geometry-cached state once. We then re-evaluate
  // dispersion every SCC iteration with the current charges (matches xtb's
  // self-consistent D4 to within a few µHa). D4SccState owns dftd4's
  // TMatrix instances which lack proper copy/move, so we hold it via
  // unique_ptr to keep it pinned in memory.
  std::optional<occ::disp::Dispersion> native_d4;
  if (opts.include_dispersion) {
    native_d4.emplace(m_atoms);
    const auto &g = m_params.globals();
    native_d4->set_damping(
        occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
  }
  double e_disp = 0.0;

  Vec qsh = Vec::Zero(m_n_shells);
  double prev_energy = 0.0;
  Vec orbital_energies, orbital_occupations;
  Mat C, P;

  // Pulay-style charge DIIS: extrapolate qsh_new from history of
  // (qsh_new_i, residual_i = qsh_new_i − qsh_in_i). Linear damping warms it
  // up for the first `diis_start` iterations.
  const std::size_t diis_start = 3;
  const std::size_t diis_subspace = 8;
  occ::core::diis::DIIS diis(diis_start, diis_subspace);

  occ::log::info("{:=^72s}", "  GFN2-xTB self-consistent charges  ");
  occ::log::info("nbf = {}   n_shells = {}   n_electrons = {}   multipoles = {}",
                 m_nbf, m_n_shells, n_elec, include_multipoles ? "on" : "off");
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                 "|ΔE|", "max|Δq|");

  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= opts.max_iterations; ++iter) {
    // Isotropic + third-order shell potential.
    Vec V = m_J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += m_shells.third_order(s) * qsh(s) * qsh(s);
    }

    // Start with H = H0 - 0.5 * S * (V_iso_μ + V_iso_ν).
    Mat H = m_H0;
    for (Eigen::Index mu = 0; mu < m_nbf; ++mu) {
      const int sh_mu = m_bf_to_shell[mu];
      for (Eigen::Index nu = 0; nu < m_nbf; ++nu) {
        const int sh_nu = m_bf_to_shell[nu];
        H(mu, nu) -= 0.5 * m_S(mu, nu) * (V(sh_mu) + V(sh_nu));
      }
    }

    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (include_multipoles && iter > 1) {
      auto m = compute_camm_moments(m_atoms, m_bf_to_atom, P, m_S, m_D_ao,
                                    m_Q_ao);
      Vec atom_q = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q(m_shells.atom[s]) += qsh(s);
      auto pot = anisotropic_potentials(m_atoms, atom_q, m, m_damped, m_params);
      apply_anisotropic_h1(H, m_S, m_D_ao, m_Q_ao, m_bf_to_atom, pot);
      e_aniso = anisotropic_energy(m_atoms, atom_q, m, m_damped, m_params);
    }

    Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(H, m_S);
    if (es.info() != Eigen::Success) {
      throw std::runtime_error("Gfn2Calculator: eigensolver failed");
    }
    orbital_energies = es.eigenvalues();
    C = es.eigenvectors();

    orbital_occupations = Vec::Zero(m_nbf);
    for (int i = 0; i < n_occ; ++i)
      orbital_occupations(i) = 2.0;

    Mat Cocc = C.leftCols(n_occ);
    P = 2.0 * (Cocc * Cocc.transpose());

    Mat PS = P * m_S;
    Vec pop = shell_populations(PS, m_bf_to_shell, m_n_shells);
    Vec qsh_new = m_z_sh - pop;

    // Compute SCC-coupled D4 with the current Mulliken charges via the
    // native occ::disp::Dispersion (uses GFN2-xTB reference data).
    if (native_d4) {
      Vec atom_q_new = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q_new(m_shells.atom[s]) += qsh_new(s);
      native_d4->set_charges(atom_q_new);
      e_disp = native_d4->energy();
    }

    double e_es = 0.5 * qsh_new.dot(m_J * qsh_new);
    double e_third = 0.0;
    for (Eigen::Index s = 0; s < qsh_new.size(); ++s) {
      const double q = qsh_new(s);
      e_third += m_shells.third_order(s) * q * q * q / 3.0;
    }
    double e_h0 = (P.cwiseProduct(m_H0)).sum();
    double scc_energy =
        e_h0 + e_es + e_third + e_aniso.aes + e_aniso.polariz;
    double total_energy = scc_energy + m_e_rep + e_disp;

    double dq_max = (qsh_new - qsh).cwiseAbs().maxCoeff();
    double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                   total_energy, de, dq_max);

    bool e_ok = (iter > 1) && de < opts.energy_threshold;
    bool q_ok = dq_max < opts.charge_threshold;
    if (e_ok && q_ok) {
      converged = true;
      Vec atom_charges = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_charges(m_shells.atom[s]) += qsh_new(s);
      SccResult r;
      r.scc_energy = scc_energy;
      r.repulsion_energy = m_e_rep;
      r.dispersion_energy = e_disp;
      r.total_energy = total_energy;
      r.shell_charges = qsh_new;
      r.atomic_charges = atom_charges;
      r.orbital_energies = orbital_energies;
      r.orbital_occupations = orbital_occupations;
      r.density_matrix = P;
      r.overlap_matrix = m_S;
      r.orbital_coefficients = C;
      r.n_iterations = iter;
      r.converged = true;
      occ::log::info("Converged in {} iterations.", iter);
      return r;
    }

    // Push (qsh_new, residual) into DIIS. For iter ≤ diis_start the call is
    // a no-op (just builds history) and we fall back to linear damping; once
    // there is enough history DIIS overwrites x with the extrapolated qsh.
    Mat x = qsh_new;
    Mat err = qsh_new - qsh;
    diis.extrapolate(x, err);
    if (static_cast<std::size_t>(iter) > diis_start) {
      qsh = x.col(0);
    } else {
      qsh = (1.0 - opts.damping_factor) * qsh_new + opts.damping_factor * qsh;
    }
    prev_energy = total_energy;
  }

  occ::log::warn("GFN2 SCC did not converge in {} iterations",
                 opts.max_iterations);
  // Unconverged — return last iterate.
  SccResult r;
  r.scc_energy = prev_energy - m_e_rep - e_disp;
  r.repulsion_energy = m_e_rep;
  r.dispersion_energy = e_disp;
  r.total_energy = prev_energy;
  r.shell_charges = qsh;
  Vec atom_charges = Vec::Zero(m_atoms.size());
  for (int s = 0; s < m_n_shells; ++s)
    atom_charges(m_shells.atom[s]) += qsh(s);
  r.atomic_charges = atom_charges;
  r.orbital_energies = orbital_energies;
  r.orbital_occupations = orbital_occupations;
  r.density_matrix = P;
  r.overlap_matrix = m_S;
  r.orbital_coefficients = C;
  r.n_iterations = iter;
  r.converged = false;
  return r;
}

} // namespace occ::xtb
