#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/disp/d4.h>
#include <optional>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/gfn2_engine.h>
#include <occ/xtb/h0.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/occupation.h>
#include <occ/xtb/repulsion.h>
#include <occ/xtb/scc_state.h>
#include <occ/xtb/spin.h>
#include <memory>
#include <stdexcept>


namespace occ::xtb {

Gfn2Engine::Gfn2Engine(std::vector<core::Atom> atoms,
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

void Gfn2Engine::update_positions(const std::vector<core::Atom> &atoms) {
  if (atoms.size() != m_atoms.size()) {
    throw std::runtime_error(
        "Gfn2Engine::update_positions: atom count changed (" +
        std::to_string(atoms.size()) + " vs " + std::to_string(m_atoms.size()) +
        ")");
  }
  // Verify atomic numbers are unchanged — basis would otherwise be invalid.
  for (size_t i = 0; i < atoms.size(); ++i) {
    if (atoms[i].atomic_number != m_atoms[i].atomic_number) {
      throw std::runtime_error(
          "Gfn2Engine::update_positions: atomic number of atom " +
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

void Gfn2Engine::recompute_geometry_caches() {
  m_S = m_engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  m_cn = gfn_coordination_numbers(m_atoms);
  m_e_rep = ::occ::xtb::repulsion_energy(m_atoms, m_params);
  m_J = klopman_ohno_gamma(m_atoms, m_shells, m_params);
  m_H0 = build_h0(m_atoms, m_params, m_shells, m_basis, m_S, m_cn);
  m_mp_radii = multipole_radii(m_atoms, m_cn, m_params);
  m_have_multipole_ints = false; // built on demand
}

const Mat &Gfn2Engine::spin_coupling(double scale) const {
  // W depends only on the elements and their shell angular momenta, both of
  // which update_positions() guarantees are unchanged — so the cache survives
  // geometry updates and only has to be rebuilt when the scale changes.
  if (!m_have_spin_coupling || m_spin_coupling_scale != scale) {
    m_spin_coupling =
        spin_coupling_matrix(m_atoms, m_shells, m_params, scale);
    m_spin_coupling_scale = scale;
    m_have_spin_coupling = true;
  }
  return m_spin_coupling;
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

// Σ_i n_i c_i c_iᵀ over the occupied (and, with smearing, fractionally
// occupied) orbitals.
Mat density_from_occupations(const Mat &C, const Vec &occupations) {
  Mat P = Mat::Zero(C.rows(), C.rows());
  for (Eigen::Index i = 0; i < occupations.size(); ++i) {
    if (occupations(i) <= 1e-12)
      continue;
    P.noalias() += occupations(i) * C.col(i) * C.col(i).transpose();
  }
  return P;
}

// Accumulate a per-atom quantity from its per-shell parts.
Vec shell_to_atom(const Vec &per_shell, const std::vector<int> &shell_atom,
                  int n_atoms) {
  Vec per_atom = Vec::Zero(n_atoms);
  for (Eigen::Index s = 0; s < per_shell.size(); ++s)
    per_atom(shell_atom[s]) += per_shell(s);
  return per_atom;
}

struct ElectronConfiguration {
  int n_electrons{0};
  int n_unpaired{0};
  bool unrestricted{false};
  AlphaBetaOccupation occupation;
};

ElectronConfiguration resolve_configuration(double n_electrons_reference,
                                            const SccOptions &opts) {
  const double total = n_electrons_reference - opts.total_charge;
  if (std::abs(std::round(total) - total) > 1e-6) {
    throw std::runtime_error(
        "Gfn2Engine: non-integer electron count not supported");
  }
  ElectronConfiguration cfg;
  cfg.n_electrons = static_cast<int>(std::round(total));
  cfg.n_unpaired = std::abs(opts.unpaired_electrons);
  cfg.unrestricted = cfg.n_unpaired != 0 || opts.force_unrestricted;
  if (cfg.n_unpaired > cfg.n_electrons) {
    throw std::runtime_error(
        "Gfn2Engine: " + std::to_string(cfg.n_unpaired) +
        " unpaired electrons requested but only " +
        std::to_string(cfg.n_electrons) + " valence electrons are available");
  }
  if ((cfg.n_electrons - cfg.n_unpaired) % 2 != 0) {
    throw std::runtime_error(
        "Gfn2Engine: electron count (" + std::to_string(cfg.n_electrons) +
        ") and unpaired-electron count (" + std::to_string(cfg.n_unpaired) +
        ") have different parity — check the charge / multiplicity");
  }
  cfg.occupation = alpha_beta_occupation(cfg.n_electrons, cfg.n_unpaired);
  return cfg;
}

} // namespace

SccResult Gfn2Engine::single_point(const SccOptions &opts,
                                       bool include_multipoles) {
  // Initialise the (optional) implicit-solvent model at the current geometry.
  // Models are re-initialised on every SCC so the same instance can outlive
  // geometry updates from XtbCalculator::update_structure.
  if (m_solvation) {
    Mat3N positions(3, m_atoms.size());
    IVec atomic_numbers(m_atoms.size());
    for (size_t a = 0; a < m_atoms.size(); ++a) {
      positions(0, a) = m_atoms[a].x;
      positions(1, a) = m_atoms[a].y;
      positions(2, a) = m_atoms[a].z;
      atomic_numbers(a) = m_atoms[a].atomic_number;
    }
    m_solvation->initialize(positions, atomic_numbers);
  }

  // Build atom-centered Bra/Ket AO multipole matrices and the molecular
  // multipole pair tensors (sd/dd/sq) on first multipole-enabled call —
  // shares its code path with the periodic SCC. Reuse the existing
  // m_engine/m_S instead of going through build_molecular_multipole_ao
  // (which would build a fresh basis+engine+S).
  if (include_multipoles && !m_have_multipole_ints) {
    MatTriple D0 = dipole_ao_matrices(m_engine);
    std::array<Mat, 6> Q0 = quadrupole_ao_matrices(m_engine);
    m_mp_ao = center_multipole_ao(m_atoms, m_bf_to_atom, m_S, D0, Q0);
    m_mp_tensors = build_molecular_multipole_tensors(m_atoms, m_mp_radii,
                                                     m_params);
    m_have_multipole_ints = true;
  }

  const auto cfg = resolve_configuration(m_z_sh.sum(), opts);
  const int n_elec = cfg.n_electrons;
  const int n_unpaired = cfg.n_unpaired;
  const bool unrestricted = cfg.unrestricted;
  const auto &occupation = cfg.occupation;

  // kB·T in Hartree. 300 K ≈ 0.95 mHa, which leaves occupations integral for
  // anything with a normal gap.
  const double kt = opts.electronic_temperature > 0.0
                        ? opts.electronic_temperature / occ::units::AU_TO_KELVIN
                        : 0.0;
  const Mat W = unrestricted ? spin_coupling(opts.spin_polarization) : Mat();

  // Re-evaluated each SCC iteration at the current charges, matching xtb's
  // self-consistent D4 to within a few µHa.
  std::optional<occ::disp::D4Dispersion> native_d4;
  if (opts.include_dispersion) {
    native_d4.emplace(m_atoms);
    const auto &g = m_params.globals();
    native_d4->set_damping(
        occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
  }
  double e_disp = 0.0;

  // Input state for the next Hamiltonian build. Magnetization starts at zero,
  // so the first iteration's α/β Hamiltonians are identical and the spin
  // density comes purely from the differing α/β electron counts.
  const int n_atoms = static_cast<int>(m_atoms.size());
  SccMixerState state = SccMixerState::zero(m_n_shells, n_atoms, unrestricted,
                                            include_multipoles);

  // Initial charges: caller-supplied warm start (a nearby geometry's converged
  // qsh, from geometry optimization or Hessian FD), else EEQ, else zeros for
  // elements EEQ lacks parameters for. Consumed once so it can't leak into a
  // later unrelated call.
  if (m_qsh_init.size() == m_n_shells) {
    state.shell_charges = m_qsh_init;
    m_qsh_init = Vec();
  } else {
    try {
      state.shell_charges =
          eeq_initial_shell_charges(m_atoms, m_shells, opts.total_charge);
    } catch (const std::exception &) {
      state.shell_charges = Vec::Zero(m_n_shells);
    }
  }

  double prev_energy = 0.0;
  Vec orbital_energies, orbital_occupations;
  Vec orbital_energies_beta, orbital_occupations_beta;
  Mat C, C_beta, P, P_alpha, P_beta;

  // Pulay-style DIIS on the whole input state, with linear damping for the
  // first `diis_start` iterations while history builds.
  const std::size_t diis_start = 3;
  const std::size_t diis_subspace = 8;
  occ::core::diis::DIIS diis(diis_start, diis_subspace);

  occ::log::info("{:=^72s}", "  GFN2-xTB self-consistent charges  ");
  occ::log::info("nbf = {}   n_shells = {}   n_electrons = {}   multipoles = {}",
                 m_nbf, m_n_shells, n_elec, include_multipoles ? "on" : "off");
  if (unrestricted) {
    occ::log::info("spin           : unrestricted, Nα = {:g}  Nβ = {:g}  "
                   "(2S+1 = {})  W scale = {:g}",
                   occupation.n_alpha, occupation.n_beta, n_unpaired + 1,
                   opts.spin_polarization);
  }
  if (m_solvation) {
    occ::log::info("solvation: {}", m_solvation->name());
  }
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                 "|ΔE|", "max|Δq|");

  if (opts.max_iterations < 1) {
    throw std::runtime_error("Gfn2Engine: max_iterations must be at least 1");
  }

  // Rebuilt from scratch each cycle; returned on convergence or, unchanged
  // from the final cycle, when the iteration limit is hit.
  SccResult result;
  for (int iter = 1; iter <= opts.max_iterations; ++iter) {
    const Vec &qsh = state.shell_charges;
    const Vec atom_q = shell_to_atom(qsh, m_shells.atom, n_atoms);
    if (m_solvation) {
      m_solvation->update(atom_q);
    }

    // Isotropic + third-order shell potential.
    Vec V = m_J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += m_shells.third_order(s) * qsh(s) * qsh(s);
    }
    // Each AO picks the atom-resolved solvation shift up via the
    // 0.5·S·(V_μ + V_ν) term that builds H.
    if (m_solvation) {
      const Vec &v_solv = m_solvation->atom_potential();
      for (Eigen::Index s = 0; s < V.size(); ++s) {
        V(s) += v_solv(m_shells.atom[s]);
      }
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

    if (include_multipoles) {
      auto pot = anisotropic_potentials_ewald(
          m_atoms, atom_q, state.multipoles, m_mp_tensors, m_params);
      apply_anisotropic_h1_periodic(H, m_S, m_mp_ao.D_ket, m_mp_ao.D_bra,
                                     m_mp_ao.Q_ket, m_mp_ao.Q_bra,
                                     m_bf_to_atom, pot);
    }

    auto solve = [&](const Mat &fock, Vec &energies, Mat &coeffs) {
      Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(fock, m_S);
      if (es.info() != Eigen::Success) {
        throw std::runtime_error("Gfn2Engine: eigensolver failed");
      }
      energies = es.eigenvalues();
      coeffs = es.eigenvectors();
    };

    SccMixerState fresh = SccMixerState::zero(m_n_shells, n_atoms,
                                              unrestricted,
                                              include_multipoles);
    double e_entropy = 0.0;
    if (unrestricted) {
      // H_σ = H_common ± ½ S_μν (v_μ + v_ν) with v = W·m; + for α, − for β.
      // W is negative, so a positive magnetization lowers the α levels.
      const Vec v_spin = W * state.magnetization;
      Mat H_alpha = H;
      Mat H_beta = H;
      for (Eigen::Index mu = 0; mu < m_nbf; ++mu) {
        const int sh_mu = m_bf_to_shell[mu];
        for (Eigen::Index nu = 0; nu < m_nbf; ++nu) {
          const int sh_nu = m_bf_to_shell[nu];
          const double shift =
              0.5 * m_S(mu, nu) * (v_spin(sh_mu) + v_spin(sh_nu));
          H_alpha(mu, nu) += shift;
          H_beta(mu, nu) -= shift;
        }
      }
      solve(H_alpha, orbital_energies, C);
      solve(H_beta, orbital_energies_beta, C_beta);

      const auto fill_a = fermi_filling(occupation.n_alpha, kt,
                                        orbital_energies);
      const auto fill_b = fermi_filling(occupation.n_beta, kt,
                                        orbital_energies_beta);
      orbital_occupations = fill_a.occupations;
      orbital_occupations_beta = fill_b.occupations;
      e_entropy = fill_a.entropy_energy + fill_b.entropy_energy;

      P_alpha = density_from_occupations(C, orbital_occupations);
      P_beta = density_from_occupations(C_beta, orbital_occupations_beta);
      P = P_alpha + P_beta;

      fresh.magnetization =
          shell_populations(P_alpha * m_S, m_bf_to_shell, m_n_shells) -
          shell_populations(P_beta * m_S, m_bf_to_shell, m_n_shells);
    } else {
      solve(H, orbital_energies, C);
      // Both channels fill from the same orbitals, giving the usual 2.0 per
      // occupied orbital for an even electron count with a gap.
      const auto fill_a = fermi_filling(occupation.n_alpha, kt,
                                        orbital_energies);
      const auto fill_b = fermi_filling(occupation.n_beta, kt,
                                        orbital_energies);
      orbital_occupations = fill_a.occupations + fill_b.occupations;
      e_entropy = fill_a.entropy_energy + fill_b.entropy_energy;
      P = density_from_occupations(C, orbital_occupations);
    }

    fresh.shell_charges =
        m_z_sh - shell_populations(P * m_S, m_bf_to_shell, m_n_shells);
    const Vec &qsh_new = fresh.shell_charges;
    const Vec atom_q_new = shell_to_atom(qsh_new, m_shells.atom, n_atoms);
    const double e_spin =
        unrestricted
            ? 0.5 * fresh.magnetization.dot(W * fresh.magnetization)
            : 0.0;

    // AES from the post-density CAMM: the energy reflects the just-solved
    // (P, q, μ), not the H1's input state.
    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (include_multipoles) {
      fresh.multipoles = compute_camm_moments_periodic(
          m_atoms, m_bf_to_atom, P, m_mp_ao.D_ket, m_mp_ao.D_bra,
          m_mp_ao.Q_ket, m_mp_ao.Q_bra);
      const CammMoments &m_new = fresh.multipoles;
      e_aniso = anisotropic_energy_ewald(m_atoms, atom_q_new, m_new,
                                          m_mp_tensors, m_params);
      for (int a = 0; a < static_cast<int>(m_atoms.size()); ++a) {
        occ::log::debug(
            "    atom {:3d} (Z={:>2d})  q={:+.6f}  d=({:+.6f}, {:+.6f}, "
            "{:+.6f})  qp_xx={:+.6f} yy={:+.6f} zz={:+.6f} xy={:+.6f} "
            "xz={:+.6f} yz={:+.6f}",
            a + 1, m_atoms[a].atomic_number, atom_q_new(a),
            m_new.dipm(0, a), m_new.dipm(1, a), m_new.dipm(2, a),
            m_new.qp(0, a), m_new.qp(2, a), m_new.qp(5, a),
            m_new.qp(1, a), m_new.qp(3, a), m_new.qp(4, a));
      }
    }

    if (native_d4) {
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
    double e_solv = m_solvation ? m_solvation->energy() : 0.0;
    double scc_energy = e_h0 + e_es + e_third + e_aniso.aes +
                        e_aniso.polariz + e_solv + e_spin + e_entropy;
    double total_energy = scc_energy + m_e_rep + e_disp;

    const double dq_max = fresh.max_change(state);
    const double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                   total_energy, de, dq_max);
    occ::log::debug(
        "    breakdown: H0={:>14.6f}  ES={:>14.6f}  3rd={:>10.3e}  "
        "AES={:>10.3e}  pol={:>10.3e}  solv={:>10.3e}  spin={:>10.3e}  "
        "-TS={:>10.3e}  rep={:>10.3e}  disp={:>10.3e}",
        e_h0, e_es, e_third, e_aniso.aes, e_aniso.polariz, e_solv, e_spin,
        e_entropy, m_e_rep, e_disp);

    bool e_ok = (iter > 1) && de < opts.energy_threshold;
    bool q_ok = dq_max < opts.charge_threshold;
    const bool converged = e_ok && q_ok;

    // Snapshotted every cycle, so an unconverged return still carries a
    // coherent (energy, density, charges) triple from the last one.
    if (converged && m_solvation) {
      // Report the per-element decomposition at the same q as the energy.
      m_solvation->update(atom_q_new);
    }
    result.scc_energy = scc_energy;
    result.repulsion_energy = m_e_rep;
    result.dispersion_energy = e_disp;
    result.total_energy = total_energy;
    result.shell_charges = qsh_new;
    result.atomic_charges = atom_q_new;
    result.orbital_energies = orbital_energies;
    result.orbital_occupations = orbital_occupations;
    result.density_matrix = P;
    result.overlap_matrix = m_S;
    result.orbital_coefficients = C;
    result.n_iterations = iter;
    result.converged = converged;
    result.unrestricted = unrestricted;
    result.num_unpaired_electrons = n_unpaired;
    result.spin_energy = e_spin;
    result.electronic_entropy_energy = e_entropy;
    if (unrestricted) {
      result.shell_magnetization = fresh.magnetization;
      result.atomic_magnetization =
          shell_to_atom(fresh.magnetization, m_shells.atom, n_atoms);
      result.density_matrix_alpha = P_alpha;
      result.density_matrix_beta = P_beta;
      result.orbital_coefficients_beta = C_beta;
      result.orbital_energies_beta = orbital_energies_beta;
      result.orbital_occupations_beta = orbital_occupations_beta;
    }
    if (m_solvation) {
      result.solvation_surfaces = m_solvation->surfaces();
    }

    if (converged) {
      m_last_shell_charges = qsh_new;
      occ::log::info("Converged in {} iterations.", iter);
      return result;
    }

    // Extrapolate the whole input state. Below `diis_start` the call only
    // accumulates history and we fall back to linear damping.
    Mat x = fresh.pack();
    Mat err = x - state.pack();
    diis.extrapolate(x, err);
    if (static_cast<std::size_t>(iter) > diis_start) {
      state.unpack(x.col(0));
    } else {
      state.damp_toward(fresh, opts.damping_factor);
    }
    prev_energy = total_energy;
  }

  occ::log::warn("GFN2 SCC did not converge in {} iterations",
                 opts.max_iterations);
  result.n_iterations = opts.max_iterations;
  result.converged = false;
  return result;
}

} // namespace occ::xtb
