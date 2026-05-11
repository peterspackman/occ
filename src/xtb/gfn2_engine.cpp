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
#include <occ/xtb/gfn2_engine.h>
#include <occ/xtb/h0.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/repulsion.h>
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

SccResult Gfn2Engine::single_point(const SccOptions &opts,
                                       bool include_multipoles) {
  if (opts.unpaired_electrons != 0) {
    throw std::runtime_error(
        "Gfn2Engine: open-shell case not yet supported");
  }

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
  // multipole pair tensors (sd/dd/sq) on first multipole-enabled call.
  // tblite convention end-to-end — the same code path as the periodic SCC.
  // Reuse the existing m_engine/m_S instead of going through
  // build_molecular_multipole_ao (which would build a fresh basis+engine+S).
  if (include_multipoles && !m_have_multipole_ints) {
    MatTriple D0 = dipole_ao_matrices(m_engine);
    std::array<Mat, 6> Q0 = quadrupole_ao_matrices(m_engine);
    m_mp_ao = center_multipole_ao(m_atoms, m_bf_to_atom, m_S, D0, Q0);
    m_mp_tensors = build_molecular_multipole_tensors(m_atoms, m_mp_radii,
                                                     m_params);
    m_have_multipole_ints = true;
  }

  // Closed-shell electron count.
  double n_elec_total = 0.0;
  for (Eigen::Index i = 0; i < m_z_sh.size(); ++i)
    n_elec_total += m_z_sh(i);
  n_elec_total -= opts.total_charge;
  if (std::abs(std::round(n_elec_total) - n_elec_total) > 1e-6) {
    throw std::runtime_error(
        "Gfn2Engine: non-integer electron count not supported");
  }
  const int n_elec = static_cast<int>(std::round(n_elec_total));
  if (n_elec % 2 != 0) {
    throw std::runtime_error(
        "Gfn2Engine: open-shell n_elec=" + std::to_string(n_elec));
  }
  const int n_occ = n_elec / 2;

  // For SCC-D4, set up the geometry-cached state once. We then re-evaluate
  // dispersion every SCC iteration with the current charges (matches xtb's
  // self-consistent D4 to within a few µHa). D4SccState owns dftd4's
  // TMatrix instances which lack proper copy/move, so we hold it via
  // unique_ptr to keep it pinned in memory.
  std::optional<occ::disp::D4Dispersion> native_d4;
  if (opts.include_dispersion) {
    native_d4.emplace(m_atoms);
    const auto &g = m_params.globals();
    native_d4->set_damping(
        occ::disp::D4Damping{g.s6, g.s8, g.s9, g.a1, g.a2, 16});
  }
  double e_disp = 0.0;

  // SCC initial guess. Priority:
  //   1. Caller-supplied warm-start `m_qsh_init` if its length matches
  //      n_shells (typically the previous gradient/opt step's converged
  //      qsh on a nearby geometry — saves several SCC iterations).
  //   2. EEQ charges (xtb convention).
  //   3. Zeros (if EEQ doesn't have parameters for one of the elements).
  // Cleared after consuming so a stale warm-start can't silently leak
  // into a subsequent unrelated call.
  Vec qsh;
  if (m_qsh_init.size() == m_n_shells) {
    qsh = m_qsh_init;
    m_qsh_init = Vec();
  } else {
    try {
      qsh = eeq_initial_shell_charges(m_atoms, m_shells, opts.total_charge);
    } catch (const std::exception &) {
      qsh = Vec::Zero(m_n_shells);
    }
  }
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
  if (m_solvation) {
    occ::log::info("solvation: {}", m_solvation->name());
  }
  occ::log::info("{:>4s}  {:>20s}  {:>12s}  {:>12s}", "iter", "E (Hartree)",
                 "|ΔE|", "max|Δq|");

  bool converged = false;
  int iter = 0;
  for (iter = 1; iter <= opts.max_iterations; ++iter) {
    // Solvation update at the current input shell charges. Projection
    // shell → atom matches the convention used by the AES H1 below.
    if (m_solvation) {
      Vec atom_q = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q(m_shells.atom[s]) += qsh(s);
      m_solvation->update(atom_q);
    }

    // Isotropic + third-order shell potential.
    Vec V = m_J * qsh;
    for (Eigen::Index s = 0; s < V.size(); ++s) {
      V(s) += m_shells.third_order(s) * qsh(s) * qsh(s);
    }
    // Fold the atom-resolved solvation shift into each shell's V (each AO
    // ultimately picks it up via the 0.5·S·(V_μ + V_ν) term that builds H).
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

    if (include_multipoles && iter > 1) {
      // H1 uses CAMM from the previous iteration's density (current `P` at
      // entry to this iter). Energy is computed below from the new P, the
      // new qsh, and the new CAMM (matches tblite's iterator.f90).
      auto m_in = compute_camm_moments_periodic(m_atoms, m_bf_to_atom, P,
                                                  m_mp_ao.D_ket, m_mp_ao.D_bra,
                                                  m_mp_ao.Q_ket, m_mp_ao.Q_bra);
      Vec atom_q = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q(m_shells.atom[s]) += qsh(s);
      auto pot = anisotropic_potentials_ewald(m_atoms, atom_q, m_in,
                                                m_mp_tensors, m_params);
      apply_anisotropic_h1_periodic(H, m_S, m_mp_ao.D_ket, m_mp_ao.D_bra,
                                     m_mp_ao.Q_ket, m_mp_ao.Q_bra,
                                     m_bf_to_atom, pot);
    }

    Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(H, m_S);
    if (es.info() != Eigen::Success) {
      throw std::runtime_error("Gfn2Engine: eigensolver failed");
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

    // Multipole AES from the post-density CAMM (matches tblite — energy
    // reflects the just-solved (P, q, μ) triple, not the H1's input state).
    AnisotropicEnergy e_aniso{0.0, 0.0};
    if (include_multipoles) {
      auto m_new = compute_camm_moments_periodic(m_atoms, m_bf_to_atom, P,
                                                   m_mp_ao.D_ket, m_mp_ao.D_bra,
                                                   m_mp_ao.Q_ket, m_mp_ao.Q_bra);
      Vec atom_q_new = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q_new(m_shells.atom[s]) += qsh_new(s);
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

    // Compute SCC-coupled D4 with the current Mulliken charges via the
    // native occ::disp::D4Dispersion (uses GFN2-xTB reference data).
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
    double e_solv = m_solvation ? m_solvation->energy() : 0.0;
    double scc_energy =
        e_h0 + e_es + e_third + e_aniso.aes + e_aniso.polariz + e_solv;
    double total_energy = scc_energy + m_e_rep + e_disp;

    double dq_max = (qsh_new - qsh).cwiseAbs().maxCoeff();
    double de = std::abs(total_energy - prev_energy);
    occ::log::info("{:>4d}  {:>20.12f}  {:>12.2e}  {:>12.2e}", iter,
                   total_energy, de, dq_max);
    occ::log::debug(
        "    breakdown: H0={:>14.6f}  ES={:>14.6f}  3rd={:>10.3e}  "
        "AES={:>10.3e}  pol={:>10.3e}  solv={:>10.3e}  rep={:>10.3e}  "
        "disp={:>10.3e}",
        e_h0, e_es, e_third, e_aniso.aes, e_aniso.polariz, e_solv, m_e_rep,
        e_disp);

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
      m_last_shell_charges = qsh_new;
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
