#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/log.h>
#include <occ/disp/dftd4.h>
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

// dftd4 lower-level pieces (used for the SCC-coupled D4 path).
#include "dftd_cutoff.h"
#include "dftd_dispersion.h"
#include "dftd_geometry.h"
#include "dftd_model.h"
#include "dftd_ncoord.h"
#include "damping/dftd_rational.h"

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

// EEQ-charge D4: uses dftd4's get_dispersion path which calls EEQ internally.
// Geometry-only result for the given functional/method parameters.
double compute_d4_energy_eeq(const std::vector<core::Atom> &atoms,
                             const GlobalParam &g, double total_charge) {
  occ::disp::D4Dispersion d4(atoms);
  dftd4::dparam dp{g.s6, g.s8, /*s10=*/0.0, g.s9, g.a1, g.a2, /*alp=*/16};
  d4.set_parameters(dp);
  d4.set_charge(static_cast<int>(std::round(total_charge)));
  return d4.energy();
}

// Persistent state for SCC-coupled D4 — computed once per geometry, reused
// across SCC iterations. dftd4's TMatrix has no proper move/copy semantics
// (it owns a raw pointer), so this struct must be heap-allocated and never
// copied.
struct D4SccState {
  dftd4::TMolecule mol;
  dftd4::TIVector real_idx;
  dftd4::TCutoff cutoff;
  dftd4::TD4Model d4{};
  dftd4::TMatrix<double> dist;
  dftd4::TVector<double> cn;
  dftd4::TMatrix<double> refq;
  int mref{0};
  int nat{0};

  D4SccState() = default;
  D4SccState(const D4SccState &) = delete;
  D4SccState &operator=(const D4SccState &) = delete;
  D4SccState(D4SccState &&) = delete;
  D4SccState &operator=(D4SccState &&) = delete;
};

void build_d4_scc_state(D4SccState &s,
                        const std::vector<core::Atom> &atoms) {
  s.nat = static_cast<int>(atoms.size());
  s.mol.GetMemory(s.nat);
  for (int i = 0; i < s.nat; ++i) {
    s.mol.CC(i, 0) = atoms[i].x;
    s.mol.CC(i, 1) = atoms[i].y;
    s.mol.CC(i, 2) = atoms[i].z;
    s.mol.ATNO(i) = atoms[i].atomic_number;
  }
  dftd4::initializeRealIdx(s.nat, s.real_idx);

  s.dist.NewMatrix(s.nat, s.nat);
  dftd4::calc_distances(s.mol, s.real_idx, s.dist);

  s.cn.NewVector(s.nat);
  dftd4::TMatrix<double> dcndr; // unused, gradient-off
  dftd4::get_ncoord_d4(s.mol, s.real_idx, s.dist, s.cutoff.cn, s.cn, dcndr,
                       false);

  dftd4::get_max_ref(s.mol, s.mref);
  s.refq.NewMat(s.mref, s.nat);
  s.d4.set_refq_eeq(s.mol, s.real_idx, s.refq);
}

// SCC-charge D4: drives weight_references with the supplied SCC charges.
// `q_atomic` follows xtb's convention (positive when electron-deficient).
double compute_d4_energy_scc(const D4SccState &s, const GlobalParam &g,
                             const Vec &q_atomic) {
  dftd4::TVector<double> q;
  q.NewVector(s.nat);
  for (int i = 0; i < s.nat; ++i)
    q(i) = q_atomic(i);

  dftd4::TMatrix<double> gwvec, dgwdcn, dgwdq;
  gwvec.NewMatrix(s.mref, s.nat);
  // Allocate dgwdcn/dgwdq defensively even though lgrad=false; some dftd4
  // builds touch them unconditionally.
  dgwdcn.NewMatrix(s.mref, s.nat);
  dgwdq.NewMatrix(s.mref, s.nat);
  s.d4.weight_references(s.mol, s.real_idx, s.cn, q, s.refq, gwvec, dgwdcn,
                         dgwdq, false);

  dftd4::TMatrix<double> c6, dc6dcn, dc6dq;
  c6.NewMatrix(s.nat, s.nat);
  dc6dcn.NewMatrix(s.nat, s.nat);
  dc6dq.NewMatrix(s.nat, s.nat);
  s.d4.get_atomic_c6(s.mol, s.real_idx, gwvec, dgwdcn, dgwdq, c6, dc6dcn,
                     dc6dq, false);

  dftd4::dparam par{g.s6, g.s8, /*s10=*/0.0, g.s9, g.a1, g.a2, /*alp=*/16};

  dftd4::TVector<double> e2body, dEdcn, dEdq, gradient;
  e2body.NewVector(s.nat);
  dEdcn.NewVector(s.nat);
  dEdq.NewVector(s.nat);
  gradient.NewVector(3 * s.nat);
  dftd4::get_dispersion2(s.mol, s.real_idx, s.dist, s.cutoff.disp2, par, c6,
                         dc6dcn, dc6dq, e2body, dEdcn, dEdq, gradient, false);

  dftd4::TVector<double> e3body;
  e3body.NewVector(s.nat);
  if (par.s9 != 0.0) {
    dftd4::get_dispersion3(s.mol, s.real_idx, s.dist, s.cutoff.disp3, par, c6,
                           dc6dcn, dc6dq, e3body, dEdcn, dEdq, gradient, false);
  }

  double total = 0.0;
  for (int i = 0; i < s.nat; ++i)
    total += e2body(i) + e3body(i);
  return total;
}

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
  std::unique_ptr<D4SccState> d4_state;
  if (opts.include_dispersion) {
    d4_state = std::make_unique<D4SccState>();
    build_d4_scc_state(*d4_state, m_atoms);
  }
  double e_disp = 0.0;

  Vec qsh = Vec::Zero(m_n_shells);
  double prev_energy = 0.0;
  Vec orbital_energies, orbital_occupations;
  Mat C, P;

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

    // Compute SCC-coupled D4 with the current Mulliken charges.
    if (d4_state) {
      Vec atom_q_new = Vec::Zero(m_atoms.size());
      for (int s = 0; s < m_n_shells; ++s)
        atom_q_new(m_shells.atom[s]) += qsh_new(s);
      e_disp = compute_d4_energy_scc(*d4_state, m_params.globals(), atom_q_new);
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

    qsh = (1.0 - opts.damping_factor) * qsh_new + opts.damping_factor * qsh;
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
