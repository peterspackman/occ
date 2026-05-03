#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/xtb/gfn2_calculator.h>
#include <occ/xtb/native_calculator.h>
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

NativeCalculator::~NativeCalculator() = default;

void NativeCalculator::initialize_calculator() {
  m_params = std::make_shared<Gfn2Parameters>(Gfn2Parameters::load_default());
  m_calc = std::make_unique<Gfn2Calculator>(
      make_atoms(m_positions_bohr, m_atomic_numbers), *m_params);
  m_opts.total_charge = m_charge;
}

double NativeCalculator::single_point_energy() {
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
