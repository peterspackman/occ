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

} // namespace occ::xtb
