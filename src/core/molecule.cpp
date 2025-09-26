#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <occ/core/constants.h>
#include <occ/core/inertia_tensor.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <stdexcept>

namespace occ::core {

Molecule::Molecule(const IVec &nums, const Mat3N &pos)
    : m_atomicNumbers(nums), m_positions(pos),
      m_partial_charges(Vec::Zero(nums.rows())) {
  for (size_t i = 0; i < size(); i++) {
    m_elements.push_back(Element(m_atomicNumbers(i)));
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const std::vector<Element> &elements,
                   const std::vector<std::array<double, 3>> &positions)
    : m_atomicNumbers(elements.size()), m_positions(3, positions.size()),
      m_elements(elements), m_partial_charges(Vec::Zero(positions.size())) {
  for (size_t i = 0; i < size(); i++) {
    m_atomicNumbers(i) = m_elements[i].atomic_number();
    m_positions(0, i) = positions[i][0];
    m_positions(1, i) = positions[i][1];
    m_positions(2, i) = positions[i][2];
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const std::vector<occ::core::Atom> &atoms)
    : m_atomicNumbers(atoms.size()), m_positions(3, atoms.size()),
      m_partial_charges(Vec::Zero(atoms.size())) {
  m_elements.reserve(atoms.size());
  for (size_t i = 0; i < atoms.size(); i++) {
    const auto &atom = atoms[i];
    m_elements.push_back(Element(atom.atomic_number));
    m_atomicNumbers(i) = atom.atomic_number;
    // Internally store in angstroms
    m_positions(0, i) = atom.x * occ::units::BOHR_TO_ANGSTROM;
    m_positions(1, i) = atom.y * occ::units::BOHR_TO_ANGSTROM;
    m_positions(2, i) = atom.z * occ::units::BOHR_TO_ANGSTROM;
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const Molecule &mol_a, const Molecule &mol_b)
    : m_atomicNumbers(mol_a.size() + mol_b.size()),
      m_positions(3, mol_a.size() + mol_b.size()),
      m_partial_charges(Vec::Zero(mol_a.size() + mol_b.size())) {

  // Combine atomic numbers and positions
  m_atomicNumbers << mol_a.atomic_numbers(), mol_b.atomic_numbers();
  m_positions << mol_a.positions(), mol_b.positions();

  // Combine elements
  m_elements.reserve(mol_a.size() + mol_b.size());
  m_elements.insert(m_elements.end(), mol_a.elements().begin(),
                    mol_a.elements().end());
  m_elements.insert(m_elements.end(), mol_b.elements().begin(),
                    mol_b.elements().end());

  // Combine partial charges if they exist
  if (mol_a.partial_charges().size() > 0 ||
      mol_b.partial_charges().size() > 0) {
    Vec charges_a = mol_a.partial_charges().size() > 0
                        ? mol_a.partial_charges()
                        : Vec::Zero(mol_a.size());
    Vec charges_b = mol_b.partial_charges().size() > 0
                        ? mol_b.partial_charges()
                        : Vec::Zero(mol_b.size());
    m_partial_charges << charges_a, charges_b;
  }

  // Combine asymmetric unit indices if they exist
  if (mol_a.asymmetric_unit_idx().size() > 0 ||
      mol_b.asymmetric_unit_idx().size() > 0) {
    IVec asym_a = mol_a.asymmetric_unit_idx().size() > 0
                      ? mol_a.asymmetric_unit_idx()
                      : IVec::Zero(mol_a.size());
    IVec asym_b = mol_b.asymmetric_unit_idx().size() > 0
                      ? mol_b.asymmetric_unit_idx()
                      : IVec::Zero(mol_b.size());
    m_asym_idx.resize(mol_a.size() + mol_b.size());
    m_asym_idx << asym_a, asym_b;
  }

  // Combine unit cell indices if they exist
  if (mol_a.unit_cell_idx().size() > 0 || mol_b.unit_cell_idx().size() > 0) {
    IVec uc_a = mol_a.unit_cell_idx().size() > 0 ? mol_a.unit_cell_idx()
                                                 : IVec::Zero(mol_a.size());
    IVec uc_b = mol_b.unit_cell_idx().size() > 0 ? mol_b.unit_cell_idx()
                                                 : IVec::Zero(mol_b.size());
    m_uc_idx.resize(mol_a.size() + mol_b.size());
    m_uc_idx << uc_a, uc_b;
  }

  // Combine unit cell shifts if they exist
  if (mol_a.unit_cell_shift().cols() > 0 ||
      mol_b.unit_cell_shift().cols() > 0) {
    IMat3N shifts_a = mol_a.unit_cell_shift().cols() > 0
                          ? mol_a.unit_cell_shift()
                          : IMat3N::Zero(3, mol_a.size());
    IMat3N shifts_b = mol_b.unit_cell_shift().cols() > 0
                          ? mol_b.unit_cell_shift()
                          : IMat3N::Zero(3, mol_b.size());
    m_uc_shifts.resize(3, mol_a.size() + mol_b.size());
    m_uc_shifts << shifts_a, shifts_b;
  }

  // Combine asymmetric unit symop if they exist
  if (mol_a.asymmetric_unit_symop().size() > 0 ||
      mol_b.asymmetric_unit_symop().size() > 0) {
    IVec symop_a = mol_a.asymmetric_unit_symop().size() > 0
                       ? mol_a.asymmetric_unit_symop()
                       : IVec::Zero(mol_a.size());
    IVec symop_b = mol_b.asymmetric_unit_symop().size() > 0
                       ? mol_b.asymmetric_unit_symop()
                       : IVec::Zero(mol_b.size());
    m_asym_symop.resize(mol_a.size() + mol_b.size());
    m_asym_symop << symop_a, symop_b;
  }

  // Combine charge and multiplicity appropriately
  m_charge = mol_a.charge() + mol_b.charge();
  m_multiplicity = mol_a.multiplicity() + mol_b.multiplicity() - 1;

  // Set name
  m_name = mol_a.name() + " + " + mol_b.name();

  // Note: asymmetric_molecule_idx and unit_cell_molecule_idx are not meaningful
  // for a combined supermolecule, so we leave them as -1 (default)
}

std::vector<occ::core::Atom> Molecule::atoms() const {
  std::vector<occ::core::Atom> result(size());
  using occ::units::ANGSTROM_TO_BOHR;
  for (size_t i = 0; i < size(); i++) {
    result[i] = {m_atomicNumbers(i), m_positions(0, i) * ANGSTROM_TO_BOHR,
                 m_positions(1, i) * ANGSTROM_TO_BOHR,
                 m_positions(2, i) * ANGSTROM_TO_BOHR};
  }
  return result;
}

Vec Molecule::vdw_radii() const {
  Vec radii(size());
  for (size_t i = 0; i < radii.size(); i++) {
    radii(i) = static_cast<double>(m_elements[i].van_der_waals_radius());
  }
  return radii;
}

Vec Molecule::covalent_radii() const {
  Vec radii(size());
  for (size_t i = 0; i < radii.size(); i++) {
    radii(i) = static_cast<double>(m_elements[i].covalent_radius());
  }
  return radii;
}

Vec Molecule::atomic_masses() const {
  Vec masses(size());
  for (size_t i = 0; i < masses.size(); i++) {
    masses(i) = static_cast<double>(m_elements[i].mass());
  }
  return masses;
}

Vec3 Molecule::centroid() const { return m_positions.rowwise().mean(); }

Vec3 Molecule::center_of_mass() const {
  occ::RowVec masses = atomic_masses();
  masses.array() /= masses.sum();
  return (m_positions.array().rowwise() * masses.array()).rowwise().sum();
}

Mat3 Molecule::inertia_tensor() const {
  if (size() == 1)
    return Mat3::Zero();
  // amu angstrom^2
  // 10^-46 kgm^2
  // amu angstrom^2 to 10^-46 kgm^2
  constexpr double kgm2_fac = 1e23 / occ::constants::avogadro<double>;
  // unit is 10^-46 kg m^2
  return occ::core::inertia_tensor(atomic_masses(), m_positions) * kgm2_fac;
}

Vec3 Molecule::principal_moments_of_inertia() const {
  // unit is 10^-46 kg m^2
  if (size() == 1)
    return Vec3::Zero();
  Mat3 T = inertia_tensor();
  Eigen::SelfAdjointEigenSolver<occ::Mat3> solver(T);
  return solver.eigenvalues();
}

Vec3 Molecule::rotational_constants() const {
  // conversion factor from 10^-46 kgm^2 to amu angstrom^2 to GHz or cm^-1
  if (size() == 1)
    return Vec3::Zero();
  constexpr double GHz_factor{505.379045961437 * 1e23 /
                              occ::constants::avogadro<double>};

  /*
  constexpr double per_cm_factor{16.8576304198232 * 1e23 /
                                 occ::constants::avogadro<double>};
  */

  return (GHz_factor / principal_moments_of_inertia().array())
      .unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
}

double Molecule::rotational_free_energy(double temperature) const {
  if (size() == 1)
    return 0.0;
  // unit is 10^-46 kg m^2
  Vec3 r = principal_moments_of_inertia();
  // linear
  double lnZr;
  if (r(0) < 1e-12) {
    double inertia_product = r(2) / 16.60538921;
    lnZr = std::log(inertia_product) * std::log(temperature) + 1.418;
  } else {
    double inertia_product = r(0) * r(1) * r(2) / std::pow(16.60538921, 3);
    lnZr = 0.5 * std::log(inertia_product) + 1.5 * std::log(temperature) +
           1.5 * 1.418;
  }
  occ::log::debug("Rotational partition function: {: 12.6f}\n", lnZr);
  return -occ::constants::boltzmann<double> * temperature * lnZr *
         occ::constants::avogadro<double> / 1000;
}

double Molecule::translational_free_energy(double temperature) const {
  // Gibbs free energy, in kJ/mol
  constexpr double pressure{1.0};
  constexpr double kB = occ::constants::boltzmann<double>;
  constexpr double NA = occ::constants::avogadro<double>;
  constexpr double h = occ::constants::planck<double>;
  const double factor =
      1.5 * std::log(2 * M_PI / (h * h)) +
      1.5 * std::log(occ::constants::molar_mass_constant<double>) -
      0.5 * std::log(NA) + 2.5 * std::log(kB) - std::log(1e5);

  double total_mass = atomic_masses().array().sum();
  double lnZt = 1.5 * std::log(total_mass) + 2.5 * std::log(temperature) -
                std::log(pressure) + factor;
  occ::log::debug("Translational partition function: {: 12.6f}\n", lnZt);
  double Gt = -kB * temperature * lnZt;
  double Gn = kB * temperature * (std::log(NA) - 1);

  // missing RT for Gibss vs Helmholtz
  double RT = temperature * kB * NA / 1000;

  return (Gt + Gn) * NA / 1000 + RT;
}

bool Molecule::is_comparable_to(const Molecule &other) const {
  if (size() != other.size())
    return false;
  return (m_atomicNumbers.array() == other.m_atomicNumbers.array()).all();
}

void Molecule::rotate(const Eigen::Affine3d &rotation, Origin origin) {
  rotate(rotation.linear(), origin);
}

Molecule Molecule::rotated(const Eigen::Affine3d &rotation,
                           Origin origin) const {
  return rotated(rotation.linear(), origin);
}

void Molecule::rotate(const occ::Mat3 &rotation, Origin origin) {
  Vec3 O = {0, 0, 0};
  switch (origin) {
  case Centroid: {
    O = centroid();
    break;
  }
  case CenterOfMass: {
    O = center_of_mass();
    break;
  }
  default:
    break;
  }
  rotate(rotation, O);
}

Vec3 Molecule::position(Origin origin) const {
  Vec3 O = {0, 0, 0};
  switch (origin) {
  case Centroid: {
    O = centroid();
    break;
  }
  case CenterOfMass: {
    O = center_of_mass();
    break;
  }
  default:
    throw std::runtime_error("Could not get molecular position.");
  }
  return O;
}

void Molecule::rotate(const occ::Mat3 &rotation, const Vec3 &origin) {
  translate(-origin);
  m_positions = rotation * m_positions;
  translate(origin);
  m_asymmetric_unit_rotation = rotation * m_asymmetric_unit_rotation;
}

Molecule Molecule::rotated(const occ::Mat3 &rotation, Origin origin) const {
  Molecule result = *this;
  result.rotate(rotation, origin);
  return result;
}

Molecule Molecule::rotated(const occ::Mat3 &rotation,
                           const Vec3 &origin) const {
  Molecule result = *this;
  result.rotate(rotation, origin);
  return result;
}

void Molecule::translate(const occ::Vec3 &translation) {
  m_positions.colwise() += translation;
  m_asymmetric_unit_translation += translation;
}

Molecule Molecule::translated(const occ::Vec3 &translation) const {
  Molecule result = *this;
  result.translate(translation);
  return result;
}

void Molecule::transform(const Mat4 &transform, Origin origin) {
  rotate(transform.block<3, 3>(0, 0), origin);
  translate(transform.block<3, 1>(0, 3));
}

void Molecule::transform(const Mat4 &transform, const Vec3 &origin) {
  rotate(transform.block<3, 3>(0, 0), origin);
  translate(transform.block<3, 1>(0, 3));
}

Molecule Molecule::transformed(const Mat4 &transform, Origin origin) const {
  Molecule result = *this;
  result.transform(transform, origin);
  return result;
}

Molecule Molecule::transformed(const Mat4 &transform,
                               const Vec3 &origin) const {
  Molecule result = *this;
  result.transform(transform, origin);
  return result;
}

std::tuple<size_t, size_t, double>
Molecule::nearest_atom(const Molecule &other) const {
  std::tuple<size_t, size_t, double> result{0, 0,
                                            std::numeric_limits<double>::max()};
  for (size_t i = 0; i < size(); i++) {
    const occ::Vec3 &p1 = m_positions.col(i);
    for (size_t j = 0; j < other.size(); j++) {
      const occ::Vec3 &p2 = other.m_positions.col(j);
      double d = (p2 - p1).norm();
      if (d < std::get<2>(result))
        result = {i, j, d};
    }
  }
  return result;
}

Vec Molecule::interatomic_distances() const {
  // upper triangle of distance matrix
  size_t N = size();
  size_t num_idxs = N * (N - 1) / 2;
  Vec result(num_idxs);
  size_t idx = 0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = i + 1; j < N; j++) {
      result(idx++) = (m_positions.col(i) - m_positions.col(j)).norm();
    }
  }
  return result;
}

bool Molecule::is_equivalent_to(const Molecule &rhs) const {
  if (!is_comparable_to(rhs))
    return false;
  auto dists_a = interatomic_distances();
  auto dists_b = rhs.interatomic_distances();
  return occ::util::all_close(dists_a, dists_b, 1e-8, 1e-8);
}

void Molecule::set_asymmetric_unit_symop(const IVec &symop) {
  m_asym_symop = symop;
}

void Molecule::set_cell_shift(const IVec3 &shift, bool update_atoms) {
  IVec3 diff = shift - m_cell_shift;
  m_cell_shift = shift;
  if (update_atoms && (m_uc_shifts.size() > 0)) {
    m_uc_shifts.colwise() += diff;
  }
}

const IVec3 &Molecule::cell_shift() const { return m_cell_shift; }

double Molecule::molar_mass() const {
  return occ::constants::molar_mass_constant<double> *
         atomic_masses().array().sum();
}

Vec Molecule::esp_partial_charges(const Mat3N &positions_angs) const {
  Vec result = Vec::Zero(positions_angs.cols());
  for (int i = 0; i < m_partial_charges.rows(); i++) {
    double q = m_partial_charges(i) / occ::units::ANGSTROM_TO_BOHR;
    Vec3 atom_pos = m_positions.col(i);
    auto ab = positions_angs.colwise() - atom_pos;
    auto r = ab.colwise().norm();
    result.array() += q / r.array();
  }
  return result;
}

inline std::string next_label(std::string current = "") {
  if (current.empty())
    return "A";
  int pos = current.size() - 1;
  while (pos >= 0) {
    if (current[pos] < 'Z') {
      current[pos]++;
      return current;
    }
    current[pos] = 'A';
    pos--;
  }
  return "A" + current;
}

void label_molecules_by_chemical_formula(
    std::vector<occ::core::Molecule> &molecules) {
  struct LabelInfo {
    int formula_id;
    std::string current_letter;
  };
  ankerl::unordered_dense::map<std::string, LabelInfo> formula_to_label;
  int next_id = 1;

  for (auto &molecule : molecules) {
    auto formula = chemical_formula(molecule.elements());
    auto it = formula_to_label.find(formula);
    if (it == formula_to_label.end()) {
      // New formula type encountered
      formula_to_label[formula] = {next_id++, "A"};
      molecule.set_name(std::to_string(formula_to_label[formula].formula_id) +
                        "A");
    } else {
      // Existing formula type - use same ID but next letter
      it->second.current_letter = next_label(it->second.current_letter);
      molecule.set_name(std::to_string(it->second.formula_id) +
                        it->second.current_letter);
    }
  }
}

Molecule Molecule::permute(const std::vector<int> &permutation) const {
  if (permutation.size() != size()) {
    throw std::runtime_error("Permutation size does not match molecule size");
  }

  // Start with a copy
  Molecule result(*this);

  // Reorder all atom-based properties using Eigen's indexing
  result.m_atomicNumbers = m_atomicNumbers(permutation);
  result.m_positions = m_positions(Eigen::all, permutation);

  // Reorder other atom-based properties if they exist
  if (m_uc_idx.size() > 0) {
    result.m_uc_idx = m_uc_idx(permutation);
  }
  if (m_uc_shifts.cols() > 0) {
    result.m_uc_shifts = m_uc_shifts(Eigen::all, permutation);
  }
  if (m_asym_idx.size() > 0) {
    result.m_asym_idx = m_asym_idx(permutation);
  }
  if (m_asym_symop.size() > 0) {
    result.m_asym_symop = m_asym_symop(permutation);
  }
  if (m_partial_charges.size() > 0) {
    result.m_partial_charges = m_partial_charges(permutation);
  }

  // Reorder elements vector
  result.m_elements.clear();
  for (int i : permutation) {
    result.m_elements.push_back(m_elements[i]);
  }

  // Update bonds - need to map old indices to new indices
  result.m_bonds.clear();
  std::vector<int> inverse_permutation(size());
  for (size_t i = 0; i < permutation.size(); i++) {
    inverse_permutation[permutation[i]] = i;
  }
  for (const auto &bond : m_bonds) {
    result.m_bonds.push_back(
        {static_cast<size_t>(inverse_permutation[bond.first]),
         static_cast<size_t>(inverse_permutation[bond.second])});
  }

  return result;
}

} // namespace occ::core
