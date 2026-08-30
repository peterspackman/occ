#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <occ/core/bondgraph.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/solvent/sigma_dispersion.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace occ::solvent::sigma {

namespace {

// Hsieh, Lin & Vrabec, Fluid Phase Equilib. 367 (2014) 72, Table 3, in K.
constexpr double eps_c_sp3 = 115.7023;
constexpr double eps_c_sp2 = 117.4650;
constexpr double eps_c_sp = 66.0691;
constexpr double eps_o_two_bonds = 95.6184;  // -O-
constexpr double eps_o_one_bond = -11.0549;  // =O
constexpr double eps_n_sp3 = 15.4901;
constexpr double eps_n_sp2 = 84.6268;
constexpr double eps_n_sp = 109.6621;
constexpr double eps_f = 52.9318;
constexpr double eps_cl = 104.2534;
constexpr double eps_h_on_o = 19.3477;
constexpr double eps_h_on_n = 141.1709;
constexpr double eps_h_water = 58.3301;

constexpr double w_dispersion = 0.27027;

using BondList = std::vector<std::vector<Eigen::Index>>;

BondList perceive_bonds(const IVec &numbers, const Mat3N &positions_bohr) {
  const Eigen::Index n = numbers.size();
  BondList bonds(n);
  if (n == 2) {
    // A diatomic is bonded by construction; the distance criterion is not
    // reliable at that size.
    bonds[0].push_back(1);
    bonds[1].push_back(0);
    return bonds;
  }

  Vec radii(n);
  for (Eigen::Index i = 0; i < n; i++)
    radii(i) = core::Element(numbers(i)).covalent_radius();
  const Mat3N positions = positions_bohr * units::BOHR_TO_ANGSTROM;

  for (Eigen::Index i = 0; i < n; i++) {
    for (Eigen::Index j = i + 1; j < n; j++) {
      const double threshold =
          radii(i) + radii(j) + core::covalent_bond_tolerance;
      if ((positions.col(i) - positions.col(j)).squaredNorm() <
          threshold * threshold) {
        bonds[i].push_back(j);
        bonds[j].push_back(i);
      }
    }
  }
  return bonds;
}

bool is_water_molecule(const IVec &numbers) {
  return numbers.size() == 3 && (numbers.array() == 1).count() == 2 &&
         (numbers.array() == 8).count() == 1;
}

/// True when the carbon at `c` is the carbon of a -C(=O)OH group: three
/// bonds, two of them to oxygen, one of those oxygens bound to exactly one
/// carbon and one hydrogen.
bool is_carboxyl_carbon(Eigen::Index c, const IVec &numbers,
                        const BondList &bonds) {
  if (bonds[c].size() != 3)
    return false;
  int oxygens = 0;
  for (auto j : bonds[c])
    oxygens += (numbers(j) == 8);
  if (oxygens != 2)
    return false;

  for (auto j : bonds[c]) {
    if (numbers(j) != 8 || bonds[j].size() != 2)
      continue;
    int carbons = 0, hydrogens = 0;
    for (auto k : bonds[j]) {
      carbons += (numbers(k) == 6);
      hydrogens += (numbers(k) == 1);
    }
    if (carbons == 1 && hydrogens == 1)
      return true;
  }
  return false;
}

/// The atom's parameter, or nullopt when it carries none and is left out of
/// the average. Elements without a table entry are rejected by the caller.
std::optional<double> atom_parameter(Eigen::Index i, const IVec &numbers,
                                     const BondList &bonds, bool water) {
  const size_t degree = bonds[i].size();
  switch (numbers(i)) {
  case 1:
    if (water)
      return eps_h_water;
    for (auto j : bonds[i])
      if (numbers(j) == 8)
        return eps_h_on_o;
    for (auto j : bonds[i])
      if (numbers(j) == 7)
        return eps_h_on_n;
    return std::nullopt; // hydrogen on carbon: no parameter
  case 6:
    if (degree == 4)
      return eps_c_sp3;
    if (degree == 3)
      return eps_c_sp2;
    if (degree == 2)
      return eps_c_sp;
    return std::nullopt;
  case 7:
    if (degree == 3)
      return eps_n_sp3;
    if (degree == 2)
      return eps_n_sp2;
    if (degree == 1)
      return eps_n_sp;
    return std::nullopt;
  case 8:
    if (degree == 2)
      return eps_o_two_bonds;
    if (degree == 1)
      return eps_o_one_bond;
    return std::nullopt;
  case 9:
    return eps_f;
  case 17:
    return eps_cl;
  default:
    return std::nullopt;
  }
}

bool has_table_entry(int atomic_number) {
  switch (atomic_number) {
  case 1:
  case 6:
  case 7:
  case 8:
  case 9:
  case 17:
    return true;
  default:
    return false;
  }
}

/// Nitrogen and oxygen outside the tabulated bond counts have no defined
/// type; carbon is skipped instead, as in the reference implementation.
bool degree_is_typable(int atomic_number, size_t degree) {
  if (atomic_number == 7)
    return degree >= 1 && degree <= 3;
  if (atomic_number == 8)
    return degree == 1 || degree == 2;
  return true;
}

DispersionClass hydrogen_bond_class(const IVec &numbers,
                                    const BondList &bonds) {
  bool has_heteroatom = false, has_donor = false;
  for (Eigen::Index i = 0; i < numbers.size(); i++) {
    const int z = numbers(i);
    if (z != 7 && z != 8 && z != 9)
      continue;
    has_heteroatom = true;
    for (auto j : bonds[i])
      if (numbers(j) == 1)
        has_donor = true;
  }
  if (!has_heteroatom)
    return DispersionClass::None;
  return has_donor ? DispersionClass::DonorAcceptor
                   : DispersionClass::Acceptor;
}

bool sign_is_flipped(DispersionClass a, DispersionClass b) {
  auto pair = [&](DispersionClass x, DispersionClass y) {
    return (a == x && b == y) || (b == x && a == y);
  };
  return pair(DispersionClass::Water, DispersionClass::Acceptor) ||
         pair(DispersionClass::Water, DispersionClass::Carboxyl) ||
         pair(DispersionClass::Carboxyl, DispersionClass::None) ||
         pair(DispersionClass::Carboxyl, DispersionClass::DonorAcceptor);
}

} // namespace

std::string_view dispersion_class_name(DispersionClass klass) {
  switch (klass) {
  case DispersionClass::None:
    return "NHB";
  case DispersionClass::Acceptor:
    return "HB-ACCEPTOR";
  case DispersionClass::DonorAcceptor:
    return "HB-DONOR-ACCEPTOR";
  case DispersionClass::Carboxyl:
    return "COOH";
  case DispersionClass::Water:
    return "H2O";
  }
  return "NHB";
}

DispersionClass dispersion_class_from_name(std::string_view name) {
  if (name == "HB-ACCEPTOR")
    return DispersionClass::Acceptor;
  if (name == "HB-DONOR-ACCEPTOR")
    return DispersionClass::DonorAcceptor;
  if (name == "COOH")
    return DispersionClass::Carboxyl;
  if (name == "H2O")
    return DispersionClass::Water;
  if (name == "NHB")
    return DispersionClass::None;
  throw std::runtime_error(
      fmt::format("unknown COSMO-SAC-dsp class '{}'", name));
}

Dispersion dispersion_parameters(const IVec &numbers,
                                 const Mat3N &positions_bohr) {
  Dispersion result;
  const Eigen::Index n = numbers.size();
  if (n == 0 || positions_bohr.cols() != n)
    return result;

  const BondList bonds = perceive_bonds(numbers, positions_bohr);
  const bool water = is_water_molecule(numbers);

  double sum = 0.0;
  int typed = 0;
  bool carboxyl = false;
  for (Eigen::Index i = 0; i < n; i++) {
    if (!has_table_entry(numbers(i)) ||
        !degree_is_typable(numbers(i), bonds[i].size()))
      return result; // known stays false
    if (numbers(i) == 6 && is_carboxyl_carbon(i, numbers, bonds))
      carboxyl = true;
    if (auto parameter = atom_parameter(i, numbers, bonds, water)) {
      sum += *parameter;
      typed++;
    }
  }
  if (typed == 0)
    return result;

  result.epsilon = sum / typed;
  result.known = true;
  if (water)
    result.klass = DispersionClass::Water;
  else if (carboxyl)
    result.klass = DispersionClass::Carboxyl;
  else
    result.klass = hydrogen_bond_class(numbers, bonds);
  return result;
}

double dispersion_coefficient(const Dispersion &a, const Dispersion &b) {
  if (!a.known || !b.known)
    return 0.0;
  if (a.epsilon <= 0.0 || b.epsilon <= 0.0)
    return 0.0;
  const double w = sign_is_flipped(a.klass, b.klass) ? -w_dispersion
                                                     : w_dispersion;
  return w * (0.5 * (a.epsilon + b.epsilon) -
              std::sqrt(a.epsilon * b.epsilon));
}

Vec dispersion_ln_gamma(const Dispersion &a, const Dispersion &b,
                        const Vec &mole_fractions) {
  if (mole_fractions.size() != 2)
    throw std::runtime_error(
        fmt::format("COSMO-SAC-dsp is defined for two components, got {}",
                    mole_fractions.size()));
  const double coefficient = dispersion_coefficient(a, b);
  Vec out(2);
  out(0) = coefficient * mole_fractions(1) * mole_fractions(1);
  out(1) = coefficient * mole_fractions(0) * mole_fractions(0);
  return out;
}

} // namespace occ::solvent::sigma
