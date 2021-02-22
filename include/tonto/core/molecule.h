#pragma once
#include <tonto/core/element.h>
#include <tonto/core/linear_algebra.h>
#include <libint2/atom.h>
#include <Eigen/Geometry>
#include <array>
#include <tuple>

namespace tonto::chem {
using tonto::IVec;
using tonto::Mat3N;
using tonto::Vec;

class Molecule {
public:
  enum Origin { Cartesian, Centroid, CenterOfMass };
  Molecule(const IVec &, const Mat3N &);
  Molecule(const std::vector<libint2::Atom> &atoms);

  template<typename N, typename D>
  Molecule(const std::vector<N> &nums, const std::vector<std::array<D, 3>> &pos)
  {
      size_t num_atoms = std::min(nums.size(), pos.size());
      m_atomicNumbers = tonto::IVec(num_atoms);
      m_positions = tonto::Mat3N(3, num_atoms);
      for(size_t i = 0; i < num_atoms; i++) {
          m_atomicNumbers(i) = static_cast<int>(nums[i]);
          m_positions(0, i) = static_cast<double>(pos[i][0]);
          m_positions(1, i) = static_cast<double>(pos[i][1]);
          m_positions(2, i) = static_cast<double>(pos[i][2]);
      }
      for (size_t i = 0; i < size(); i++) {
        m_elements.push_back(Element(m_atomicNumbers(i)));
      }
      m_name = chemical_formula(m_elements);
  }

  size_t size() const { return m_atomicNumbers.size(); }

  void set_name(const std::string &);
  const std::string &name() const { return m_name; }

  const Mat3N &positions() const { return m_positions; }
  const IVec &atomic_numbers() const { return m_atomicNumbers; }
  const Vec vdw_radii() const;
  const Vec atomic_masses() const;
  std::vector<libint2::Atom> atoms() const;

  const tonto::Vec3 centroid() const;
  const tonto::Vec3 center_of_mass() const;

  std::tuple<size_t, size_t, double> nearest_atom(const Molecule&) const;

  void add_bond(size_t l, size_t r) { m_bonds.push_back({l, r}); }
  void set_bonds(const std::vector<std::pair<size_t, size_t>> &bonds) {
    m_bonds = bonds;
  }

  const std::vector<std::pair<size_t, size_t>> &bonds() const {
    return m_bonds;
  }
  int charge() const { return m_charge; }
  int multiplicity() const { return m_multiplicity; }
  int num_electrons() const { return m_atomicNumbers.sum() - m_charge; }

  bool comparable_to(const Molecule&) const;

  void set_unit_cell_idx(const IVec &idx) { m_uc_idx = idx; }
  void set_asymmetric_unit_idx(const IVec &idx) { m_asym_idx = idx; }

  void rotate(const Eigen::Affine3d &r, Origin o = Cartesian);
  void rotate(const tonto::Mat3 &r, Origin o = Cartesian);
  void transform(const tonto::Mat4 &t, Origin o = Cartesian);
  void translate(const tonto::Vec3&);

private:
  int m_charge{0};
  int m_multiplicity{1};
  std::string m_name{""};
  std::vector<libint2::Atom> m_atoms;
  IVec m_atomicNumbers;
  Mat3N m_positions;
  IVec m_uc_idx;
  IVec m_asym_idx;
  std::vector<std::pair<size_t, size_t>> m_bonds;
  std::vector<Element> m_elements;
};

Molecule read_xyz_file(const std::string &);

} // namespace tonto::chem
