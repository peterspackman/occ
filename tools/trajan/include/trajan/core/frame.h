#pragma once
#include <occ/crystal/unitcell.h>
#include <trajan/core/atom.h>
#include <trajan/core/molecule.h>
#include <trajan/core/neigh.h>
// #include <trajan/core/unit_cell.h>
#include <vector>

namespace trajan::core {

using occ::crystal::UnitCell;

struct Frame {

public:
  inline const UnitCell &unit_cell() const { return m_uc; }
  void set_uc(UnitCell &uc);

  inline const std::vector<Atom> &atoms() const { return m_atoms; }
  void set_atoms(const std::vector<Atom> &atoms);
  inline size_t num_atoms() const { return m_num_atoms; }

  void update_atom_position(size_t idx, Vec3 &pos);

  const Mat3N cart_pos() const;
  inline const Mat3N &wrapped_cart_pos() const { return m_wrapped_cart_pos; }
  inline const Mat3N &frac_pos() const { return m_frac_pos; }

  inline void set_cart_pos(Mat3N &cart_pos) { m_cart_pos = cart_pos; }
  inline void set_frac_pos(Mat3N &frac_pos) { m_frac_pos = frac_pos; }
  inline void set_wrapped_cart_pos(Mat3N &wrapped_cart_pos) {
    m_wrapped_cart_pos = wrapped_cart_pos;
  }
  inline const size_t &index() const { return m_index; }
  inline void set_index(size_t index) { m_index = index; }
  inline const double &timestep() const { return m_timestep; }
  inline void set_timestep(double timestep) { m_timestep = timestep; }

  Frame() = default;

  Frame(const Frame &other) {
    this->m_index = other.index();
    this->m_uc = other.unit_cell();
    this->m_atoms = other.atoms();
    this->m_num_atoms = other.num_atoms();
    this->m_cart_pos = other.cart_pos();
    this->m_frac_pos = other.frac_pos();
    this->m_wrapped_cart_pos = other.wrapped_cart_pos();
  }

private:
  size_t m_index = 0;
  UnitCell m_uc;
  std::vector<Atom> m_atoms;
  size_t m_num_atoms = 0;
  Mat3N m_cart_pos;
  Mat3N m_frac_pos;
  Mat3N m_wrapped_cart_pos;
  double m_timestep;

  void update_positions();
};

}; // namespace trajan::core
