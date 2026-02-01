#include <fmt/core.h>
#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/frame.h>
#include <trajan/core/log.h>

namespace trajan::core {

using occ::crystal::UnitCell;

void Frame::set_atoms(const std::vector<Atom> &atoms) {
  int num_atoms = atoms.size();
  if (num_atoms == 0) {
    throw std::runtime_error("No atoms!");
  }
  if (m_num_atoms != num_atoms && m_num_atoms != 0) {
    trajan::log::debug(fmt::format(
        "Number of atoms has changed. Previously {} atoms, now {} atoms.",
        m_num_atoms, num_atoms));
  }
  m_num_atoms = num_atoms;
  m_atoms = atoms;
}

void Frame::set_unit_cell(const UnitCell &unit_cell) {
  m_unit_cell = unit_cell;
}

const Mat3N Frame::cart_pos(std::optional<double> scale) const {
  Mat3N cart_pos(3, m_num_atoms);
  if (scale.has_value()) {
    const double scale_val = scale.value();
    for (size_t i = 0; i < m_num_atoms; i++) {
      const Atom &atom = m_atoms[i];
      cart_pos(0, i) = atom.x * scale_val;
      cart_pos(1, i) = atom.y * scale_val;
      cart_pos(2, i) = atom.z * scale_val;
    }
  } else {
    for (size_t i = 0; i < m_num_atoms; i++) {
      const Atom &atom = m_atoms[i];
      cart_pos(0, i) = atom.x;
      cart_pos(1, i) = atom.y;
      cart_pos(2, i) = atom.z;
    }
  }
  return cart_pos;
}

const occ::Vec Frame::cart_pos_flat(std::optional<double> scale) const {
  occ::Vec cart_pos(m_num_atoms * 3);
  if (scale.has_value()) {
    const double scale_val = scale.value();
    for (size_t i = 0; i < m_num_atoms; ++i) {
      Atom atom = m_atoms[i];
      cart_pos[3 * i] = atom.x * scale_val;
      cart_pos[3 * i + 1] = atom.y * scale_val;
      cart_pos[3 * i + 2] = atom.z * scale_val;
    }
  } else {
    for (size_t i = 0; i < m_num_atoms; ++i) {
      Atom atom = m_atoms[i];
      cart_pos[3 * i] = atom.x;
      cart_pos[3 * i + 1] = atom.y;
      cart_pos[3 * i + 2] = atom.z;
    }
  }
  return cart_pos;
}

std::vector<int> Frame::atomic_numbers() const {
  std::vector<int> atomic_numbers(m_atoms.size());
  for (int i = 0; i < m_atoms.size(); i++) {
    atomic_numbers[i] = m_atoms[i].atomic_number();
  }
  return atomic_numbers;
};

void Frame::update_atom_position(size_t idx, Vec3 &pos) {
  if (idx >= m_num_atoms) {
    throw std::runtime_error("Bad index.");
  }
  m_atoms[idx].set_position(pos);
}

} // namespace trajan::core
