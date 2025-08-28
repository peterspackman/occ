#include <fmt/core.h>
#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/frame.h>
#include <trajan/core/log.h>
// #include <trajan/core/unit_cell.h>

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
  // Mat3N cart_pos(3, m_num_atoms);
  // for (size_t i = 0; i < m_num_atoms; ++i) {
  //   Atom atom = m_atoms[i];
  //   cart_pos(0, i) = atom.x;
  //   cart_pos(1, i) = atom.y;
  //   cart_pos(2, i) = atom.z;
  // }
  // m_cart_pos = cart_pos;
  // m_positions_needs_update |= ATOMS_INIT;
  // if (!(m_positions_needs_update & UC_INIT)) {
  //   return;
  // }
  // auto result = trajan::core::wrap_coordinates(m_cart_pos, m_uc);
  // m_frac_pos = result.first;
  // m_wrapped_cart_pos = result.second;
}

void Frame::set_uc(UnitCell &uc) {
  m_uc = uc;
  // m_positions_needs_update |= UC_INIT;
  // if (!(m_positions_needs_update & ATOMS_INIT)) {
  //   return;
  // }
  // auto result = trajan::core::wrap_coordinates(m_cart_pos, m_uc);
  // m_frac_pos = result.first;
  // m_wrapped_cart_pos = result.second;
}

const Mat3N Frame::cart_pos() const {
  Mat3N cart_pos(3, m_num_atoms);
  for (size_t i = 0; i < m_num_atoms; ++i) {
    Atom atom = m_atoms[i];
    cart_pos(0, i) = atom.x;
    cart_pos(1, i) = atom.y;
    cart_pos(2, i) = atom.z;
  }
  return cart_pos;
}

void Frame::update_atom_position(size_t idx, Vec3 &pos) {
  if (idx >= m_num_atoms) {
    throw std::runtime_error("Bad index.");
  }
  m_atoms[idx].update_position(pos);
}

} // namespace trajan::core
