#include "trajan/core/util.h"
#include <memory>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/unitcell.h>
#include <trajan/io/xtc.h>

namespace trajan::io {

using occ::crystal::UnitCell;
using trajan::core::Frame;

bool XTCHandler::_initialise() {
  m_infile.open(this->file_path(), std::ios::binary);
  if (!m_infile.is_open()) {
    return false;
  }

  if (m_mode == Mode::Read) {
    m_xtcreader = std::make_unique<XTCReader>(this->file_path());
  } else {
    m_xtcwriter = std::make_unique<XTCWriter>(this->file_path());
  }
  m_infile.close();
  return true;
}

void XTCHandler::_finalise() {}

bool XTCHandler::read_next_frame(Frame &frame) {
  if (m_xtcreader->eot()) {
    return false;
  }
  m_xtcreader->next_frame();

  occ::Mat3 m;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      // convert to column major and from nm to ang
      m(col, row) = static_cast<double>(m_xtcreader->box[row][col]) * 10.0;
    }
  }
  if (trajan::util::unitcell_is_reasonable(m)) {
    UnitCell uc(m);
    frame.set_unit_cell(uc);
  }

  size_t natoms = m_xtcreader->X.size() / 3;
  for (int i; i < natoms; i++) {
    int iatom = i * 3;
    occ::Vec3 pos = {m_xtcreader->X[iatom] * 10.0,
                     m_xtcreader->X[iatom + 1] * 10.0,
                     m_xtcreader->X[iatom + 2] * 10.0};
    frame.update_atom_position(i, pos);
  }

  return true;
}

bool XTCHandler::write_next_frame(const Frame &frame) {
  size_t natoms = frame.num_atoms();
  uint32_t step = 0;
  float time = 0.0;
  occ::Mat3 m = occ::Mat3::Zero();
  if (frame.has_unit_cell()) {
    m = frame.unit_cell().value().direct();
  };
  std::array<std::array<float, 3>, 3> box;
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      box[row][col] = m(col, row) / 10.0;
    }
  }
  occ::Mat3N mat = frame.cart_pos();
  occ::FMat3N fmat = (mat / 10.0).cast<float>();
  std::vector<float> coords(fmat.data(), fmat.data() + fmat.size());

  return m_xtcwriter->write_frame(natoms, step, time, box, coords.data());
}

} // namespace trajan::io
