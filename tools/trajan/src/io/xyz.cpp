#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/unitcell.h>
#include <trajan/io/xyz.h>

#include <sstream>

namespace trajan::io {

using occ::Vec3;
using occ::crystal::UnitCell;
using trajan::core::Frame;

bool XYZHandler::_initialise() {
  if (m_mode == Mode::Read) {
    m_infile.open(file_path());
    return m_infile.is_open();
  } else {
    m_outfile.open(file_path());
    return m_outfile.is_open();
  }
}

void XYZHandler::_finalise() {
  if (m_mode == Mode::Read) {
    if (m_infile.is_open()) {
      m_infile.close();
    }
  } else {
    if (m_outfile.is_open()) {
      m_outfile.close();
    }
  }
}

bool XYZHandler::read_next_frame(Frame &frame) {
  // std::string line;
  // // First line is the number of atoms
  // if (!std::getline(m_infile, line)) {
  //   return false; // End of file
  // }
  //
  // size_t num_atoms = std::stoul(line);
  // if (frame.num_atoms() == 0) {
  //   frame.resize(num_atoms);
  // } else if (frame.num_atoms() != num_atoms) {
  //   // Handle error or resize
  //   return false;
  // }
  //
  // // Second line is the comment/title, may contain unit cell info
  // if (!std::getline(m_infile, line)) {
  //   return false;
  // }
  //
  // // Attempt to parse unit cell from comment line
  // std::stringstream ss(line);
  // std::string keyword;
  // ss >> keyword;
  // if (keyword == "Lattice=") {
  //   UnitCell uc;
  //   double a, b, c, alpha, beta, gamma;
  //   ss >> a >> b >> c >> alpha >> beta >> gamma;
  //   uc.set_a(a);
  //   uc.set_b(b);
  //   uc.set_c(c);
  //   uc.set_alpha(alpha);
  //   uc.set_beta(beta);
  //   uc.set_gamma(gamma);
  //   frame.set_uc(uc);
  // }
  //
  // // Read atom lines
  // for (size_t i = 0; i < num_atoms; ++i) {
  //   if (!std::getline(m_infile, line)) {
  //     return false; // Should not happen in a valid file
  //   }
  //   std::stringstream atom_ss(line);
  //   std::string element;
  //   double x, y, z;
  //   atom_ss >> element >> x >> y >> z;
  //   frame.update_atom_position(i, {x, y, z});
  //   // TODO: Set element/atom type if topology supports it
  // }

  return true;
}

bool XYZHandler::write_next_frame(const Frame &frame) {
  // m_outfile << frame.num_atoms() << "\n";
  //
  // const auto &uc = frame.unit_cell();
  // m_outfile << fmt::format("Lattice=\"{:f} {:f} {:f} {:f} {:f} {:f}\"\n",
  //                          uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(),
  //                          uc.gamma());
  //
  // const auto &atoms = frame.atoms();
  // for (const auto &atom : atoms) {
  //   m_outfile << fmt::format("{} {:f} {:f} {:f}\n", "X", atom.x, atom.y,
  //                            atom.z);
  // }

  return m_outfile.good();
}

} // namespace trajan::io
