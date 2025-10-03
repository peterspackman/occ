#include <fmt/core.h>
#include <fstream>
#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/atom.h>
// #include <trajan/core/element.h>
#include <trajan/core/frame.h>
#include <trajan/core/log.h>
// #include <trajan/core/unit_cell.h>
#include <trajan/core/units.h>
#include <trajan/io/pdb.h>

namespace trajan::io {

using occ::crystal::triclinic_cell;
using occ::crystal::UnitCell;
using trajan::core::Frame;
using trajan::units::radians;
using Atom = trajan::core::EnhancedAtom;

bool PDBHandler::_initialise() {
  if (m_mode == Mode::Read) {
    m_infile.open(this->file_path());
    return m_infile.is_open();
  } else {
    m_outfile.open(this->file_path());
    return m_outfile.is_open();
  }
}

void PDBHandler::_finalise() {
  if (m_mode == Mode::Read) {
    if (m_infile.is_open()) {
      m_infile.close();
    }
  } else {
    if (m_outfile.is_open()) {
      m_outfile.flush();
      m_outfile.close();
    }
  }
}

bool PDBHandler::parse_pdb(Frame &frame) {

  std::vector<core::Atom> atoms;
  std::string line;
  while (std::getline(m_infile, line)) {
    if (line.substr(0, 6) == "CRYST1") {
      trajan::log::debug("Found unit cell from CRYST1 line in PDB.");
      double a, b, c, alpha, beta, gamma;
      char record_name[7], sg[12], z[5];
      int result =
          std::sscanf(line.c_str(), PDB_CRYST_FMT_READ.data(), record_name, &a,
                      &b, &c, &alpha, &beta, &gamma, sg, z);
      record_name[6] = '\0';
      sg[11] = '\0';
      z[4] = '\0';
      UnitCell uc = triclinic_cell(a, b, c, radians(alpha), radians(beta),
                                   radians(gamma));
      frame.set_unit_cell(uc);
      continue;
    }
    if (line.substr(0, 4) != "ATOM" && line.substr(0, 6) != "HETATM") {
      continue;
    };
    Atom atom;
    char record_name[7], name_buffer[5], alt_loc[2], res_name[4], ins_code[2],
        tmp[12];
    char chain_id[2], element_buffer[3], charge[3];
    int serial, res_seq;
    float occupancy, temp_factor;

    int result =
        std::sscanf(line.c_str(), PDB_LINE_FMT_READ.data(), record_name,
                    &serial, tmp, name_buffer, alt_loc, res_name, tmp, chain_id,
                    &res_seq, ins_code, tmp, &atom.x, &atom.y, &atom.z,
                    &occupancy, &temp_factor, tmp, element_buffer, charge);
    if (result < 16) {
      std::runtime_error(fmt::format("Failed to parse line: '{}'", line));
    }
    name_buffer[4] = '\0';
    alt_loc[1] = '\0';
    res_name[3] = '\0';
    chain_id[1] = '\0';
    ins_code[1] = '\0';
    element_buffer[2] = '\0';
    charge[2] = '\0';
    atom.uindex = serial;
    atom.type = name_buffer;
    occ::util::trim(atom.type);
    std::string element_identifier;
    bool exact;
    std::string element = element_buffer;
    if (!element.empty()) {
      exact = true;
      element_identifier = element;
    } else {
      exact = false;
      element_identifier = name_buffer;
    }
    atom.element = core::Element(element_identifier, exact);

    atom.index = atoms.size();
    atoms.push_back(atom);
  }
  frame.set_atoms(atoms);
  return true;
}

bool PDBHandler::read_next_frame(Frame &frame) {
  if (m_has_read) {
    return false;
  }
  bool success = this->parse_pdb(frame);

  m_has_read = true;

  return success;
}

bool PDBHandler::write_next_frame(const Frame &frame) {
  const auto &atoms = frame.atoms();
  const auto &uc = frame.unit_cell();

  if (uc.has_value()) {
    const auto &unit_cell = uc.value();
    m_outfile << fmt::format("CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} "
                             "P 1           1",
                             unit_cell.a(), unit_cell.b(), unit_cell.c(),
                             trajan::units::degrees(unit_cell.alpha()),
                             trajan::units::degrees(unit_cell.beta()),
                             trajan::units::degrees(unit_cell.gamma()))
              << std::endl;
  }

  int i = 1;
  for (const auto &atom : atoms) {
    std::string line =
        fmt::format(PDB_LINE_FMT_WRITE.data(),
                    "ATOM",                 // field 1: 6 chars
                    i,                      // field 2: 5 digits
                    ' ',                    // field 3: 1 char
                    atom.type,              // field 4: 4 chars
                    ' ',                    // field 5: 1 char
                    "RES",                  // field 6: 3 chars (residue name)
                    ' ',                    // field 7: 1 char
                    'A',                    // field 8: 1 char (chain ID)
                    1,                      // field 9: 4 digits (resSeq)
                    ' ',                    // field 10: 1 char (iCode)
                    "",                     // field 11: 3 chars (altLoc?)
                    atom.x, atom.y, atom.z, // 3 coordinates, 8.3f
                    1.0,                    // occupancy
                    0.0,                    // tempFactor
                    "",                     // segment ID (10 chars)
                    atom.element.symbol(),  // 2 chars
                    ""                      // charge (2 chars)
        );
    trajan::log::debug("Writing PDB line: {}", line);
    m_outfile << line << std::endl;
    i++;
  }

  m_outfile << "ENDMDL" << std::endl;

  return true;
}

}; // namespace trajan::io
