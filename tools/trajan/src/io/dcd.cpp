// #include "trajan/core/unit_cell.h"
#include "trajan/core/util.h"
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/crystal/unitcell.h>
#include <trajan/io/dcd.h>

namespace trajan::io {

using occ::Vec3;
using occ::crystal::UnitCell;
using trajan::core::Frame;

bool DCDHandler::_initialise() {
  if (m_mode == Mode::Read) {
    m_infile.open(this->file_path(), std::ios::binary);
    if (!m_infile.is_open()) {
      return false;
    }
    return this->parse_dcd_header();
  } else {
    m_outfile.open(this->file_path(), std::ios::binary);
    return m_outfile.is_open();
  }
}

void DCDHandler::_finalise() {
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

bool DCDHandler::read_next_frame(Frame &frame) {
  if (m_current_frame >= m_total_frames) {
    return false;
  }

  bool success = this->parse_dcd(frame);
  if (success) {
    m_current_frame++;
  }

  return success;
}

bool DCDHandler::write_next_frame(const Frame &frame) {
  if (m_current_frame == 0) {
    if (!write_dcd_header(frame)) {
      return false;
    }
  }

  if (!write_dcd_frame(frame)) {
    return false;
  }

  std::streampos current_pos = m_outfile.tellp();

  m_outfile.seekp(8, std::ios::beg);

  if (!write_binary(static_cast<int32_t>(m_current_frame + 1))) {
    return false;
  }

  m_outfile.seekp(current_pos);

  m_outfile.flush();

  m_current_frame++;

  return true;
}

template <typename T> bool DCDHandler::read_binary(T &value) {
  m_infile.read(reinterpret_cast<char *>(&value), sizeof(T));
  return m_infile.good();
}

template <typename T> bool DCDHandler::write_binary(const T &value) {
  m_outfile.write(reinterpret_cast<const char *>(&value), sizeof(T));
  return m_outfile.good();
}

bool DCDHandler::read_fortran_record(std::vector<char> &buffer) {
  int32_t record_length;
  if (!read_binary(record_length)) {
    return false;
  }

  buffer.resize(record_length);
  m_infile.read(buffer.data(), record_length);
  if (!m_infile.good()) {
    return false;
  }

  int32_t trailing_length;
  if (!read_binary(trailing_length)) {
    return false;
  }

  return (record_length == trailing_length);
}

bool DCDHandler::write_fortran_record(const std::vector<char> &buffer) {
  int32_t record_length = static_cast<int32_t>(buffer.size());

  if (!write_binary(record_length)) {
    return false;
  }

  m_outfile.write(buffer.data(), record_length);
  if (!m_outfile.good()) {
    return false;
  }

  if (!write_binary(record_length)) {
    return false;
  }

  return true;
}

bool DCDHandler::skip_fortran_record() {
  int32_t record_length;
  if (!read_binary(record_length)) {
    return false;
  }

  m_infile.seekg(record_length, std::ios::cur);
  if (!m_infile.good()) {
    return false;
  }

  int32_t trailing_length;
  return read_binary(trailing_length) && (record_length == trailing_length);
}

bool DCDHandler::_parse_dcd_header() {
  // First record: Main header (84 bytes)
  std::vector<char> header_buffer;
  if (!read_fortran_record(header_buffer)) {
    throw std::runtime_error("Failed to read DCD header record");
  }

  if (header_buffer.size() < 84) {
    throw std::runtime_error("DCD header record too small");
  }

  // Check signature
  char signature[4];
  std::memcpy(signature, header_buffer.data(), 4);
  if (std::strncmp(signature, "CORD", 4) != 0) {
    throw std::runtime_error("Invalid DCD signature");
  }

  // Parse header fields
  int32_t *int_data = reinterpret_cast<int32_t *>(header_buffer.data());
  trajan::log::debug("DCD Header Integer Fields");
  for (int i = 0; i < 21; ++i) {
    trajan::log::debug("int_data[{}] = {} (0x{:08x}) [byte {}]", i, int_data[i],
                       static_cast<uint32_t>(int_data[i]), i * 4);
  }

  m_total_frames = static_cast<size_t>(int_data[1]); // NFILE
  m_total_frames = static_cast<size_t>(int_data[1]); // NFILE
  int32_t first_frame = int_data[2];                 // NPRIV
  int32_t frame_skip = int_data[3];                  // NSAVC
  int32_t total_steps = int_data[4];                 // NSTEP
  int32_t namnf = int_data[9]; // NAMNF (fixed atoms) - byte 36
  m_timestep = static_cast<double>(int_data[10]);

  trajan::log::debug("DCD Header Parsed Fields");
  trajan::log::debug("NFILE (total frames): {}", m_total_frames);
  trajan::log::debug("ISTART (first frame): {}", first_frame);
  trajan::log::debug("NSAVC (frame skip): {}", frame_skip);
  trajan::log::debug("NSTEP (total steps): {}", total_steps);
  trajan::log::debug("NAMNF (fixed atoms): {}", namnf);

  // Parse delta (timestep) - different format for CHARMM vs X-PLOR
  double delta;
  if (int_data[20] != 0) { // CHARMM version field
    m_is_charmm_format = true;
    m_has_extra_block = (int_data[11] != 0); // Position 40 in bytes / 4
    m_has_4d_coords = (int_data[12] == 1);   // Position 44 in bytes / 4

    // CHARMM stores delta as float
    delta = static_cast<double>(int_data[10]); // Position 36 in bytes / 4
    trajan::log::debug("CHARMM version field: {}", int_data[20]);
    trajan::log::debug("Extra block flag (unit cell flag): {}", int_data[11]);
    trajan::log::debug("4D coords flag: {}", int_data[12]);
    trajan::log::debug("Delta (timestep): {}", delta);
  } else {
    m_is_charmm_format = false;
    m_has_extra_block = false;
    m_has_4d_coords = false;

    // X-PLOR stores delta as double
    delta = static_cast<double>(int_data[10]); // Position 36 in bytes / 4
    trajan::log::debug("X-PLOR Format");
    trajan::log::debug("Delta as double (byte 40): {}", delta);
  }

  trajan::log::debug("Summary");
  trajan::log::debug("Total frames: {}", m_total_frames);
  trajan::log::debug("CHARMM format: {}", m_is_charmm_format ? "Yes" : "No");
  trajan::log::debug("Has extra block: {}", m_has_extra_block ? "Yes" : "No");
  trajan::log::debug("Has 4D coords: {}", m_has_4d_coords ? "Yes" : "No");
  trajan::log::debug("Delta: {}", delta);
  trajan::log::debug("Fixed atoms (NAMNF): {}", namnf);

  // Check for potential VMD memory allocation issues
  if (namnf < 0 || namnf > static_cast<int32_t>(m_total_frames * 10000)) {
    trajan::log::warn("WARNING: NAMNF value {} seems suspicious - this could "
                      "cause VMD memory issues",
                      namnf);
  }
  if (m_total_frames > 1000000) {
    trajan::log::warn(
        "WARNING: Frame count {} is very large - check if this is correct",
        m_total_frames);
  }

  // Second record: Title information
  std::vector<char> title_buffer;
  if (!read_fortran_record(title_buffer)) {
    throw std::runtime_error("Failed to read DCD title record");
  }

  if (title_buffer.size() >= 4) {
    int32_t num_titles = *reinterpret_cast<int32_t *>(title_buffer.data());
    trajan::log::debug("Number of titles: {}", num_titles);
    trajan::log::debug("Title buffer size: {} bytes", title_buffer.size());

    if (num_titles < 0 || num_titles > 1000) {
      trajan::log::warn("WARNING: Suspicious title count: {}", num_titles);
    }
  }

  // Third record: Number of atoms
  std::vector<char> atom_buffer;
  if (!read_fortran_record(atom_buffer)) {
    throw std::runtime_error("Failed to read DCD atom count record");
  }

  if (atom_buffer.size() < 4) {
    throw std::runtime_error("DCD atom count record too small");
  }

  m_num_atoms =
      static_cast<size_t>(*reinterpret_cast<int32_t *>(atom_buffer.data()));
  trajan::log::debug("Number of atoms: {}", m_num_atoms);

  // Allocate coordinate buffers
  m_x_coords.resize(m_num_atoms);
  m_y_coords.resize(m_num_atoms);
  m_z_coords.resize(m_num_atoms);

  return true;
}

bool DCDHandler::parse_dcd_header() {
  try {
    bool success = this->_parse_dcd_header();
    if (success) {
      trajan::log::debug("Successfully parsed DCD header");
    }
    return success;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        fmt::format("Exception while parsing DCD header: {}", e.what()));
  }
}

bool DCDHandler::_parse_dcd(core::Frame &frame) {
  // Read CHARMM extra block (unit cell) if present
  if (m_is_charmm_format && m_has_extra_block) {
    std::vector<char> unitcell_buffer;
    if (!read_fortran_record(unitcell_buffer)) {
      throw std::runtime_error("Failed to read unit cell record");
    }

    // Parse unit cell data (6 doubles: a, b, c, alpha, beta, gamma)
    if (unitcell_buffer.size() >= 48) { // 6 * 8 bytes
      double *cell_data = reinterpret_cast<double *>(unitcell_buffer.data());

      double a = cell_data[0];
      double b = cell_data[2];
      double c = cell_data[5];
      double alpha = cell_data[4];
      double beta = cell_data[3];
      double gamma = cell_data[1];
      trajan::log::debug(
          "Raw DCD Unit Cell: {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f}",
          a, b, c, alpha, beta, gamma);
      // Check if angles are stored as cosines (CHARMM/NAMD format)
      if (alpha >= -1.0 && alpha <= 1.0 && beta >= -1.0 && beta <= 1.0 &&
          gamma >= -1.0 && gamma <= 1.0) {
        // Angles stored as cosines - convert to radians
        alpha = std::acos(alpha);
        beta = std::acos(beta);
        gamma = std::acos(gamma);
      } else {
        alpha = occ::units::radians(alpha);
        beta = occ::units::radians(beta);
        gamma = occ::units::radians(gamma);
      }
      if (trajan::util::unitcell_is_reasonable(a, b, c, alpha, beta, gamma)) {
        UnitCell uc = occ::crystal::triclinic_cell(a, b, c, alpha, beta, gamma);
        frame.set_unit_cell(uc);
      }
    }
  }

  // Read X coordinates
  std::vector<char> coord_buffer;
  if (!read_fortran_record(coord_buffer)) {
    throw std::runtime_error("Failed to read X coordinates");
  }

  if (coord_buffer.size() != m_num_atoms * sizeof(float)) {
    throw std::runtime_error("X coordinate record size mismatch");
  }
  std::memcpy(m_x_coords.data(), coord_buffer.data(), coord_buffer.size());

  // Read Y coordinates
  if (!read_fortran_record(coord_buffer)) {
    throw std::runtime_error("Failed to read Y coordinates");
  }

  if (coord_buffer.size() != m_num_atoms * sizeof(float)) {
    throw std::runtime_error("Y coordinate record size mismatch");
  }
  std::memcpy(m_y_coords.data(), coord_buffer.data(), coord_buffer.size());

  // Read Z coordinates
  if (!read_fortran_record(coord_buffer)) {
    throw std::runtime_error("Failed to read Z coordinates");
  }

  if (coord_buffer.size() != m_num_atoms * sizeof(float)) {
    throw std::runtime_error("Z coordinate record size mismatch");
  }
  std::memcpy(m_z_coords.data(), coord_buffer.data(), coord_buffer.size());

  // Skip 4th dimension if present
  if (m_is_charmm_format && m_has_4d_coords) {
    if (!skip_fortran_record()) {
      throw std::runtime_error("Failed to skip 4D coordinates");
    }
  }

  // Update frame with coordinates
  for (size_t i = 0; i < m_num_atoms; ++i) {
    Vec3 pos = {m_x_coords[i], m_y_coords[i], m_z_coords[i]};
    frame.update_atom_position(i, pos);
  }

  return true;
}

bool DCDHandler::parse_dcd(Frame &frame) {
  try {
    return this->_parse_dcd(frame);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        fmt::format("Exception while parsing DCD frame: {}", e.what()));
    return false;
  }
}

bool DCDHandler::write_dcd_header(const Frame &frame) {
  m_num_atoms = frame.num_atoms();

  // Create 84-byte main header
  std::vector<char> header_buffer(84, 0);
  int32_t *int_data = reinterpret_cast<int32_t *>(header_buffer.data());

  // Set signature
  std::memcpy(&int_data[0], "CORD", 4);

  // Set header fields
  int_data[1] = 0; // NFILE (will be updated as frames are written)
  int_data[2] = 0; // ISTART
  int_data[3] = 1; // NSAVC
  int_data[4] = 0; // NSTEP (total steps)

  // Initialize remaining fields to 0
  for (int i = 5; i <= 19; ++i) {
    int_data[i] = 0;
  }

  // Set CHARMM version to indicate CHARMM format
  int_data[20] = 24; // CHARMM version

  // Timestep
  int_data[10] = static_cast<int32_t>(1.0f);

  // Set unit cell flag
  int_data[11] = frame.has_unit_cell() ? 1 : 0; // Has unit cell data

  if (!write_fortran_record(header_buffer)) {
    return false;
  }

  // Write title record
  std::vector<char> title_buffer(84, 0);
  int32_t num_titles = 1;
  std::memcpy(title_buffer.data(), &num_titles, sizeof(int32_t));
  std::string title = "Created by TrajAn";
  std::memcpy(title_buffer.data() + 4, title.c_str(),
              std::min(title.size(), size_t(79)));

  if (!write_fortran_record(title_buffer)) {
    return false;
  }

  // Write atom count record
  std::vector<char> atom_count_buffer(4);
  int32_t num_atoms = static_cast<int32_t>(m_num_atoms);
  std::memcpy(atom_count_buffer.data(), &num_atoms, sizeof(int32_t));

  return write_fortran_record(atom_count_buffer);
}

bool DCDHandler::write_unit_cell_data(const Frame &frame) {
  const auto &uc = frame.unit_cell();

  std::vector<char> unit_cell_buffer(48, 0); // 6 doubles = 48 bytes
  double *cell_data = reinterpret_cast<double *>(unit_cell_buffer.data());

  // Store unit cell parameters
  if (uc.has_value()) {
    cell_data[0] = uc->a();               // a
    cell_data[1] = std::cos(uc->gamma()); // cos(gamma) - angle between A and B
    cell_data[2] = uc->b();               // b
    cell_data[3] = std::cos(uc->beta());  // cos(beta) - angle between A and C
    cell_data[4] = std::cos(uc->alpha()); // cos(alpha) - angle between B and C
    cell_data[5] = uc->c();               // c
  } else {
    cell_data[0] = 0.0;
    cell_data[1] = 0.0;
    cell_data[2] = 0.0;
    cell_data[3] = 0.0;
    cell_data[4] = 0.0;
    cell_data[5] = 0.0;
  }

  return write_fortran_record(unit_cell_buffer);
}

bool DCDHandler::write_dcd_frame(const Frame &frame) {
  if (!write_unit_cell_data(frame)) {
    return false;
  }

  const auto &atoms = frame.atoms();
  m_x_coords.resize(m_num_atoms);
  m_y_coords.resize(m_num_atoms);
  m_z_coords.resize(m_num_atoms);

  for (size_t i = 0; i < m_num_atoms; ++i) {
    m_x_coords[i] = static_cast<float>(atoms[i].x);
    m_y_coords[i] = static_cast<float>(atoms[i].y);
    m_z_coords[i] = static_cast<float>(atoms[i].z);
  }

  std::vector<char> coord_buffer(m_num_atoms * sizeof(float));

  std::memcpy(coord_buffer.data(), m_x_coords.data(), coord_buffer.size());
  if (!write_fortran_record(coord_buffer)) {
    return false;
  }

  std::memcpy(coord_buffer.data(), m_y_coords.data(), coord_buffer.size());
  if (!write_fortran_record(coord_buffer)) {
    return false;
  }

  std::memcpy(coord_buffer.data(), m_z_coords.data(), coord_buffer.size());
  if (!write_fortran_record(coord_buffer)) {
    return false;
  }

  return true;
}

}; // namespace trajan::io
