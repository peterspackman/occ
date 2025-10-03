#include <algorithm>
#include <memory>
#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/log.h>
#include <trajan/core/units.h>
#include <trajan/io/dcd.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/pdb.h>
#include <trajan/io/xtc.h>
#include <trajan/io/xyz.h>
#include <unordered_map>

namespace trajan::io {

namespace fs = std::filesystem;

// using occ::crystal::UnitCell;

static const std::unordered_map<std::string, std::function<FileHandlerPtr()>>
    handler_map = {{".pdb", []() { return std::make_unique<PDBHandler>(); }},
                   {".dcd", []() { return std::make_unique<DCDHandler>(); }},
{ ".xyz", []() { return std::make_unique<XYZHandler>(); }}, 
                   {".xtc", []() { return std::make_unique<XTCHandler>(); }}};

bool FileHandler::initialise(Mode mode) {
  m_mode = mode;
  if (m_initialised) {
    return true;
  }
  trajan::log::debug(fmt::format("Parsing file '{}'", this->file_name()));
  m_initialised = this->_initialise();
  if (!m_initialised) {
    throw std::runtime_error(fmt::format(
        "Unable to open file for {}: '{}'",
        (mode == Mode::Read) ? "reading" : "writing", this->file_name()));
  }
  trajan::log::debug(
      fmt::format("Successfully parsed file '{}'", this->file_name()));
  return m_initialised;
}

void FileHandler::finalise() { return this->_finalise(); }

bool FileHandler::read_frame(core::Frame &frame) {
  // FIXME: removing validate_frame method
  bool has_more_frames = this->read_next_frame(frame);
  if (!has_more_frames) {
    return false;
  }
  // return this->validate_frame(frame);
  return true;
}

bool FileHandler::write_frame(const core::Frame &frame) {
  return this->write_next_frame(frame);
}

// bool FileHandler::validate_frame(core::Frame &frame) {
//   std::vector<core::Atom> atoms = frame.atoms();
//   if (atoms.empty()) {
//     throw std::runtime_error("No atoms found.");
//   }
//   trajan::log::debug("Found {} atoms for frame {}", atoms.size(),
//                      frame.index());
//   UnitCell uc = frame.unit_cell();
//   Mat3N cart_pos = frame.cart_pos();
//   if (uc.volume() == 0.0) {
//     trajan::log::debug(
//         "Found no real unit cell for frame {}. Creating a dummy unit cell.",
//         frame.index());
//     // create dummy unit cell for neighlist
//     Vec3 min_vals = cart_pos.rowwise().minCoeff();
//     Vec3 max_vals = cart_pos.rowwise().maxCoeff();
//     Vec3 max_dims = max_vals - min_vals;
//     UnitCell dummy_uc = core::dummy_cell(max_dims[0], max_dims[1],
//     max_dims[2]); frame.set_uc(dummy_uc); Mat3N shifted_cart_pos =
//     cart_pos.colwise() - min_vals; Mat3N frac_pos =
//     dummy_uc.to_fractional(shifted_cart_pos); frame.set_frac_pos(frac_pos);
//     frame.set_wrapped_cart_pos(cart_pos);
//     trajan::log::debug(
//         "  Unit cell: {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f}",
//         dummy_uc.a(), dummy_uc.b(), dummy_uc.c(),
//         units::degrees(dummy_uc.alpha()), units::degrees(dummy_uc.beta()),
//         units::degrees(dummy_uc.gamma()));
//     return true;
//   }
//   trajan::log::debug("Found a real unit cell for frame {}.", frame.index());
//   Mat3N frac_pos = uc.to_fractional(cart_pos);
//   frac_pos = frac_pos.array() - frac_pos.array().floor();
//   frame.set_frac_pos(frac_pos);
//   Mat3N wrapped_cart_pos = uc.to_cartesian(frac_pos);
//   frame.set_wrapped_cart_pos(wrapped_cart_pos);
//   trajan::log::debug(
//       "  Unit cell: {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f} {:8.6f}", uc.a(),
//       uc.b(), uc.c(), units::degrees(uc.alpha()), units::degrees(uc.beta()),
//       units::degrees(uc.gamma()));
//   return true;
// }

void check_handlers(std::vector<FileHandlerPtr> &handlers) {
  std::unordered_map<FileType, int> file_type_counts;

  for (const auto &handler : handlers) {
    file_type_counts[handler->file_type()]++;
  }
  if (file_type_counts.count(FileType::PDB) > 1) {
    trajan::log::warn("Multiple PDB files read. Using last.");
    auto it = std::find_if(handlers.begin(), handlers.end(),
                           [](const FileHandlerPtr &handler) {
                             return handler->file_type() == FileType::PDB;
                           });
    if (it != handlers.end()) {
      handlers.erase(it);
      file_type_counts[FileType::PDB]--;
    }
  }
  if (file_type_counts.count(FileType::PDB) == 0 &&
      file_type_counts.count(FileType::DCD) > 0) {
    trajan::log::warn("DCD(s) loaded without a topology file.");
  }
  if (file_type_counts.count(FileType::PDB) == 0 &&
      file_type_counts.count(FileType::DCD) == 0) {
    throw std::runtime_error("No files!");
  }
};

FileHandlerPtr get_handler(std::string ext) {
  auto it = handler_map.find(ext);
  if (it != handler_map.end()) {
    FileHandlerPtr handler = it->second();
    trajan::log::debug(fmt::format("Recognised '{}' file extension", ext));
    return handler;
  } else {
    throw std::runtime_error(fmt::format("Unknown file extension: '{}'", ext));
  }
};

FileHandlerPtr read_input_file(const fs::path &file) {
  std::string ext = file.extension().string();
  trajan::log::debug("Attempting to read input from {}, file extension = {}",
                     file.generic_string(), ext);
  if (!fs::exists(file)) {
    throw std::runtime_error(
        fmt::format("Input file does not exist: '{}'", file.generic_string()));
  };

  FileHandlerPtr handler = get_handler(ext);
  handler->set_file_path(file);

  return handler;
}

std::vector<FileHandlerPtr>
read_input_files(const std::vector<fs::path> &files) {
  std::vector<FileHandlerPtr> handlers;
  for (const fs::path &file : files) {
    handlers.push_back(read_input_file(file));
  }
  check_handlers(handlers);
  return handlers;
}

std::vector<FileHandlerPtr>
read_input_files(const std::vector<std::string> &filenames) {
  std::vector<fs::path> files;
  for (auto filename : filenames) {
    files.push_back(filename);
  }
  return read_input_files(files);
}

FileHandlerPtr write_output_file(const fs::path &file) {
  std::string ext = file.extension().string();
  trajan::log::debug("Attempting to write output to {}, file extension = {}",
                     file.generic_string(), ext);

  FileHandlerPtr handler = get_handler(ext);
  handler->set_file_path(file);

  return handler;
}

} // namespace trajan::io
