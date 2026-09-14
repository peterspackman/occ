#pragma once
#include <ankerl/unordered_dense.h>
#include <fstream>
#include <occ/core/linear_algebra.h>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace occ::crystal {
class Crystal;
class UnitCell;
} // namespace occ::crystal

namespace occ::io {

/**
 * \brief Unified reader/writer for SHELX .res/.ins crystallographic files
 *
 * The ShelxFile class provides functionality to read and write crystal
 * structures in the SHELX format, which is widely used in crystallography.
 *
 * SHELX files contain crystallographic data including:
 * - Unit cell parameters (CELL line)
 * - Lattice type and centrosymmetry (LATT line)
 * - Symmetry operations (SYMM lines)
 * - Scattering factor types (SFAC line)
 * - Atomic coordinates and site occupancies
 *
 * The class handles both .res (result) and .ins (instruction) file extensions
 * and provides round-trip compatibility for reading and writing structures.
 *
 * \par Example Usage:
 * \code
 * occ::io::ShelxFile shelx;
 * shelx.set_title("My Crystal Structure");
 * shelx.set_wavelength(1.54178); // Cu Kα
 *
 * // Write crystal to file
 * if (shelx.write_crystal_to_file(crystal, "structure.res")) {
 *     std::cout << "Successfully wrote SHELX file" << std::endl;
 * }
 *
 * // Read crystal from file
 * auto result = shelx.read_crystal_from_file("structure.res");
 * if (result.has_value()) {
 *     auto crystal = result.value();
 *     // Use crystal...
 * }
 * \endcode
 */
class ShelxFile {
public:
  /**
   * \brief Default constructor
   *
   * Creates a ShelxFile instance with default settings:
   * - Title: "Crystal structure"
   * - Wavelength: 1.54178 Å (Cu Kα)
   */
  ShelxFile() = default;

  /**
   * \brief Read a crystal structure from a SHELX file
   *
   * Parses a SHELX .res or .ins file and constructs a Crystal object
   * from the contained crystallographic data.
   *
   * \param filename Path to the SHELX file to read
   * \return Optional Crystal object, nullopt if parsing failed
   * \note Use error_message() to get details if parsing fails
   */
  std::optional<occ::crystal::Crystal>
  read_crystal_from_file(const std::string &filename);

  /**
   * \brief Read a crystal structure from a SHELX-formatted string
   *
   * Parses SHELX format data directly from a string buffer.
   *
   * \param contents String containing SHELX-formatted data
   * \return Optional Crystal object, nullopt if parsing failed
   * \note Use error_message() to get details if parsing fails
   */
  std::optional<occ::crystal::Crystal>
  read_crystal_from_string(const std::string &contents);

  /**
   * \brief Write a crystal structure to a SHELX file
   *
   * Writes the crystal structure in SHELX format to the specified file.
   * The output includes all necessary SHELX keywords and formatting.
   *
   * \param crystal Crystal structure to write
   * \param filename Path where the SHELX file should be written
   * \return true if successful, false otherwise
   * \note Use error_message() to get details if writing fails
   */
  bool write_crystal_to_file(const occ::crystal::Crystal &crystal,
                             const std::string &filename);

  /**
   * \brief Write a crystal structure to a SHELX-formatted string
   *
   * Converts the crystal structure to SHELX format and returns as string.
   *
   * \param crystal Crystal structure to convert
   * \return String containing SHELX-formatted data
   */
  std::string write_crystal_to_string(const occ::crystal::Crystal &crystal);

  /**
   * \brief Write a crystal structure to an output stream
   *
   * Writes the crystal structure in SHELX format to the provided stream.
   *
   * \param crystal Crystal structure to write
   * \param stream Output stream to write to
   */
  void write_crystal_to_stream(const occ::crystal::Crystal &crystal,
                               std::ostream &stream);

  /**
   * \brief Check if a filename appears to be a SHELX file
   *
   * Examines the file extension to determine if it's likely a SHELX file.
   *
   * \param filename File path to check
   * \return true if extension is .res or .ins
   */
  static bool is_likely_shelx_filename(const std::string &filename);

  /**
   * \brief Set the title/comment for SHELX output
   *
   * Sets the title that will be written to the TITL line in SHELX files.
   *
   * \param title Title string (will be written after "TITL ")
   */
  void set_title(const std::string &title) { m_title = title; }

  /**
   * \brief Set the X-ray wavelength for SHELX output
   *
   * Sets the wavelength that will be written to the CELL line.
   * Common values: 1.54178 Å (Cu Kα), 0.71073 Å (Mo Kα)
   *
   * \param wavelength Wavelength in Angstroms (default: 1.54178 Å)
   */
  void set_wavelength(double wavelength) { m_wavelength = wavelength; }

  /**
   * \brief Get the last error message
   *
   * Returns a description of the last parsing or writing error that occurred.
   *
   * \return Error message string, empty if no error
   */
  const std::string &error_message() const { return m_error_message; }

private:
  // Data structures for parsed SHELX data
  struct AtomData {
    std::string label;
    std::string element;
    int sfac_index{0};
    double x{0.0}, y{0.0}, z{0.0};
    double occupation{1.0};
    /// Displacement parameters in the u_cif convention,
    /// (u11, u22, u33, u12, u13, u23). An isotropic atom carries U on the
    /// diagonal alone until the cell is known and the off-diagonal cosines can
    /// be filled in.
    Vec6 adp{Vec6::Zero()};
    bool anisotropic{false};
    /// SHELX lets an atom -- almost always a hydrogen -- ride on its parent,
    /// with U_iso a fixed multiple of the parent's U_eq rather than a
    /// parameter of its own. Resolved once the cell is available.
    double u_eq_multiple{0.0};
    int u_eq_pivot{-1};
  };

  struct CellData {
    double wavelength = 1.54178;
    double a, b, c;
    double alpha, beta, gamma;
  };

  struct SymmetryData {
    int latt = 1;
    std::vector<std::string> symops;
  };

  enum class LineType {
    Title,
    Cell,
    Latt,
    Sfac,
    Symm,
    Fvar,
    Part,
    Atom,
    /// An electron density peak from a difference map: shaped like an atom, but
    /// not one
    Peak,
    End,
    Ignored
  };

  // Reading methods
  /// Strip comments and join SHELX's `=` continuation lines, so each element of
  /// the result is one complete instruction or atom.
  static std::vector<std::string> logical_lines(const std::string &contents);
  LineType classify_line(const std::string &line) const;
  void parse_title_line(const std::string &line);
  void parse_cell_line(const std::string &line);
  void parse_latt_line(const std::string &line);
  void parse_sfac_line(const std::string &line);
  void parse_symm_line(const std::string &line);
  void parse_fvar_line(const std::string &line);
  void parse_part_line(const std::string &line);
  void parse_atom_line(const std::string &line);
  /// Resolve riding U values and fill in the isotropic off-diagonals, both of
  /// which need the unit cell.
  void resolve_displacement_parameters(const occ::crystal::UnitCell &cell);

  /**
   * \brief Decode a SHELX refinable parameter.
   *
   * Any parameter may be tied to a free variable rather than given outright,
   * encoded as `10 m + p`: `m` of 0 or 1 means the value is `p` as written,
   * `m >= 2` means `p` times free variable `m`, and `m <= -2` means
   * `p (fv_|m| - 1)`, which is how the two halves of a disordered site are
   * made to sum to one.
   */
  double decode_variable(double coded) const;

  // Writing methods
  void write_title_line(std::ostream &stream);
  void write_cell_line(const occ::crystal::Crystal &crystal,
                       std::ostream &stream);
  void write_latt_line(const occ::crystal::Crystal &crystal,
                       std::ostream &stream);
  void write_symm_lines(const occ::crystal::Crystal &crystal,
                        std::ostream &stream);
  void write_sfac_line(const occ::crystal::Crystal &crystal,
                       std::ostream &stream);
  void write_atom_lines(const occ::crystal::Crystal &crystal,
                        std::ostream &stream);
  void write_end_line(std::ostream &stream);

  // Utility methods
  int determine_latt_type(const occ::crystal::Crystal &crystal);
  std::vector<std::string>
  get_unique_elements(const occ::crystal::Crystal &crystal);
  bool cell_valid() const;
  size_t num_atoms() const { return m_atoms.size(); }
  void clear_data();

  // Member data
  std::string m_title{"Crystal structure"};
  double m_wavelength{1.54178}; // Cu Kα wavelength
  CellData m_cell;
  SymmetryData m_sym;
  std::vector<std::string> m_sfac;
  std::vector<AtomData> m_atoms;
  /// FVAR values, with the overall scale factor first
  std::vector<double> m_free_variables;
  /// The disorder component atoms currently belong to, and its occupancy when
  /// the PART instruction supplied one
  int m_part_number{0};
  std::optional<double> m_part_occupancy;
  /// The most recent atom whose U was its own, which is what a riding atom
  /// takes its U_eq from
  int m_u_eq_pivot{-1};
  std::string m_error_message;

  /// Instruction names, so a line can be told from an atom by something better
  /// than the shape of its arguments.
  static const ankerl::unordered_dense::set<std::string> m_instructions;
};

} // namespace occ::io
