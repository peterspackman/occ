#pragma once
#include <ankerl/unordered_dense.h>
#include <gemmi/cif.hpp>
#include <iostream>
#include <map>
#include <occ/crystal/crystal.h>
#include <occ/io/occ_input.h>
#include <regex>
#include <string>

namespace occ::io {

struct AtomData {
  std::string element;
  std::string site_label;
  std::string residue_name;
  std::string chain_id;
  std::string adp_type;
  int residue_number;
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double uiso{0.0};
};

struct AdpData {
  std::string aniso_label;
  double u11{0.0};
  double u22{0.0};
  double u33{0.0};
  double u12{0.0};
  double u13{0.0};
  double u23{0.0};
};

struct CellData {
  double a{-1}, b{-1}, c{-1};
  double alpha{-1}, beta{-1}, gamma{-1};
  bool valid() const {
    if (a <= 0)
      return false;
    if (b <= 0)
      return false;
    if (c <= 0)
      return false;
    if (alpha <= 0)
      return false;
    if (beta <= 0)
      return false;
    if (gamma <= 0)
      return false;
    return true;
  }
};

struct SymmetryData {
  int number{-1};
  std::string nameHM{"Not set"};
  std::string nameHall{"Not set"};
  std::vector<std::string> symops;
  size_t num_symops() const { return symops.size(); }
  bool valid() const {
    if (number > 0)
      return true;
    if (nameHM != "Not set")
      return true;
    if (nameHall != "Not set")
      return true;
    if (num_symops() > 0)
      return true;
    return false;
  }
};

class CifParser {
public:
  CifParser();
  const std::vector<AtomData> &atom_data() const { return m_atoms; }
  const SymmetryData &symmetry_data() const { return m_sym; }
  const CellData &cell_data() const { return m_cell; }
  bool symmetry_valid() const { return m_sym.valid(); }
  bool cell_valid() const { return m_cell.valid(); }
  bool crystal_valid() const { return symmetry_valid() && cell_valid(); }
  const std::string &failure_description() const { return m_failure_desc; }
  size_t num_atoms() const { return m_atoms.size(); }
  std::optional<occ::crystal::Crystal>
  parse_crystal_from_file(const std::string &);
  std::optional<occ::crystal::Crystal>
  parse_crystal_from_string(const std::string &);
  std::optional<occ::crystal::Crystal>
  parse_crystal_from_document(const gemmi::cif::Document &);

  static bool is_likely_cif_filename(const std::string &);

private:
  enum class AtomField {
    Ignore,
    Label,
    Element,
    FracX,
    FracY,
    FracZ,
    AdpType,
    Uiso,
    AdpLabel,
    AdpU11,
    AdpU22,
    AdpU33,
    AdpU12,
    AdpU13,
    AdpU23
  };

  enum class CellField {
    Ignore,
    LengthA,
    LengthB,
    LengthC,
    AngleAlpha,
    AngleBeta,
    AngleGamma
  };

  enum class SymmetryField { Ignore, HallSymbol, HMSymbol, Number };

  void extract_atom_sites(const gemmi::cif::Loop &);
  void extract_cell_parameter(const gemmi::cif::Pair &);
  void extract_symmetry_operations(const gemmi::cif::Loop &);
  void extract_symmetry_data(const gemmi::cif::Pair &);
  std::string m_failure_desc;
  std::vector<AtomData> m_atoms;
  ankerl::unordered_dense::map<std::string, AdpData> m_adps;
  SymmetryData m_sym;
  CellData m_cell;

  static void set_atom_data(int index, const std::vector<AtomField> &fields,
                            const gemmi::cif::Loop &loop, AtomData &atom,
                            AdpData &adp);

  static const ankerl::unordered_dense::map<std::string, AtomField>
      m_known_atom_fields;

  static const ankerl::unordered_dense::map<std::string, CellField>
      m_known_cell_fields;

  static const ankerl::unordered_dense::map<std::string, SymmetryField>
      m_known_symmetry_fields;
};

} // namespace occ::io
