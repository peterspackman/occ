#pragma once
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
    int residue_number;
    double position[3];
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
    std::optional<occ::crystal::Crystal> parse_crystal(const std::string &);

    static bool is_likely_cif_filename(const std::string &);

  private:
    void extract_atom_sites(const gemmi::cif::Loop &);
    void extract_cell_parameter(const gemmi::cif::Pair &);
    void extract_symmetry_operations(const gemmi::cif::Loop &);
    void extract_symmetry_data(const gemmi::cif::Pair &);
    std::string m_failure_desc;
    std::vector<AtomData> m_atoms;
    SymmetryData m_sym;
    CellData m_cell;
};

} // namespace occ::io
