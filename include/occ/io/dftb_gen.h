#pragma once
#include <occ/crystal/crystal.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <istream>

namespace occ::io {

class DftbGenFormat {
public:

    DftbGenFormat();

    void parse(const std::string &);
    void parse(std::istream &);

    void write(const std::string &);
    void write(std::ostream &);

    inline size_t num_atoms() const { return m_positions.cols(); }

    const IVec &atomic_numbers() const;
    const Mat3N &positions() const;

    bool is_periodic() const;

    std::optional<occ::crystal::Crystal> crystal() const;
    std::optional<occ::core::Molecule> molecule() const;

    void set_molecule(const occ::core::Molecule &);
    void set_crystal(const occ::crystal::Crystal &);

    static bool is_likely_gen_filename(const std::string &);

private:
    char format_character() const;
    
    void build_symbol_mapping();

    std::vector<std::string> m_symbols;
    Mat3N m_positions;
    IVec m_atomic_numbers;
    IVec m_symbol_index;
    Vec3 m_origin;
    Mat3 m_lattice;
    bool m_periodic{true};
    bool m_fractional{true};
};

}
