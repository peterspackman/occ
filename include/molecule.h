#pragma once
#include "element.h"
#include "linear_algebra.h"
#include <libint2/atom.h>

namespace craso::chem {
    using craso::IVec;
    using craso::Mat3N;

class Molecule
{
public:
    Molecule(const IVec&, const Mat3N&);
    Molecule(const std::vector<libint2::Atom>& atoms);
    size_t size() const { return m_atomicNumbers.size(); }

    void set_name(const std::string&);
    const std::string& name() const { return m_name; }

    const Mat3N& positions() const { return m_positions; }
    const IVec& atomic_numbers() const {return m_atomicNumbers;}

    void add_bond(size_t l, size_t r) { m_bonds.push_back({l, r}); }
    void set_bonds(const std::vector<std::pair<size_t, size_t>>& bonds) { m_bonds = bonds; }
    
    const std::vector<std::pair<size_t, size_t>>& bonds() const { return m_bonds; }
    const auto& atoms() const { return m_atoms; }
    int num_electrons() const { return m_atomicNumbers.sum() - m_charge; }
    void set_unit_cell_idx(const IVec& idx) { m_uc_idx = idx; }
    void set_asymmetric_unit_idx(const IVec& idx) { m_asym_idx = idx; }
private:
    int m_charge{0};
    int m_multiplicity{1};
    std::string m_name{""};
    std::vector<libint2::Atom> m_atoms;
    IVec m_atomicNumbers;
    Mat3N m_positions;
    IVec m_uc_idx;
    IVec m_asym_idx;
    std::vector<std::pair<size_t, size_t>> m_bonds;
    std::vector<Element> m_elements;
};


Molecule read_xyz_file(const std::string&);

}
