#pragma once
#include <Eigen/Dense>
#include "element.h"
#include <libint2/atom.h>

namespace craso::chem {

class Molecule
{
public:
    Molecule(const Eigen::VectorXi&, const Eigen::Matrix3Xd&);
    Molecule(const std::vector<libint2::Atom>& atoms);
    size_t size() const { return m_atomicNumbers.size(); }

    void set_name(const std::string&);
    const std::string& name() const { return m_name; }

    const Eigen::Matrix3Xd& positions() const { return m_positions; }
    const Eigen::VectorXi& atomic_numbers() const {return m_atomicNumbers;}

    void add_bond(size_t l, size_t r) { m_bonds.push_back({l, r}); }
    void set_bonds(const std::vector<std::pair<size_t, size_t>>& bonds) { m_bonds = bonds; }
    
    const std::vector<std::pair<size_t, size_t>>& bonds() const { return m_bonds; }
    const auto& atoms() const { return m_atoms; }
    int num_electrons() const { return m_atomicNumbers.sum() - m_charge; }
    void set_unit_cell_idx(const Eigen::VectorXi& idx) { m_uc_idx = idx; }
    void set_asymmetric_unit_idx(const Eigen::VectorXi& idx) { m_asym_idx = idx; }
private:
    int m_charge{0};
    int m_multiplicity{1};
    std::string m_name{""};
    std::vector<libint2::Atom> m_atoms;
    Eigen::VectorXi m_atomicNumbers;
    Eigen::Matrix3Xd m_positions;
    Eigen::VectorXi m_uc_idx;
    Eigen::VectorXi m_asym_idx;
    std::vector<std::pair<size_t, size_t>> m_bonds;
    std::vector<Element> m_elements;
};


Molecule read_xyz_file(const std::string&);

}
