#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <tuple>

namespace occ::chem {

class Molecule {
  public:
    enum Origin { Cartesian, Centroid, CenterOfMass };
    using CellShift = std::array<int, 3>;
    Molecule(const IVec &, const Mat3N &);
    Molecule(const std::vector<core::Atom> &atoms);

    template <typename N, typename D>
    Molecule(const std::vector<N> &nums,
             const std::vector<std::array<D, 3>> &pos) {
        size_t num_atoms = std::min(nums.size(), pos.size());
        m_atomicNumbers = IVec(num_atoms);
        m_positions = Mat3N(3, num_atoms);
        for (size_t i = 0; i < num_atoms; i++) {
            m_atomicNumbers(i) = static_cast<int>(nums[i]);
            m_positions(0, i) = static_cast<double>(pos[i][0]);
            m_positions(1, i) = static_cast<double>(pos[i][1]);
            m_positions(2, i) = static_cast<double>(pos[i][2]);
        }
        for (size_t i = 0; i < size(); i++) {
            m_elements.push_back(Element(m_atomicNumbers(i)));
        }
        m_name = chemical_formula(m_elements);
    }

    size_t size() const { return m_atomicNumbers.size(); }

    void set_name(const std::string &);
    const std::string &name() const { return m_name; }

    Vec interatomic_distances() const;
    const auto &elements() const { return m_elements; }
    const Mat3N &positions() const { return m_positions; }
    const IVec &atomic_numbers() const { return m_atomicNumbers; }
    const Vec vdw_radii() const;
    const Vec atomic_masses() const;
    double molar_mass() const;
    std::vector<core::Atom> atoms() const;
    void set_cell_shift(const CellShift &);
    const CellShift &cell_shift() const;

    Vec3 centroid() const;
    Vec3 center_of_mass() const;
    Mat3 inertia_tensor() const;
    Vec3 principal_moments_of_inertia() const;
    Vec3 rotational_constants() const;
    double rotational_free_energy(double) const;
    double translational_free_energy(double) const;

    std::tuple<size_t, size_t, double> nearest_atom(const Molecule &) const;

    void add_bond(size_t l, size_t r) { m_bonds.push_back({l, r}); }
    void set_bonds(const std::vector<std::pair<size_t, size_t>> &bonds) {
        m_bonds = bonds;
    }

    const std::vector<std::pair<size_t, size_t>> &bonds() const {
        return m_bonds;
    }
    int charge() const { return m_charge; }
    int multiplicity() const { return m_multiplicity; }
    int num_electrons() const { return m_atomicNumbers.sum() - m_charge; }

    bool comparable_to(const Molecule &) const;

    void set_asymmetric_molecule_idx(size_t idx) { m_asym_mol_idx = idx; }
    void set_unit_cell_molecule_idx(size_t idx) { m_uc_mol_idx = idx; }
    int asymmetric_molecule_idx() const { return m_asym_mol_idx; }
    int unit_cell_molecule_idx() const { return m_uc_mol_idx; }

    void set_asymmetric_unit_transformation(const Mat3 &rot,
                                            const Vec3 &trans) {
        m_asymmetric_unit_rotation = rot;
        m_asymmetric_unit_translation = trans;
    }
    std::pair<Mat3, Vec3> asymmetric_unit_transformation() const {
        return {m_asymmetric_unit_rotation, m_asymmetric_unit_translation};
    }

    void set_unit_cell_idx(const IVec &idx) { m_uc_idx = idx; }
    void set_asymmetric_unit_idx(const IVec &idx) { m_asym_idx = idx; }
    void set_asymmetric_unit_symop(const IVec &symop);
    const auto &unit_cell_idx() const { return m_uc_idx; }
    const auto &asymmetric_unit_idx() const { return m_asym_idx; }
    const auto &asymmetric_unit_symop() const { return m_asym_symop; }

    void rotate(const Eigen::Affine3d &r, Origin o = Cartesian);
    void rotate(const Mat3 &r, Origin o = Cartesian);
    void rotate(const Mat3 &r, const Vec3 &o);
    void transform(const Mat4 &t, Origin o = Cartesian);
    void transform(const Mat4 &t, const Vec3 &o);
    void translate(const Vec3 &);
    Molecule rotated(const Eigen::Affine3d &r, Origin o = Cartesian) const;
    Molecule rotated(const Mat3 &r, Origin o = Cartesian) const;
    Molecule rotated(const Mat3 &r, const Vec3 &) const;
    Molecule transformed(const Mat4 &t, Origin o = Cartesian) const;
    Molecule transformed(const Mat4 &t, const Vec3 &) const;
    Molecule translated(const Vec3 &) const;

    bool equivalent_to(const Molecule &) const;

  private:
    int m_charge{0};
    int m_multiplicity{1};
    int m_asym_mol_idx{-1};
    int m_uc_mol_idx{-1};
    std::string m_name{""};
    std::vector<core::Atom> m_atoms;
    IVec m_atomicNumbers;
    Mat3N m_positions;
    IVec m_uc_idx;
    IVec m_asym_idx;
    IVec m_asym_symop;
    std::vector<std::pair<size_t, size_t>> m_bonds;
    std::vector<Element> m_elements;
    Mat3 m_asymmetric_unit_rotation = Mat3::Identity(3, 3);
    Vec3 m_asymmetric_unit_translation = Vec3::Zero(3);
    CellShift m_cell_shift{0, 0, 0};
};

Molecule read_xyz_file(const std::string &);

} // namespace occ::chem
