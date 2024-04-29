#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <tuple>

namespace occ::core {

/**
 * Storage class for relevant information of a Molecule.
 *
 * The role of the Molecule class is to store basic information
 * about atoms which are bonded, and to facilitate convenient
 * calculations of properties of these atoms.
 */
class Molecule {
  public:
    /**
     * An enum to specify the origin used in some calculations
     * on the Molecule object.
     */
    enum Origin {
        Cartesian,   /**< The Cartesian origin i.e. (0, 0, 0) in R3 */
        Centroid,    /**< The molecular centroid i.e. average position of atoms
                        (ignoring mass) */
        CenterOfMass /**< The centre of mass i.e. the weighted average positon
                      */
    };

    using CellShift = std::array<int, 3>;

    inline explicit Molecule() {}

    /**
     * Construct a Molecule from a vector of atomic numbers and positions
     *
     * \param nums Vector of length N atomic numbers
     * \param pos Matrix (3, N) of atomic positions (Angstroms)
     */
    Molecule(const IVec &nums, const Mat3N &pos);

    /**
     * Construct a Molecule from a vector of Atom objects
     *
     * \param atoms std::vector<Atom> of length N atoms.
     *
     * Will convert from the expected Bohr unit of the Atom objects
     * internally to Angstroms.
     */
    Molecule(const std::vector<core::Atom> &atoms);

    /**
     * Construct a Molecule from a vector of Element objects, and
     * std::array<double, 3> positions
     *
     * \param els std::vector<Element> of length N Element objects.
     * \param pos std::vector<std::array<double, 3> > of length N of atomic
     * positions.
     *
     * Positions are assumed to be in Angstroms.
     */
    Molecule(const std::vector<Element> &els,
             const std::vector<std::array<double, 3>> &pos);

    /**
     * Construct a Molecule from a vector of atomic numbers a vector
     * arrays of positions
     *
     * \param nums std::vector<N> of atomic numbers, convertible to int
     * \param pos std::vector<std::array<D, 3> > of atomic
     * positions, convertible to double.
     *
     * Positions are assumed to be in Angstroms.
     */
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

    /**
     * The number of atoms in this molecule.
     *
     * \returns size size_t representing the number of atoms in this Molecule.
     *
     * Calculated based on the internal IVec of atomic numbers.
     */
    size_t size() const { return m_atomicNumbers.size(); }

    /**
     * Set the name for this molecule.
     *
     * \param name std::string representing the name/identifier for this
     * Molecule.
     *
     * See Molecule::name
     */
    inline void set_name(const std::string &name) { m_name = name; }

    /**
     * Get the name for this molecule.
     *
     * \returns name std::string representing the name/identifier for this
     * Molecule.
     *
     * See Molecule::set_name
     */
    inline const std::string &name() const { return m_name; }

    /**
     * A vector representing the compressed distance matrix for this Molecule.
     *
     * \returns distances a Vec object representing the distances
     *
     * Since the NxN distance matrix for all atoms in this molecule is by
     * definition symmetric, and the diagonals would be 0, this will
     * return a `N * (N - 1) / 2` length vector where the distance for atom i
     * to atom j can be calculated in the following loop:
     *
     * ```
     * for (size_t i = 0; i < N; i++) {
     *     for (size_t j = i + 1; j < N; j++) {
     *      // distance
     *     }
     * }
     * ```
     */
    Vec interatomic_distances() const;

    /**
     * The Element objects for the atoms in this Molecule
     *
     * \returns a const std::vector<Element>& to the internal vector of elements
     */
    inline const auto &elements() const { return m_elements; }

    /**
     * The positions of the atoms in this Molecule
     *
     * \returns a const ref to the internal matrix of positions (Angstroms).
     */
    inline const Mat3N &positions() const { return m_positions; }

    /**
     * The atomic numbers for the atoms in this Molecule
     *
     * \returns a const ref to the internal vector of atomic numbers.
     */
    inline const IVec &atomic_numbers() const { return m_atomicNumbers; }

    /**
     * The van der Waals radii for the atoms in this Molecule
     *
     * \returns a newly constructed Vec containing van der Waals radii
     *
     * Creates a vector, they are not stored internally to the Molecule object.
     */
    Vec vdw_radii() const;

    /**
     * The van der Waals radii for the atoms in this Molecule
     *
     * \returns a newly constructed Vec containing van der Waals radii
     *
     * Creates a vector, they are not stored internally to the Molecule object.
     */
    Vec covalent_radii() const;

    /**
     * The atomic masses of the atoms in this Molecule
     *
     * \returns a newly constructed Vec containing the masses.
     *
     * Creates a vector, they are not stored internally to the Molecule object.
     */
    Vec atomic_masses() const;

    /**
     * The total molecular mass
     *
     * \returns a double representing the sum of atomic masses in this molecule
     */
    double molar_mass() const;

    /**
     * Convert the atoms represented by this molecule to a std:vector<Atom>
     *
     * \returns a vector of Atom objects (positions in Bohr).
     */
    std::vector<core::Atom> atoms() const;

    /**
     * Set the unit cell offset for this Molecule (default 000)
     *
     * \param shift the desired CellShift
     *
     * See Molecule::cell_shift
     */
    void set_cell_shift(const CellShift &shift);

    /**
     * Get the unit cell offset for this Molecule (default 000)
     *
     * \returns the CellShift stored in this Molecule
     *
     * See Molecule::set_cell_shift
     */
    const CellShift &cell_shift() const;

    /**
     * The geometric centre of this molecule
     *
     * \returns a Vec3 representing the average atomic position (ignoring
     * masses)
     *
     * See Molecule::center_of_mass if you need to incorporate masses
     */
    Vec3 centroid() const;

    /**
     * The mass-weighted geometric centre of this molecule
     *
     * \returns a Vec3 representing the average atomic position (including
     * masses)
     *
     * See Molecule::centroid if you don't need to incorporate masses
     */
    Vec3 center_of_mass() const;

    /**
     * The tensor of inertia (in matrix form).
     *
     * \returns a Mat3 representing the distribution of mass about the
     * Molecule::center_of_mass of this Molecule.
     */
    Mat3 inertia_tensor() const;

    /**
     * The principal moments of inertia, derived from the intertia tensor.
     *
     * \returns a Vec3 representing the three moments of inertia. Eigenvalues
     * calculated using Eigen::SelfAdjointEigenSolver.
     */
    Vec3 principal_moments_of_inertia() const;

    /**
     * The rotational constants of this Molecule in GHz.
     *
     * \returns a Vec3 representing the three rotational constants. Calculated
     * directly from Molecule::principal_moments_of_inertia.
     */
    Vec3 rotational_constants() const;

    /**
     * The rotational component of the entropic free energy of this Molecule
     * (rigid molecule, ideal gas approximation), units are kJ/mol.
     *
     * \param T the temperature (K).
     *
     * \returns a double representing the rotational free energy for this
     * Molecule at the given temperature in kJ/mol
     */
    double rotational_free_energy(double T) const;

    /**
     * The translational component of the entropic free energy of this Molecule
     * (rigid molecule, ideal gas approximation), units are kJ/mol.
     *
     * \param T the temperature (K).
     *
     * \returns a double representing the translational component of the free
     * energy for this Molecule at the given temperature in kJ/mol
     */
    double translational_free_energy(double T) const;

    /**
     * The nearest atom-atom pair between atoms in this Molecule and atoms in
     * another.
     *
     * \param rhs The other Molecule object to find distances.
     *
     * \returns a std::tuple<size_t, size_t, double> consisting of the index
     * into atoms in this Molecule, the index into the atoms of the Molecule
     * rhs, and the distance in Angstroms.
     *
     * See Molecule::nearest_atom_distance if you only need the distance
     */
    std::tuple<size_t, size_t, double> nearest_atom(const Molecule &rhs) const;

    /**
     * Manually add a bond connection between two atoms in this Molecule.
     *
     * \param l the first index of an atom into this Molecule.
     * \param r the second index of an atom into this Molecule.
     *
     * Internally, the set of bonds in this Molecule will be updated.
     */
    void add_bond(size_t l, size_t r) { m_bonds.push_back({l, r}); }

    /**
     * Set the known bond connections in this Molecule.
     *
     * \param bonds a std::vector<std::pair<size_t, size_t>> i.e. list of the
     * bond edges to set in this Molecule.
     *
     * \warning Existing bonds will be removed, this method will overwrite bonds
     * already stored.
     */
    void set_bonds(const std::vector<std::pair<size_t, size_t>> &bonds) {
        m_bonds = bonds;
    }

    /**
     * Get the list of bonds in this Molecule.
     *
     * \returns a std::vector<std::pair<size_t, size_t>> i.e. list of the
     * bond edges that have been set in this Molecule.
     */
    const std::vector<std::pair<size_t, size_t>> &bonds() const {
        return m_bonds;
    }

    /**
     * The net charge of this Molecule.
     *
     * \returns an integer representing the net charge. Default is neutral (0)
     */
    int charge() const { return m_charge; }

    /**
     * Set the net charge of this Molecule.
     *
     * \param c an integer representing the net charge.
     *
     * No checks are performed to ensure this charge is sensible.
     */
    void set_charge(int c) { m_charge = c; }

    /**
     * Get the spin multiplicity of this molecule.
     *
     * \returns an integer representing the spin multiplicity of this molecule
     * (default 1)
     */
    int multiplicity() const { return m_multiplicity; }

    /**
     * Set the spin multiplicity of this molecule.
     *
     * \param m an integer representing the spin multiplicity of this molecule.
     *
     * No checks are performed to ensure this is a sensible value.
     */
    void set_multiplicity(int m) { m_multiplicity = m; }

    /**
     * The number of electrons in this molecule.
     *
     * \returns an integer representing total number of electrons.
     *
     * Molecule::charge is incorporated, so the result should be the sum
     * of atomic numbers - net charge (as electrons have -ve charge).
     */
    int num_electrons() const { return m_atomicNumbers.sum() - m_charge; }

    /**
     * Get the index into Crystal::symmetry_unique_molecules for this
     * Molecule.
     *
     * \returns an integer representing the index (default is -1)
     *
     * \warning This will return an invalid index (-1) by default, and
     * should be checked or ensured to have a value before use.
     */
    int asymmetric_molecule_idx() const { return m_asym_mol_idx; }

    /**
     * Set the index into Crystal::symmetry_unique_molecules for this
     * Molecule.
     *
     * \param idx an integer representing the index.
     *
     * \warning No check is performed to ensure this is a sensible value.
     */
    void set_asymmetric_molecule_idx(size_t idx) { m_asym_mol_idx = idx; }

    /**
     * Get the index into Crystal::unit_cell_molecules for this
     * Molecule.
     *
     * \returns an integer representing the index (default is -1)
     *
     * \warning This will return an invalid index (-1) by default, and
     * should be checked or ensured to have a value before use.
     */
    int unit_cell_molecule_idx() const { return m_uc_mol_idx; }

    /**
     * Set the index into Crystal::unit_cell_molecules for this
     * Molecule.
     *
     * \param idx an integer representing the index.
     *
     * \warning No check is performed to ensure this is a sensible value.
     */
    void set_unit_cell_molecule_idx(size_t idx) { m_uc_mol_idx = idx; }

    /**
     * Set the transformation from the corresponding molecule in the
     * asymmetric unit of a Crystal to this geometry.
     *
     * \param rot A Mat3 representing the rotation part of the transform.
     * \param trans A Vec3 representing the translation part of the transform.
     *
     * By default the rotation is the identity matrix, and the translation is
     * the zero vector. Molecule::set_asymmetric_molecule_index should also
     * be set for this to make sense. It is the responsibility of the caller
     * to ensure these values correspond.
     */
    void set_asymmetric_unit_transformation(const Mat3 &rot,
                                            const Vec3 &trans) {
        m_asymmetric_unit_rotation = rot;
        m_asymmetric_unit_translation = trans;
    }

    /**
     * Get the transformation from the corresponding molecule in the
     * asymmetric unit of a Crystal to this geometry.
     *
     * \returns a std::pair<Mat3, Vec3> representing the rotation part of the
     * transform. and the translation part of the transform.
     *
     * By default the rotation is the identity matrix, and the translation is
     * the zero vector.
     *
     * See also Molecule::asymmetric_molecule_idx
     */
    std::pair<Mat3, Vec3> asymmetric_unit_transformation() const {
        return {m_asymmetric_unit_rotation, m_asymmetric_unit_translation};
    }

    /**
     * Set the unit cell atom indices for all atoms in this Molecule
     *
     * \param idx an IVec containing the relevant indices into the set of unit
     * cell atoms for the Crystal associated with this Molecule.
     *
     * See also Molecule::set_unit_cell_molecule_idx if you wish to set the
     * molecule index rather than atom indices.
     */
    void set_unit_cell_idx(const IVec &idx) { m_uc_idx = idx; }

    /**
     * Set the asymmetric unit atom indices for all atoms in this Molecule
     *
     * \param idx an IVec containing the relevant indices into the set of
     * asymmetric unit atoms for the Crystal associated with this Molecule.
     *
     * See also Molecule::set_asymmetric_molecule_idx if you wish to set the
     * molecule index rather than atom indices.
     */
    void set_asymmetric_unit_idx(const IVec &idx) { m_asym_idx = idx; }

    /**
     * Set the associated SymmetryOperation for all all atoms in from their
     * asymmetric unit counterpart (encoded as an int).
     *
     * \param symop an IVec containing the integer encoded SymmetryOperation
     * from the corresponding asymmetric unit atoms in the Crystal for all atoms
     * in.
     *
     * See also Molecule::set_asymmetric_unit_transformation if you wish to set
     * the transformation from the molecule. And note that these are not the
     * same thing.
     */
    void set_asymmetric_unit_symop(const IVec &symop);

    /**
     * Get the unit cell atom indices for all atoms in this Molecule
     *
     * \return a const ref to the internal IVec containing the relevant indices
     * into the set of unit cell atoms for the Crystal associated with this
     * Molecule.
     *
     * See also Molecule::unit_cell_molecule_idx if you wish to get the mapping
     * for the molecule index rather than atom indices.
     */
    const auto &unit_cell_idx() const { return m_uc_idx; }

    /**
     * Get the asymmetric unit atom indices for all atoms in this Molecule
     *
     * \return a const ref to the internal IVec containing the relevant indices
     * into the set of asymmetric unit atoms for the Crystal associated with
     * this Molecule.
     *
     * See also Molecule::asymmetric_molecule_idx if you wish to get the mapping
     * for the molecule index rather than atom indices.
     */
    const auto &asymmetric_unit_idx() const { return m_asym_idx; }

    /**
     * Get the set of SymmetryOperation for all atoms in this Molecule (i.e.
     * mapping from their corresponding asymmetric unit atom to their position
     * here).
     *
     * \return a const ref to the internal IVec containing the relevant integer
     * encoded SymmeteryOperations for the mapping from asymmetric unit atoms
     * for the Crystal associated with this Molecule.
     *
     * See also Molecule::asymmetric_unit_transformation if you wish to get the
     * mapping for the molecule index rather than atom indices.
     */
    const auto &asymmetric_unit_symop() const { return m_asym_symop; }

    /**
     * Rotate this Molecule by the given rotation, about the specified origin.
     *
     * \param r The rotation.
     * \param o The desired Origin about which to rotate.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::rotated if you wish to return a copy of this Molecule.
     */
    void rotate(const Eigen::Affine3d &r, Origin o = Cartesian);

    /**
     * Rotate this Molecule by the given rotation, about the specified origin.
     * Overload for any Mat3.
     *
     * \param r The rotation.
     * \param o The desired Origin about which to rotate.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin).
     *
     * \warning The provided Mat3 is not checked to be a valid rotation matrix.
     *
     * See Molecule::rotated if you wish to return a copy of this Molecule.
     */
    void rotate(const Mat3 &r, Origin o = Cartesian);

    /**
     * Rotate this Molecule by the given rotation, about the specified origin.
     * Overload for any Mat3.
     *
     * \param r The rotation.
     * \param o The desired position about which to rotate (Angstroms).
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin).
     *
     * \warning The provided Mat3 is not checked to be a valid rotation matrix.
     *
     * See Molecule::rotated if you wish to return a copy of this Molecule.
     */
    void rotate(const Mat3 &r, const Vec3 &o);

    /**
     * Transform this Molecule by the given homogeneous transformation matrix,
     * with rotation performed about the specified origin.
     *
     * \param t The homogeneous transformation matrix.
     * \param o The desired Origin about which to rotate.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::transformed if you wish to return a copy of this Molecule.
     */
    void transform(const Mat4 &t, Origin o = Cartesian);

    /**
     * Transform this Molecule by the given homogeneous transformation matrix,
     * with rotation performed about the specified origin.
     *
     * \param t The homogeneous transformation matrix.
     * \param o The desired position about which to rotate (Angstroms).
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::transformed if you wish to return a copy of this Molecule.
     */
    void transform(const Mat4 &t, const Vec3 &o);

    /**
     * Translate this Molecule by the given vector.
     *
     * \param t translation vector (Angstroms).
     *
     * See Molecule::translated if you wish to return a copy of this Molecule.
     */
    void translate(const Vec3 &t);

    /**
     * A copy of this Molecule transformed by the given rotation, about the
     * specified origin.
     *
     * \param r The rotation.
     * \param o The desired Origin about which to rotate.
     * \returns a rotated copy of this Molecule.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::rotate if you wish to modify this Molecule.
     */
    Molecule rotated(const Eigen::Affine3d &r, Origin o = Cartesian) const;

    /**
     * A copy of this Molecule transformed by the given rotation, about the
     * specified origin.
     *
     * \param r The rotation.
     * \param o The desired Origin about which to rotate.
     * \returns a rotated copy of this Molecule.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * \warning The provided Mat3 is not checked to be a valid rotation matrix.
     *
     * See Molecule::rotate if you wish to modify this Molecule.
     */
    Molecule rotated(const Mat3 &r, Origin o = Cartesian) const;

    /**
     * A copy of this Molecule transformed by the given rotation, about the
     * specified origin.
     *
     * \param r The rotation.
     * \param o The desired position about which to rotate (Angstroms).
     * \returns a rotated copy of this Molecule.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * \warning The provided Mat3 is not checked to be a valid rotation matrix.
     *
     * See Molecule::rotate if you wish to modify this Molecule.
     */
    Molecule rotated(const Mat3 &r, const Vec3 &o) const;

    /**
     * A copy of this Molecule transformed by the given homogeneous
     * transformation matrix, with rotation performed about the specified
     * origin.
     *
     * \param t The homogeneous transformation matrix.
     * \param o The desired origin about which to rotate.
     *
     * \returns transformed copy of this Molecule.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::transform if you wish to modify this Molecule.
     */
    Molecule transformed(const Mat4 &t, Origin o = Cartesian) const;

    /**
     * A copy of this Molecule transformed by the given homogeneous
     * transformation matrix, with rotation performed about the specified
     * origin.
     *
     * \param t The homogeneous transformation matrix.
     * \param o The desired position about which to rotate (Angstroms).
     *
     * \returns transformed copy of this Molecule.
     *
     * All rotations will be in the Cartesian axis frame (though not necessarily
     * about the Cartesian origin)
     *
     * See Molecule::transforme if you wish to modify this Molecule.
     */
    Molecule transformed(const Mat4 &t, const Vec3 &o) const;

    /**
     * A copy of this Molecule translated by the given vector.
     *
     * \param t translation vector (Angstroms).
     * \returns a translated copy of this Molecule.
     *
     * See Molecule::translate if you wish to modify this Molecule.
     */
    Molecule translated(const Vec3 &t) const;

    /**
     * Determine whether this Molecule and another can be sensibly compared.
     *
     * \returns true if they are comparable Molecules per the definition, false
     * otherwise.
     *
     * The working definition here of whether two molecules are comparable is
     * whether they have the exact same IVec of atomic numbers, in the same
     * order (i.e. the same chemical composition, and the order of atoms is not
     * permuted.
     */
    bool is_comparable_to(const Molecule &rhs) const;

    /**
     * Determine whether this Molecule and another are equivalent.
     *
     * \param rhs The other Molecule object to check equivalence
     *
     * \returns true if they are comparable Molecules per the definition, false
     * otherwise.
     *
     * The working definition here of whether two molecules are equivalent if
     * a) they are comparable (i.e. all atomic numbers are the same) and b)
     * all interatomic distances are the same.
     */
    bool is_equivalent_to(const Molecule &rhs) const;

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

} // namespace occ::core
