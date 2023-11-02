#pragma once
#include <occ/core/bondgraph.h>
#include <occ/core/dimer.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/crystal/asymmetric_unit.h>
#include <occ/crystal/hkl.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/unitcell.h>
#include <vector>

namespace occ::crystal {

using occ::IVec;
using occ::Mat3N;
using occ::core::Molecule;
using occ::core::graph::PeriodicBondGraph;

/**
 * \brief A class representing a region of atoms in a crystal lattice.
 *
 * A crystal atom region is a collection of atoms in a crystal lattice,
 * characterized by their fractional and cartesian coordinates, their
 * atomic numbers, and their symmetry operations.
 */
struct CrystalAtomRegion {
    /**
     * \brief The fractional coordinates of the atoms in the region, expressed
     *        as a matrix of size (3, n), where n is the number of atoms. Each
     *        column of the matrix represents the (x, y, z) coordinates of an
     *        atom, expressed as a fraction of the unit cell dimensions.
     */
    Mat3N frac_pos;

    /**
     * \brief The cartesian coordinates of the atoms in the region, expressed as
     *        a matrix of size (3, n), where n is the number of atoms. Each
     *        column of the matrix represents the (x, y, z) coordinates of an
     *        atom, expressed in angstroms.
     */
    Mat3N cart_pos;

    /**
     * \brief The indices of the asymmetric units containing the atoms in the
     *        region, expressed as a vector of size n. Each element of the
     *        vector represents the index of the asymmetric unit that contains
     *        the corresponding atom.
     */
    IVec asym_idx;

    /**
     * \brief The atomic numbers of the atoms in the region, expressed as a
     *        vector of size n.
     */
    IVec atomic_numbers;

    /**
     * \brief The symmetry operations applied to the atoms in the region,
     *        expressed as a vector of size n. Each element of the vector
     *        represents the symmetry operation that maps the corresponding atom
     *        to its equivalent position in the unit cell.
     */
    IVec symop;

    /**
     * \brief Resizes the region to contain the given number of atoms.
     *
     * \param n The new size of the region, i.e., the number of atoms it will
     *        contain.
     */
    void resize(size_t n) {
        frac_pos.resize(3, n);
        cart_pos.resize(3, n);
        asym_idx.resize(n);
        atomic_numbers.resize(n);
        symop.resize(n);
    }

    /**
     * \brief Returns the number of atoms in the region.
     *
     * \return The size of the region, i.e., the number of atoms it contains.
     */
    size_t size() const { return frac_pos.cols(); }
};

/**
 * \brief A class representing the molecular dimers in a crystal lattice.
 *
 * A dimer is a pair of molecules that may or may not be symmetry related.
 * A crystal dimers object stores the unique dimers in the lattice, as well
 * as the complete set of dimers that are equivalent to the unique dimers
 */
struct CrystalDimers {
    struct SymmetryRelatedDimer {
        occ::core::Dimer dimer;
        int unique_index{-1};
    };
    using MoleculeNeighbors = std::vector<SymmetryRelatedDimer>;
    /**
     * \brief The search radius used to create this set of dimers
     *
     * In angstroms
     */
    double radius{0.0};

    /**
     * \brief A vector containing the unique dimers in the lattice.
     *
     * Each dimer is represented by its two molecules, in a `Dimer` class
     */
    std::vector<occ::core::Dimer> unique_dimers;

    /**
     * \brief A vector of vectors containing the dimers surrounding each
     * symmetry unique molecule in the lattice along with the indices of
     * the unique dimers they correspond to.
     *
     * Each inner vector contains the complete set of dimers that are
     * surrounding a particular molecule
     */
    std::vector<MoleculeNeighbors> molecule_neighbors;
};

/**
 * \brief A class representing a crystal structure.
 *
 * A crystal is a periodic arrangement of atoms in 3D space,
 * characterized by its unit cell, space group, and the atoms in its
 * asymmetric unit. This class provides methods to access and manipulate
 * the properties of the crystal lattice, such as its atoms, bonds,
 * molecules, and symmetry operations.
 */
class Crystal {
  public:
    /**
     * \brief Constructs a crystal lattice from the given asymmetric unit, space
     * group, and unit cell.
     *
     * \param asymmetric_unit The asymmetric unit of the crystal lattice,
     * containing the atoms and their properties.
     *
     * \param space_group The space group of the crystal lattice, defining its
     * symmetry operations.
     *
     * \param unit_cell The unit cell of the crystal lattice, defining its
     * lengths and angles (cell vectors).
     */
    Crystal(const AsymmetricUnit &asymmetric_unit,
            const SpaceGroup &space_group, const UnitCell &unit_cell);

    /**
     * \brief Returns the labels of the atoms in the asymmetric unit of this
     * crystal.
     *
     * The labels are assigned to each atom in the asymmetric unit,
     * and they are used to identify the atoms in the lattice.
     *
     * \return A vector of strings containing the labels of the atoms in the
     * lattice, in the same order as their positions.
     */
    const std::vector<std::string> &labels() const {
        return m_asymmetric_unit.labels;
    }

    /**
     * \brief Returns the fractional coordinates of the atoms in the asymmetric
     * unit of this crystal
     *
     * The fractional coordinates of an atom are its position in the lattice,
     * expressed as a fraction of the unit cell dimensions.
     *
     * \return A matrix of size (3, n), where n is the number of atoms in the
     * lattice. Each column of the matrix represents the (a, b, c) fractional
     * coordinates of an atom in the lattice.
     */
    const Mat3N &frac() const { return m_asymmetric_unit.positions; }

    /**
     * \brief Converts the given cartesian coordinates to fractional
     * coordinates.
     *
     * \param p A matrix of size (3, n), where n is the number of coordinates to
     * convert. Each column of the matrix represents the (x, y, z) cartesian
     * coordinates of a point in the lattice.
     *
     * \return A matrix of size (3, n), where n is the number of coordinates to
     * convert. Each column of the matrix represents the (a, b, c) fractional
     * coordinates of the same points, after converting them from cartesian to
     * fractional coordinates.
     */
    inline auto to_fractional(const Mat3N &p) const {
        return m_unit_cell.to_fractional(p);
    }

    /**
     * \brief Converts the given fractional coordinates to cartesian
     * coordinates.
     *
     * \param p A matrix of size (3, n), where n is the number of coordinates to
     * convert. Each column of the matrix represents the (x, y, z) fractional
     * coordinates of a point in the lattice.
     *
     * \return A matrix of size (3, n), where n is the number of coordinates to
     * convert. Each column of the matrix represents the (x, y, z) cartesian
     * coordinates of the same points, after converting them from fractional to
     * cartesian coordinates.
     */
    inline auto to_cartesian(const Mat3N &p) const {
        return m_unit_cell.to_cartesian(p);
    }

    /**
     * \brief Returns the number of sites in the crystal lattice.
     *
     * The number of sites in the lattice is equal to the number of atoms
     * in the asymmetric unit
     *
     * \return The number of sites in the lattice.
     */
    inline int num_sites() const {
        return m_asymmetric_unit.atomic_numbers.size();
    }

    /**
     * \brief Returns the symmetry operations of the crystal lattice.
     *
     * The symmetry operations of the lattice are defined by its space group,
     * and they are used to generate the equivalent positions of each atom in
     * the lattice.
     *
     * \return A vector of `SymmetryOperation` objects, containing the symmetry
     * operations of the lattice, in the same order as they are defined in the
     * space group.
     */
    inline const std::vector<SymmetryOperation> &symmetry_operations() const {
        return m_space_group.symmetry_operations();
    }

    /**
     * \brief Returns the space group of the crystal lattice.
     *
     * The space group of the lattice defines its symmetry operations,
     * which are used to generate the equivalent positions of each atom in the
     * lattice.
     *
     * \return A constant reference to the `SpaceGroup` object that defines the
     * space group of the lattice.
     */
    const SpaceGroup &space_group() const { return m_space_group; }

    /**
     * \brief Returns a constant reference to the asymmetric unit of the crystal
     * lattice.
     *
     * The asymmetric unit of the lattice contains the atoms and their
     * properties, such as their positions, labels, and atomic numbers.
     *
     * \return A constant reference to the `AsymmetricUnit` object that defines
     * the asymmetric unit of the lattice.
     */
    const AsymmetricUnit &asymmetric_unit() const { return m_asymmetric_unit; }

    /**
     * \brief Returns a reference to the asymmetric unit of the
     * crystal lattice.
     *
     * The asymmetric unit of the lattice contains the atoms and their
     * properties, such as their positions, labels, and atomic numbers. This
     * method allows modifying the properties of the atoms in the lattice.
     *
     * \return A reference to the `AsymmetricUnit` object that
     * defines the asymmetric unit of the lattice.
     */
    AsymmetricUnit &asymmetric_unit() { return m_asymmetric_unit; }

    /**
     * \brief Returns the unit cell of the crystal lattice.
     *
     * The unit cell of the lattice defines its dimensions and geometry,
     * which are used to generate the equivalent positions of each atom in the
     * lattice.
     *
     * \return A reference to the `UnitCell` object that defines the unit cell
     * of the lattice.
     */
    const UnitCell &unit_cell() const { return m_unit_cell; }

    /**
     * \brief Returns the atoms in the specified slab of the crystal lattice.
     *
     * A slab is a subset of the lattice that is defined by a pair of Miller
     * indices (hkl), which specify the corners and of the slab (rectangular
     * prism in lattice space).
     *
     * The atoms in the slab are those whose fractional coordinates lie within
     * inside the range of these corners, generated from the unit cell atoms,
     * translated to all possible unit cells.
     *
     * \param lower The Miller indices (hkl) of the
     * first corner of the slab.
     * \param upper The Miller indices (hkl) of the
     * second corner of the slab.
     *
     * \return A `CrystalAtomRegion` object containing the atoms in the
     * specified slab of the lattice.
     */
    CrystalAtomRegion slab(const HKL &lower, const HKL &upper) const;

    /**
     * \brief Returns the atoms in the unit cell of the crystal lattice.
     *
     * The atoms in the unit cell are those that are equivalent to the atoms
     * in the asymmetric unit, after applying the symmetry operations of the
     * lattice.
     *
     * \return A constant reference to a `CrystalAtomRegion` object containing
     * the atoms in the unit cell of the lattice. The `CrystalAtomRegion` class
     * provides methods to access the positions, atomic numbers, and symmetry
     * operations of the atoms in the unit cell.
     */
    const CrystalAtomRegion &unit_cell_atoms() const;

    /**
     * \brief Returns the atoms within a certain radius of the specified atom in
     * the crystal lattice.
     *
     * This method can be used to find the neighbors of an atom, or to identify
     * the atoms that are within a certain range of distances from the specified
     * atom.
     *
     * \param asym_idx The index of the atom in the asymmetric unit of the
     * lattice. By default, this parameter is set to 0, indicating the first
     * atom in the asymmetric unit.
     *
     * \param radius The maximum distance (Angstroms) from the
     * specified atom that an atom can be in order to be included in the result.
     * (default = 6.0)
     *
     * \return A `CrystalAtomRegion` object containing the atoms within the
     * specified radius of the specified atom. The `CrystalAtomRegion` class
     * provides methods to access the positions, atomic numbers, and symmetry
     * operations of the atoms in the region.
     */
    CrystalAtomRegion atom_surroundings(int asym_idx = 0,
                                        double radius = 6.0) const;

    /**
     * \brief Returns the atoms within a certain radius of each atom in the
     * asymmetric unit of the crystal lattice.
     *
     * This method can be used to find the neighbors of each atom in the
     * asymmetric unit, or to identify the atoms that are within a certain range
     * of distances from each atom.
     *
     * \param radius The maximum distance from each atom that an atom can be in
     * order to be included in the result. This parameter is used for all atoms
     * in the asymmetric unit, and its value must be positive.
     *
     * \return A vector of `CrystalAtomRegion` objects, where each element of
     * the vector contains the atoms within the specified radius of a different
     * atom in the asymmetric unit. The `CrystalAtomRegion` class provides
     * methods to access the positions, atomic numbers, and symmetry operations
     * of the atoms in the region.
     */
    std::vector<CrystalAtomRegion>
    asymmetric_unit_atom_surroundings(double radius) const;

    /**
     * \brief Returns the connectivity graph of the atoms in the unit cell of
     * the crystal lattice.
     *
     * The connectivity graph of the atoms in the unit cell is a graph data
     * structure that represents the bonds between the atoms in the unit cell.
     * The graph is built by considering the bonds between the atoms in the
     * unit cell and neighbouring cells.
     *
     * \return A reference to a `PeriodicBondGraph` object representing the
     * connectivity graph of the atoms in the unit cell. The `PeriodicBondGraph`
     * class provides methods to access and manipulate the graph data, such as
     * adding or removing bonds, or traversing the graph to find connected
     * components or shortest paths.
     */
    const PeriodicBondGraph &unit_cell_connectivity() const;

    /**
     * \brief Returns the molecules in the unit cell of the crystal lattice.
     *
     * A molecule is a connected subgraph of the connectivity graph of the atoms
     * in the unit cell. This method returns the molecules that are identified
     * by traversing the graph and partitioning it into connected components.
     *
     * \return A vector of `Molecule` objects, where each element of the vector
     * represents a different molecule in the unit cell of the lattice.
     * */
    const std::vector<Molecule> &unit_cell_molecules() const;

    /**
     * \brief Returns the symmetry-unique molecules in the unit cell of the
     * crystal lattice.
     *
     * A symmetry-unique molecule is a the subset of `unit_cell_molecules` that
     * are not related to each other by symmetry (i.e. removing symmetry related
     * duplicates)
     *
     * \return A vector of `Molecule` objects, where each element of the vector
     * represents a different symmetry-unique molecule in the unit cell of the
     * lattice.
     * */
    const std::vector<Molecule> &symmetry_unique_molecules() const;

    /**
     * \brief Returns the volume of the unit cell of the crystal lattice.
     *
     * The volume of the unit cell is calculated from the lattice vectors of the
     * unit cell, using the formula V = |a x b x c|, where a, b and c are the
     * lattice vectors of the unit cell.
     *
     * \return The volume of the unit cell of the lattice, as a double.
     */
    double volume() const;

    /**
     * \brief Returns the neighbouring dimers for the symmetry-unique molecules
     * in this crystal, within a certain neighbour radius.
     *
     * A dimer is a pair of molecules in the crystal.
     *
     * \param distance_tolerance The maximum distance between two atoms in order
     * for them to be considered.
     *
     * \return A `CrystalDimers` object containing the symmetry-unique dimers in
     * the crystal.
     */
    CrystalDimers symmetry_unique_dimers(double distance_tolerance) const;

    /**
     * \brief Returns the neighbouring dimers for the unit cell molecules
     * in this crystal, within a certain neighbour radius.
     *
     * A dimer is a pair of molecules in the crystal.
     *
     * \param distance_tolerance The maximum distance between two atoms in order
     * for them to be considered.
     *
     * \return A `CrystalDimers` object containing the unit cell dimers in
     * the crystal.
     */
    CrystalDimers unit_cell_dimers(double distance_tolerance) const;

    /**
     * \brief Returns a string representing the symmetry of a dimer in the
     * crystal lattice.
     *
     * Given a dimer in the lattice, returns a string representation of a the
     * symmetry operation relating them (including translations)
     *
     * \param dimer The dimer for which to calculate the symmetry string. The
     * dimer must be a valid dimer in the crystal lattice, for example from the
     * `CrystalDimers` class.
     *
     * \return A string representing the symmetry of the dimer, composed of
     * rotation and translation components
     */
    std::string dimer_symmetry_string(const occ::core::Dimer &dimer) const;

    /**
     * \brief Specify the behaviour for guessing/finding bonds.
     */
    void set_connectivity_criteria(bool guess = true);

    /**
     * \brief Creates a primitive supercell from a crystal lattice.
     *
     * Given a crystal lattice, this method constructs a new lattice that is a
     * primitive supercell of the original lattice. The dimensions of the
     * supercell are given in the HKL parameter, and all symmetry beyond
     * translational symmetry is removed in the resulting crystal.
     *
     * \param c The crystal lattice from which to create the primitive
     * supercell.
     *
     * \param hkl The lattice vector triplet (h, k, l) that defines
     * the primitive supercell.
     *
     * \return A new `Crystal` object representing the primitive supercell of
     * the original lattice.
     */
    static Crystal create_primitive_supercell(const Crystal &c, HKL hkl);

  private:
    AsymmetricUnit m_asymmetric_unit;
    SpaceGroup m_space_group;
    UnitCell m_unit_cell;
    bool m_guess_connectivity{true};
    void update_unit_cell_molecules() const;
    void update_symmetry_unique_molecules() const;
    void update_unit_cell_connectivity() const;
    void update_unit_cell_atoms() const;

    mutable std::vector<typename PeriodicBondGraph::VertexDescriptor>
        m_bond_graph_vertices;
    mutable PeriodicBondGraph m_bond_graph;
    mutable CrystalAtomRegion m_unit_cell_atoms;
    mutable bool m_symmetry_unique_molecules_needs_update{true};
    mutable bool m_unit_cell_atoms_needs_update{true};
    mutable bool m_unit_cell_molecules_needs_update{true};
    mutable bool m_unit_cell_connectivity_needs_update{true};
    mutable std::vector<Molecule> m_unit_cell_molecules{};
    mutable std::vector<Molecule> m_symmetry_unique_molecules{};
};

} // namespace occ::crystal
