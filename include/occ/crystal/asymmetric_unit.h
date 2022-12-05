#pragma once
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::crystal {
/**
 * \brief A class representing an asymmetric unit of a crystal structure.
 *
 * An asymmetric unit is a building block of a crystal lattice, and it contains
 * a set of atoms with their positions, atomic numbers, occupations, charges,
 * and labels.
 */
struct AsymmetricUnit {
    /**
     * \brief Constructs an empty asymmetric unit.
     */
    AsymmetricUnit() {}

    /**
     * \brief Constructs an asymmetric unit from the given positions and atomic
     * numbers.
     *
     * \param positions The positions of the atoms in the unit cell, expressed
     * as a matrix of size (3, n), where n is the number of atoms. Each column
     * of the matrix represents the (x, y, z) coordinates of an atom.
     * \param atomic_numbers The atomic numbers of the atoms in the unit cell,
     * expressed as a vector of size n.
     */
    AsymmetricUnit(const Mat3N &positions, const IVec &atomic_numbers);

    /**
     * \brief Constructs an asymmetric unit from the given positions, atomic
     * numbers, and labels.
     *
     * \param positions The positions of the atoms in the unit cell, expressed
     * as a matrix of size (3, n), where n is the number of atoms. Each column
     * of the matrix represents the (x, y, z) coordinates of an atom.
     * \param atomic_numbers The atomic numbers of the atoms in the unit cell,
     *    expressed as a vector of size n. \param labels The labels of the atoms
     * in the unit cell, expressed as a vector of strings of size n. Each string
     *    represents the label of the corresponding atom.
     */
    AsymmetricUnit(const Mat3N &positions, const IVec &atomic_numbers,
                   const std::vector<std::string> &labels);

    /**
     * \brief The positions of the atoms in the unit cell, expressed as a matrix
     * of size (3, n), where n is the number of atoms. Each column of the matrix
     * represents the (x, y, z) coordinates of an atom.
     */
    Mat3N positions;

    /**
     * \brief The atomic numbers of the atoms in the unit cell, expressed as a
     * vector of size n.
     */
    IVec atomic_numbers;

    /**
     * \brief The occupations of the atoms in the unit cell, expressed as a
     * vector of size n.
     */
    Vec occupations;

    /**
     * \brief The charges of the atoms in the unit cell, expressed as a vector
     * of size n.
     */
    Vec charges;

    /**
     * \brief The labels of the atoms in the unit cell, expressed as a vector of
     * strings of size n. Each string represents the label of the corresponding
     * atom.
     */
    std::vector<std::string> labels;

    /**
     * \brief Returns the chemical formula of the asymmetric unit.
     *
     * The chemical formula is a string that represents the composition of the
     * unit cell, using the symbols of the chemical elements. For example, "H2O"
     * or "CuZn" \return The chemical formula of the asymmetric unit.
     */
    std::string chemical_formula() const;

    /**
     * \brief Returns the covalent radii of the atoms in the asymmetric unit.
     *
     * The covalent radii of an atom is a measure of the size of its atomic
     * nucleus, and it is typically used to calculate the distances between
     * atoms in a molecule.
     *
     * \return A vector of size n containing the covalent radii of the atoms in
     * the unit cell, in the same order as their atomic numbers.
     */
    Vec covalent_radii() const;

    /**
     * \brief Generates default labels for the atoms in the unit cell.
     *
     * The default labels are assigned based on the chemical element of each
     * atom, using the standard one- or two-letter symbol followed by a number.
     * For example, "C1", "N2", "Fe3".
     */
    void generate_default_labels();

    /**
     * \brief Returns the number of atoms in the unit cell.
     *
     * \return The size of the unit cell, i.e., the number of atoms it contains.
     */
    size_t size() const { return atomic_numbers.size(); }
};

} // namespace occ::crystal
