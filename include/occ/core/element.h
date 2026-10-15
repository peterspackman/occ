#pragma once
#include <string>
#include <vector>
#include <occ/core/linear_algebra.h>
#define ELEMENT_MAX 103

namespace occ::core {

/// @cond DEV
struct ElementData {
  int atomic_number;
  std::string name;
  std::string symbol;
  float cov_radius;
  float vdw_radius;
  float mass;
};

/// @endcond

/**
 * Utility class representing and holding data for a chemical element.
 *
 * Element is a simple class with getter methods for typical useful values
 * of common data convenient for common operations in computational chemistry.
 *
 * Defined for the periodic table from H-Lr, using average isotopic data.
 *
 */
class Element {
public:
  Element() = delete;
  /**
   * Construct an Element instance from its chemical symbol or name.
   *
   * \param string a string object representing the chemical symbol, element
   * name, or label to identify the relevant Element.
   * \param exact_match whether to only find exact matches for the given
   * string.
   *
   * The string parameter is trimmed of whitespace, then capitalized prior to
   * attempted matching so it should be case insensitive.
   *
   * If exact_match is false, then things like labels e.g. H1, HA will be able
   * to match (based on a partial match) with the relevant element. This
   * behaviour may be undesirable, hence the optional parameter.
   *
   */
  Element(const std::string &string, bool exact_match = false);

  /**
   * Construct an Element instance from its atomic number.
   *
   * \param num the atomic number of the element, in range [0,103] (0 is the
   * dummy element)
   *
   * \throws std::out_of_range if num is outside [0,103]
   */
  Element(int num);

  /**
   * The Element symbol e.g. `"H", "He", "Li"`
   *
   * \returns string representation of the element symbol, capitalized.
   *
   */
  inline const std::string &symbol() const { return m_data->symbol; }

  /**
   * The Element name e.g. `"hydrogen", "helium", "lithium"`
   *
   * \returns string representation of the element name, lower case.
   *
   */
  inline const std::string &name() const { return m_data->name; }

  /**
   * The average isotopic mass of this Element e.g.
   *
   * `1.00794f, 4.00262f`
   *
   * \returns a float representing the average isotopic mass in atomic mass
   * units.
   *
   */
  inline float mass() const { return m_data->mass; }

  /**
   * The covalent radius of this Element.
   *
   * \returns a float representing the covalent radius in Angstroms.
   *
   */
  inline float covalent_radius() const { return m_data->cov_radius; }

  /**
   * The van der Waals radius of this Element.
   *
   * \returns a float representing the van der Waals radius in Angstroms.
   *
   */
  inline float van_der_waals_radius() const { return m_data->vdw_radius; }

  /**
   * The atomic number this Element.
   *
   * \returns an int representing the atomic number.
   *
   */
  inline int atomic_number() const { return m_data->atomic_number; }

  /**
   *
   * Overload of the < operator
   *
   * This operator compares two elements by their atomic number
   *
   * \param rhs another Element for comparison
   *
   * \returns True if this Element has an atomic number lower than rhs
   *
   */
  bool operator<(const Element &rhs) const {
    return m_data->atomic_number < rhs.m_data->atomic_number;
  }

  /**
   *
   * Overload of the > operator
   *
   * This operator compares two elements by their atomic number
   *
   * \param rhs another Element for comparison
   *
   * \returns True if this Element has an atomic number greater than rhs
   *
   */
  bool operator>(const Element &rhs) const {
    return m_data->atomic_number > rhs.m_data->atomic_number;
  }

  /**
   *
   * Overload of the == operator
   *
   * This operator compares two elements by their atomic number to determine
   * if they are equal
   *
   * \param rhs another Element for comparison
   *
   * \returns True if this Element has the same atomic number as rhs
   *
   */
  bool operator==(const Element &rhs) const {
    return m_data->atomic_number == rhs.m_data->atomic_number;
  }

  /**
   *
   * Overload of the != operator
   *
   * This operator compares two elements by their atomic number to determine
   * if they are equal
   *
   * \param rhs another Element for comparison
   *
   * \returns True if this Element has a different atomic number to rhs
   *
   */
  bool operator!=(const Element &rhs) const {
    return m_data->atomic_number != rhs.m_data->atomic_number;
  }

  /**
   *
   * The free-atom atom polarizibility of this Element
   *
   * Data taken from Thakkar
   *
   * \returns a double representing the atomic polariziblity of an isolated
   * atom of this element in its neutral (ground) state.
   *
   */
  double polarizability(bool charged = false) const;

private:
  /// \internal
  const ElementData *m_data;
};

/**
 *
 * The chemical formula of a given std::vector of Elements e.g. `"H2O"`
 *
 * The result is in plain ASCII characters (i.e. no subscripts) and elements
 * with a count of 1 the number suffix is omitted. i.e. `H2O` not `H2O1`
 *
 * \param elements a std::vector of Elements used to calculate the chemical
 * formula
 *
 * \returns a string representing the chemical formula
 *
 */

std::string chemical_formula(const std::vector<Element> &elements);

/**
 * Calculate the total atomic mass of a collection of atoms.
 *
 * \param atomic_numbers Vector of atomic numbers
 * \returns Total mass in atomic mass units (u)
 *
 * Sums the atomic masses of all atoms specified by their atomic numbers.
 * Useful for calculating molecular weights or crystal densities.
 */
double total_atomic_mass(const IVec &atomic_numbers);

} // namespace occ::core
