#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <optional>

namespace occ::core {
using occ::IVec;
using occ::Mat3N;
using occ::Vec;
using occ::core::Molecule;

/**
 * Storage class for relevant information of a dimer (pair) of Molecule objects.
 *
 * The role of the Dimer class is to store information about a pair of molecules
 * which represents a dimer, and to calculate relevant properties.
 *
 */
class Dimer {
public:
  /**
   * An enum to clarify the order of information presented when
   * for merged arrays e.g. atomic_numbers(), positions() etc.
   */
  enum class MoleculeOrder : bool {
    AB = true, /**< The A molecule data is first, then B */
    BA = false /**< The B molecule data is first, then A */
  };

  // Default (empty) constructor.
  Dimer() = default;

  /**
   * Constructor from two Molecule objects
   *
   * \param mol_A Molecule A
   * \param mol_B Molecule B
   *
   * A copy of these two molecules is kept internally, so they need
   * not exist outside this Dimer. Further, any changes to those Molecule
   * objects after Dimer construction will not appear here.
   */
  Dimer(const Molecule &mol_A, const Molecule &mol_B);

  /**
   * Constructor from two std::vector<Atom> objects
   *
   * \param atoms_A atoms representing/constituting Molecule A
   * \param atoms_B atoms representing/constituting Molecule B
   *
   * The data from these two vectors is copied internally, so they need
   * not exist outside this Dimer. Further, any changes to those Atom objects
   * after Dimer construction will not appear here.
   */
  Dimer(const std::vector<occ::core::Atom> &atoms_A,
        const std::vector<occ::core::Atom> &atoms_B);

  Vec3 centroid() const;

  /**
   * Convenience wrapper to access the Molecule instance for Molecule A
   *
   * \returns Molecule A
   */
  const Molecule &a() const { return m_a; }

  /**
   * Convenience wrapper to access the Molecule instance for Molecule B
   *
   * \returns Molecule A
   */
  const Molecule &b() const { return m_b; }

  /**
   * Calculate the centroid - centroid distance between A & B
   *
   * \returns a double representing the distance (in Angstroms)
   * between the centroid of Molecule A and Molecule B. The centroid
   * is the geometric average positions (it is not the centre of mass).
   *
   */
  double centroid_distance() const;

  /**
   * Calculate the distance between centre of mass of Molecules A & B
   *
   * \returns a double representing the distance (in Angstroms)
   * between the centre of mass of Molecule A and centre of mass of Molecule
   * B.
   */
  double center_of_mass_distance() const;

  /**
   * Calculate the distance between closest pair of Atoms in Molecules A & B
   *
   * \returns a double representing the distance (in Angstroms)
   * between the nearest pair of Atoms (one from Molecule A & one from
   * Molecule B). \note This will be significantly more expensive and scale as
   * `O(Na * Nb)` for the number of atoms `Na` in Molecule A and `Nb` for the
   * number of atoms in Molecule B
   *
   */
  double nearest_distance() const;

  /**
   * The symmetry operation to transform from Molecule A to Molecule B,
   * if there is one.
   *
   * \returns either std::nullopt if Molecule A and Molecule B are not
   * symmetry related, or a Mat4 object containing the rotation and
   * translation to transform Molecule A to Molecule B.
   *
   * Under the hood, this uses the Kabsch algorithm
   */
  std::optional<occ::Mat4> symmetry_relation() const;

  /**
   * The vector from centroid of Molecule A to the centroid of Molecule B.
   *
   * \returns a Vec3 representing the vector between the centroid of A and
   * the centroid of B.
   */
  Vec3 v_ab() const;

  /**
   * Vector of van der Waals radii for the atoms in this Dimer.
   *
   * \param order specify whether the radii of the atoms in Molecule A should
   * be first, or those in Molecule B.
   *
   * \returns a Vec object representing the van der Waals radii of
   * atoms in the Dimer in the specified (AB or BA) order, in Angstroms.
   */
  Vec vdw_radii(MoleculeOrder order = MoleculeOrder::AB) const;

  /**
   * Vector of atomic numbers for the atoms in this Dimer.
   *
   * \param order specify whether the atomic numbers of the atoms in Molecule
   * A should be first, or those in Molecule B.
   *
   * \returns an IVec object representing the atomic numbers of
   * atoms in the Dimer in the specified (AB or BA) order.
   */
  IVec atomic_numbers(MoleculeOrder order = MoleculeOrder::AB) const;

  /**
   * The positions of all atoms in this Dimer.
   *
   * \param order specify whether the positions of the atoms in Molecule
   * A should be first, or those in Molecule B.
   *
   * \returns A `(3, N)` matrix of atomic positions for the atoms in this
   * dimer, in Angstroms
   */
  Mat3N positions(MoleculeOrder order = MoleculeOrder::AB) const;

  /**
   * The total number of electrons in this Dimer
   *
   * \returns an integer with value equal to the sum of the number of
   * electrons in Molecule A and the number of electrons in Molecule B.
   */
  inline int num_electrons() const {
    return m_a.num_electrons() + m_b.num_electrons();
  }

  /**
   * The net charge of this Dimer
   *
   * \returns an integer with value equal to the sum of the net charge of
   * Molecule A and the net charge of Molecule B.
   */
  inline int charge() const { return m_a.charge() + m_b.charge(); };

  /**
   * The net spin multiplicity of this Dimer
   *
   * \returns an integer with value equal to net spin multiplicity when
   * combining Molecule A and Molecule B.
   */
  inline int multiplicity() const {
    return m_a.multiplicity() + m_b.multiplicity() - 1;
  };

  /**
   * Set the index or identifier for which interaction this Dimer represents.
   *
   * \param i the interaction index.
   */
  inline void set_interaction_id(size_t i) { m_interaction_id = i; }

  /**
   * Get the index or identifier for which interaction this Dimer represents.
   *
   * \returns a size_t representing the interaction index (default = 0)
   */
  inline size_t interaction_id() const { return m_interaction_id; }

  /**
   * Set the value of the interaction energy for this Dimer
   *
   * \param e a double representing the interaction energy.
   * \param key a string labelling the component of the interaction energy.
   */
  void set_interaction_energy(double e, const std::string &key = "Total");

  /**
   * Get the stored interaction energy for this Dimer.
   *
   * \param key a string labelling the component of the interaction energy.
   *
   * \returns a double representing the interaction energy (default = 0.0)
   */
  double interaction_energy(const std::string &key = "Total") const;

  inline void set_interaction_energies(
      const ankerl::unordered_dense::map<std::string, double> &e) {
    m_interaction_energies = e;
  }

  /**
   * Get the stored interaction energy components for this Dimer.
   *
   * \returns a map representing the interaction energies, index by string
   * identifiers
   */
  inline const auto &interaction_energies() const {
    return m_interaction_energies;
  }

  /**
   * Check if two dimers have the same asymmetric molecul indexes
   *
   * \param rhs another Dimer object
   * \returns True if all the asymmetric Molecule indices of all Molecules
   * have been set, and they are equal (ignoring ordering of AB in either
   * Dimer).
   */
  bool same_asymmetric_molecule_idxs(const Dimer &rhs) const;

  /**
   * Check if two dimers are identical under some transformation.
   *
   * \param rhs another Dimer object
   * \returns True if the dimers are found to be identical.
   *
   * The working definition of equality here is such that all the following
   * must be true:
   *
   * 1. The asymmetric molecule indices must be the same i.e. calling
   * same_asymmetric_molecule_idxs yields true.
   * 2. Both Dimers have the same centroid_distance, center_of_mass_distance
   * and nearest_atom_distance
   * 3. Either this.a is equivalent to rhs.a and this.b is equivalent to
   * rhs.b, or this.a is equivalent to rhs.b and rhs.a is equivalent to
   * this.b. See Molecule::equivalent_to
   *
   */
  bool operator==(const Dimer &rhs) const;

  /**
   * Check if two dimers are not identical under some transformation.
   *
   * \param rhs another Dimer object
   * \returns True if the dimers are not found to be identical.
   *
   * Uses Dimer::operator== to find the result.
   *
   */
  inline bool operator!=(const Dimer &rhs) const { return !(*this == rhs); }

  /**
   * Check if two dimers are identical in the opposite reference frame
   *
   * \param b another Dimer object
   * \param rot a Matrix representing a rotation, applied to A.
   * \returns True if the Dimer objects are found to be equivalent.
   *
   * The opposite reference frame here is that Dimer b will be equivalent
   * if Molecule B from the reference frame of Molecule A is equivalent to
   * Molecule rhs.A in the reference frame of rhs.B
   */
  bool equivalent_in_opposite_frame(const Dimer &b,
                                    const Mat3 &rot = Mat3::Identity()) const;

  /**
   * Check if two dimers are identical in the same reference frame
   *
   * \param b another Dimer object
   * \param rot a Matrix representing a rotation, applied to A.
   * \returns True if the Dimer objects are found to be equivalent.
   *
   * This is defined such that that Dimer b will be equivalent
   * if Molecule B from the reference frame of Molecule A is equivalent to
   * Molecule rhs.B in the reference frame of rhs.A
   */
  bool equivalent(const Dimer &b, const Mat3 &rot = Mat3::Identity()) const;

  inline const auto &name() const { return m_name; }
  inline void set_name(const std::string &name) { m_name = name; }
  std::string xyz_string() const;

  /**
   * Set the value of a property for this Dimer
   *
   * \param key a string labelling the property.
   * \param value a string containing the value of the property
   */
  inline void set_property(const std::string &key, const std::string &value) {
    m_additional_properties[key] = value;
  }

  inline const auto &properties() const { return m_additional_properties; }

  inline void set_properties(
      const ankerl::unordered_dense::map<std::string, std::string> &p) {
    m_additional_properties = p;
  }

private:
  Molecule m_a, m_b;
  std::string m_name{"dimer"};
  size_t m_interaction_id{0};
  ankerl::unordered_dense::map<std::string, double> m_interaction_energies;
  ankerl::unordered_dense::map<std::string, std::string>
      m_additional_properties;
};

} // namespace occ::core
