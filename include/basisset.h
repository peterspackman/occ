#pragma once
#include <libint2/cxxapi.h>

#include <cerrno>
#include <iostream>
#include <fstream>
#include <locale>
#include <vector>
#include <stdexcept>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <libint2/shell.h>
#include <libint2/atom.h>
#include "linear_algebra.h"

/* this has been modified from libint2::BasisSet
 * due to requirements in construction etc that were not possible
 * with the existing basisset
 */

namespace tonto::qm {

using libint2::Atom;
using libint2::Shell;
using libint2::svector;

/// Computes the number of basis functions in a range of shells
/// @tparam a range type
/// @param[in] shells a sequence of shells
/// @return the number of basis functions
template <typename ShellRange> size_t nbf(ShellRange && shells) {
  size_t n = 0;
  for (const auto& shell: std::forward<ShellRange>(shells))
    n += shell.size();
  return n;
}

/// Computes the maximum number of primitives in any Shell among a range of shells
/// @tparam a range type
/// @param[in] shells a sequence of shells
/// @return the maximum number of primitives
template <typename ShellRange> size_t max_nprim(ShellRange && shells) {
  size_t n = 0;
  for (auto shell: std::forward<ShellRange>(shells))
    n = std::max(shell.nprim(), n);
  return n;
}

/// Computes the maximum angular momentum quantum number @c l in any Shell among a range of shells
/// @tparam a range type
/// @param[in] shells a sequence of shells
/// @return the maximum angular momentum
template <typename ShellRange> int max_l(ShellRange && shells) {
  int l = 0;
  for (auto shell: std::forward<ShellRange>(shells))
    for (auto c: shell.contr)
      l = std::max(c.l, l);
  return l;
}

/// BasisSet is a slightly decorated \c std::vector of \c libint2::Shell objects.
class BasisSet : public std::vector<libint2::Shell> {
  public:
    BasisSet() : m_name(""), m_nbf(-1), m_max_nprim(0), m_max_l(-1) {}
    BasisSet(const BasisSet&) = default;
    BasisSet(BasisSet&& other) :
      std::vector<libint2::Shell>(std::move(other)),
      m_name(std::move(other.m_name)),
      m_nbf(other.m_nbf),
      m_max_nprim(other.m_max_nprim),
      m_max_l(other.m_max_l),
      m_shell2bf(std::move(other.m_shell2bf))
    {
    }
    ~BasisSet() = default;
    BasisSet& operator=(const BasisSet&) = default;

    /// @brief Construct from the basis set name and a vector of atoms.

    /**
     * @param[in] name the basis set name
     * @param[in] atoms \c std::vector of Atom objects
     * @param[in] throw_if_no_match If true, and the basis is not found for this atomic number, throw a std::logic_error.
     *       Otherwise omit the basis quietly.
     * @throw std::logic_error if throw_if_no_match is true and for at least one atom no matching basis is found.
     * @throw std::ios_base::failure if throw_if_no_match is true and the basis file could not be read
     * \note All instances of the same chemical element receive the same basis set.
     * \note \c name will be "canonicalized" using BasisSet::canonicalize(name) to
     *       produce the file name where the basis will be sought. This file needs to contain
     *       the basis set definition in Gaussian94 format (see \c lib/basis directory for examples).
     *       The expected location of the file is determined by BasisSet::data_path as follows:
     *       <ol>
     *         <li> specified by LIBINT_DATA_PATH environmental variable, if defined </li>
     *         <li> specified by DATADIR macro variable, if defined </li>
     *         <li> specified by SRCDATADIR macro variable, if defined </li>
     *         <li> hardwired to directory \c /usr/local/share/libint/2.7.0/basis </li>
     *       </ol>
     */
    BasisSet(std::string name,
             const std::vector<Atom>& atoms,
             const bool throw_if_no_match = false);

    /// @brief Construct from a vector of atoms and the per-element basis set specification

    /**
     * @param[in] atoms \c std::vector of Atom objects
     * @param[in] name the basis set name
     * @param[in] throw_if_no_match If true, and the basis is not found for this atomic number, throw a std::logic_error.
     *       Otherwise omit the basis quietly.
     * @throw std::logic_error if throw_if_no_match is true and for at least one atom no matching basis is found.
     * \note All instances of the same chemical element receive the same basis set.
     */
    BasisSet(const std::vector<Atom>& atoms,
             const std::vector<std::vector<Shell>>& element_bases,
             std::string name = "",
             const bool throw_if_no_match = false);

    /// forces solid harmonics/Cartesian Gaussians
    /// @param solid if true, force all shells with L>1 to be solid harmonics, otherwise force all shells to Cartesian
    void set_pure(bool solid) {
      for(auto& s: *this) {
        s.contr[0].pure = solid;
      }
      update();
    }

    bool is_pure() const
    {
        for(const auto& s: *this)
        {
            if (s.contr[0].pure) return true;
        }
        return false;
    }

    /// @return the number of basis functions in the basis; -1 if uninitialized
    long nbf() const {
      return m_nbf;
    }
    /// @return the maximum number of primitives in a contracted Shell, i.e. maximum contraction length; 0 if uninitialized
    size_t max_nprim() const {
      return m_max_nprim;
    }
    /// @return the maximum angular momentum of a contraction; -1 if uninitialized
    long max_l() const {
      return m_max_l;
    }
    /// @return the map from shell index to index of the first basis function from this shell
    /// \note basis functions are ordered as shells, i.e. shell2bf[i] >= shell2bf[j] iff i >= j
    const std::vector<size_t>& shell2bf() const {
      return m_shell2bf;
    }
    /// Computes the map from this object's shells to the corresponding atoms in \c atoms. If no atom matches the origin of a shell, it is mapped to -1.
    /// @note coordinates must match \em exactly , i.e. shell2atom[k] == l iff atoms[l].x == *this[k].O[0] && atoms[l].y == *this[k].O[1] &&  atoms[l].z == *this[k].O[2]
    /// @return the map from shell index to the atom in the list \c atoms that coincides with its origin;
    std::vector<long> shell2atom(const std::vector<Atom>& atoms) const {
      return shell2atom(*this, atoms, false);
    }
    /// Computes the map from \c atoms to the corresponding shells in this object. Coordinates are compared bit-wise (@sa BasisSet::shell2atom() )
    /// @return the map from atom index to the vector of shell indices whose origins conincide with the atom;
    /// @note this does not assume that \c shells are ordered in the order of atoms, as does BasisSet
    std::vector<std::vector<long>> atom2shell(const std::vector<Atom>& atoms) const {
      return atom2shell(atoms, *this);
    }

    /// Computes the map from \c shells to the corresponding atoms in \c atoms. Coordinates are compared bit-wise, i.e.
    /// shell2atom[k] == l iff atoms[l].x == *this[k].O[0] && atoms[l].y == *this[k].O[1] &&  atoms[l].z == *this[k].O[2]
    /// @param throw_if_no_match If true, and no atom matches the origin of a shell, throw a std::logic_error.
    ///        Otherwise such shells will be mapped to -1.
    /// @return the map from shell index to the atom in the list \c atoms that coincides with its origin;
    /// @throw std::logic_error if throw_if_no_match is true and for at least one shell no matching atom is found.
    static std::vector<long> shell2atom(const std::vector<Shell>& shells,
                                        const std::vector<Atom>& atoms,
                                        bool throw_if_no_match = false);

    /// Computes the map from \c atoms to the corresponding shells in \c shells. Coordinates are compared bit-wise (@sa BasisSet::shell2atom() )
    /// @return the map from atom index to the vector of shell indices whose origins conincide with the atom;
    /// @note this does not assume that \c shells are ordered in the order of atoms, as does BasisSet
    static std::vector<std::vector<long>> atom2shell(const std::vector<Atom>& atoms,
                                                     const std::vector<Shell>& shells);

    void update();

    void rotate(const tonto::Mat3& rotation) {
        for(auto& shell: *this)
        {
            Eigen::Map<tonto::Vec3, 0> pos(shell.O.data());
            pos = rotation * pos;
        }
    }

  private:
    std::string m_name;
    long m_nbf;
    size_t m_max_nprim;
    int m_max_l;
    std::vector<size_t> m_shell2bf;





    static std::string canonicalize_name(const std::string& name);

    // see http://gaussian.com/basissets/
    bool gaussian_cartesian_d_convention(const std::string& canonical_name);

    /// decompose basis set name into components
    std::vector<std::string> decompose_name_into_components(std::string name);

    /** determines the path to the data directory, as follows:
     *       <ol>
     *         <li> specified by LIBINT_DATA_PATH environmental variable, if defined </li>
     *         <li> specified by DATADIR macro variable, if defined </li>
     *         <li> specified by SRCDATADIR macro variable, if defined </li>
     *         <li> hardwired to directory \c /usr/local/share/libint/2.7.0/basis </li>
     *       </ol>
     *  @throw std::system_error if the path is not valid, or cannot be determined
     *  @return valid path to the data directory
     */
    static std::string data_path();

    /// converts fortran scientific-notation floats that use d/D instead of e/E in \c str
    /// @param[in,out] str string in which chars 'd' and 'D' are replaced with 'e' and 'E',
    ///                respectively
    static void fortran_dfloats_to_efloats(std::string& str);

  public:

    /** reads in all basis sets from a Gaussian94-formatted basis set file (see https://bse.pnl.gov/bse/portal)
     *  @param[in] file_dot_g94 file name
     *  @param[in] force_cartesian_d force use of Cartesian d shells, if true
     *  @param[in] locale_name specifies the locale to use
     *  @throw std::ios_base::failure if the path is not valid, or cannot be determined
     *  @throw std::logic_error if the contents of the file cannot be interpreted
     *  @return vector of basis sets for each element
     *  @warning the included library basis sets should be parsed using POSIX locale
     */
    static std::vector<std::vector<libint2::Shell>> read_g94_basis_library(std::string file_dot_g94,
                                                                           bool force_cartesian_d = false,
                                                                           bool throw_if_missing = true,
                                                                           std::string locale_name = std::string("POSIX"));

}; // BasisSet

tonto::MatRM rotate_molecular_orbitals(const BasisSet&, const tonto::Mat3&, const tonto::MatRM&);

void rotate_atoms(std::vector<libint2::Atom>& atoms, const tonto::Mat3& rotation);

inline std::vector<libint2::Atom> rotated_atoms(const std::vector<libint2::Atom>& atoms, const tonto::Mat3& rotation)
{
    auto result = atoms;
    rotate_atoms(result, rotation);
    return result;
}

}
