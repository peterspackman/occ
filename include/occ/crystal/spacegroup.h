#pragma once

#include <gemmi/symmetry.hpp>
#include <map>
#include <occ/crystal/hkl.h>
#include <occ/crystal/symmetryoperation.h>
#include <string>
#include <tuple>
#include <vector>

namespace occ::crystal {

using occ::IVec;
using occ::Mat3N;

using SGData = gemmi::SpaceGroup;

class ReciprocalAsymmetricUnit {
  public:
    ReciprocalAsymmetricUnit(const SGData *sg) : m_asu(sg) {}
    bool is_in(const HKL &hkl) const {
        return m_asu.is_in({hkl.h, hkl.k, hkl.l});
    }

  private:
    gemmi::ReciprocalAsu m_asu;
};

/**
 * This class represents a space group.
 *
 * A space group describes the symmetries within a crystal structure
 * of a crystal structure. Provides methods for accessing and
 * manipulating space group data.
 */
class SpaceGroup {
  public:
    /**
     * Constructs a space group with the given space group number.
     *
     * \param num The space group number.
     *
     * \throws std::invalid_argument if the given number is not a valid space
     * group number.
     */
    SpaceGroup(int num);

    /**
     * \brief Constructs a space group with the given space symbol.
     *
     * \param symbol The space group symbol.
     *
     * \throws std::invalid_argument if the given space group can't be found
     */
    SpaceGroup(const std::string &symbol);

    /**
     * \brief Constructs a space group with the list of symmetry operations
     * in their string form, if the list of symops is not a known space group,
     * it will still construct the object and may be used (even though it is not
     * necessarily valid)
     *
     * \param symops The space group symmetry operations
     */
    SpaceGroup(const std::vector<std::string> &symops);

    /**
     * \brief Constructs a space group with the list of symmetry operations.
     * if the list of symops is not a known space group,
     * it will still construct the object and may be used (even though it is not
     * necessarily valid)
     *
     * \param symops The space group symmetry operations
     */
    SpaceGroup(const std::vector<SymmetryOperation> &symops);

    /**
     * \brief Returns the space group number of this space group.
     * \return The space group number.
     * \example
     * SpaceGroup sg(3);
     * int num = sg.number(); // num == 3
     */
    int number() const;

    /**
     * \brief Returns the Hermann-Mauguin (international tables) symbol for this
     * object
     * \return The space group symbol.
     * \example
     * SpaceGroup sg(3);
     * std::string symbol = sg.symbol(); // symbol == "P 1 2 1"
     */
    const std::string &symbol() const;

    /**
     * \brief Returns the Hermann-Mauguin (international tables) symbol for this
     * object, shortened e.g P 1 2 1 -> P2
     * \return The space group name.
     * \example
     * SpaceGroup sg(3);
     * std::string name = sg.short_name(); // name == "P2"
     */
    const std::string &short_name() const;

    /**
     * \brief Returns the list of symmetry operations for this space group.
     *
     * \return The list of symmetry operations.
     *
     * \example
     * SpaceGroup sg(3);
     * std::vector<SymmetryOperation> ops = sg.symmetry_operations();
     * for (const SymmetryOperation &op : ops) {
     *   // do something with each symmetry operation
     * }
     */
    const std::vector<SymmetryOperation> &symmetry_operations() const;

    /**
     * \brief Determine whether this space group has the choice between
     * hexagonal (H) and rhombohedral (R) settings.
     *
     * \return true if there's a choice to be made, false otherwise
     *
     * \example
     * bool x = SpaceGroup(3).has_H_R_choice(); // x == false
     * bool y = SpaceGroup(169).has_H_R_choice(); // y == true
     */
    bool has_H_R_choice() const;

    /**
     * \brief Apply all symmetry operations to a provided set of fractional
     * coordinates, with the identity symop always first
     *
     * \return a pair of `IVec`, `Mat3N` corresponding to the integer
     * representation `SymmetryOperation::to_int` of the symop applied to each
     * point, and the points after that symop application. If N points were
     * provided and this SpaceGroup has 8 symmetry operations, then the
     * resulting `IVec` will have dimension (N*8,) and the `Mat3N` of
     * coordinates will have dimension (3, N*8).
     *
     * \example
     * Mat3N points(3, 100); // unit cell atomic positions  or similar
     * auto [symop_id, transformed] =
     * SpaceGroup(14).apply_all_symmetry_operations(points);
     */
    std::pair<IVec, Mat3N> apply_all_symmetry_operations(const Mat3N &) const;

    inline auto reciprocal_asu() const {
        return ReciprocalAsymmetricUnit(m_sgdata);
    }

  private:
    void update_from_sgdata();
    int m_number{0};
    std::string m_symbol{"XX"};
    std::string m_short_name{"unknown"};
    const SGData *m_sgdata{nullptr};
    std::vector<SymmetryOperation> m_symops;
};

} // namespace occ::crystal
