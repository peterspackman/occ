#pragma once
#include <occ/mults/short_range.h>
#include <occ/core/linear_algebra.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace occ::mults {

/// Forward declaration of ForceFieldType (defined in crystal_energy.h for now).
enum class ForceFieldType;

/**
 * @brief Manages force field parameters for short-range interactions.
 *
 * Handles element-based Buckingham, type-code-based Buckingham,
 * anisotropic repulsion, and Williams DE built-in tables.
 */
class ForceFieldParams {
public:
    // ==================== Element-based Buckingham ====================

    void set_buckingham(int Z1, int Z2, const BuckinghamParams& p);
    BuckinghamParams get_buckingham(int Z1, int Z2) const;
    bool has_buckingham(int Z1, int Z2) const;

    // ==================== Type-code-based Buckingham ====================

    void set_typed_buckingham(int type1, int type2, const BuckinghamParams& p);
    void set_typed_buckingham(const std::map<std::pair<int,int>, BuckinghamParams>& params);
    void clear_typed_buckingham();
    bool has_typed_buckingham(int type1, int type2) const;
    BuckinghamParams get_typed_buckingham(int type1, int type2) const;

    /// Look up typed params, falling back to element-based if not found.
    BuckinghamParams get_buckingham_for_types(int type1, int type2) const;

    // ==================== Anisotropic repulsion ====================

    void set_typed_aniso(const std::map<std::pair<int,int>, AnisotropicRepulsionParams>& params);
    bool has_aniso(int type1, int type2) const;
    AnisotropicRepulsionParams get_aniso(int type1, int type2) const;
    bool has_any_aniso() const { return !m_typed_aniso_params.empty(); }

    // ==================== Type labels ====================

    void set_type_labels(const std::map<int, std::string>& labels);
    std::string type_name(int type_code) const;

    // ==================== LJ params (legacy) ====================

    const std::map<std::pair<int,int>, LennardJonesParams>& lj_params() const {
        return m_lj_params;
    }

    // ==================== State flags ====================

    bool use_williams_atom_typing() const { return m_use_williams_atom_typing; }
    bool use_short_range_typing() const { return m_use_short_range_typing; }

    void set_use_williams_atom_typing(bool v) { m_use_williams_atom_typing = v; }
    void set_use_short_range_typing(bool v) { m_use_short_range_typing = v; }

    // ==================== Williams DE built-in tables (static) ====================

    /// Williams DE Buckingham parameters by element pair (kJ/mol, Angstrom).
    static std::map<std::pair<int,int>, BuckinghamParams> williams_de_params();

    /// Williams typed Buckingham parameters by type code pair (kJ/mol, Angstrom).
    static std::map<std::pair<int,int>, BuckinghamParams> williams_typed_params();

    /// Convert a Williams/NEIGHCRYS type code to a short label (e.g. 512 -> "C_W3").
    static const char* short_range_type_label(int type_code);

    /// Map a Williams/NEIGHCRYS type code to element Z, or 0 if unknown.
    static int short_range_type_atomic_number(int type_code);

    // ==================== Williams atom type classification ====================

    /// Classify atom types using Williams bonding rules.
    /// Updates short_range_type_codes in-place.
    static int classify_williams_type(
        int idx,
        const std::vector<std::vector<int>>& neighbors,
        const std::vector<int>& atomic_numbers);

    /// Find bonded neighbors by covalent radius + tolerance.
    static std::vector<std::vector<int>> bonded_neighbors(
        const std::vector<int>& atomic_numbers,
        const std::vector<Vec3>& positions);

private:
    std::map<std::pair<int,int>, BuckinghamParams> m_buckingham_params;
    std::map<std::pair<int,int>, BuckinghamParams> m_typed_buckingham_params;
    std::map<int, std::string> m_short_range_type_labels;
    mutable std::set<std::pair<int,int>> m_missing_buckingham_warned;
    mutable std::set<std::pair<int,int>> m_missing_typed_buckingham_warned;
    std::map<std::pair<int,int>, LennardJonesParams> m_lj_params;
    std::map<std::pair<int,int>, AnisotropicRepulsionParams> m_typed_aniso_params;
    bool m_use_williams_atom_typing = false;
    bool m_use_short_range_typing = false;
};

} // namespace occ::mults
