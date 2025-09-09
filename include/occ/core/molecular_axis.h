#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::core {

/**
 * @brief Enumeration of molecular axis calculation methods
 */
enum class AxisMethod {
    None,       ///< No axis calculation
    Neighcrys,  ///< Neighcrys method using 3 specified atoms
    PCA,        ///< Principal component analysis
    MOI         ///< Moment of inertia tensor eigenvectors
};

/**
 * @brief Structure to hold molecular axis calculation results
 */
struct MolecularAxisResult {
    Mat3 axes;                ///< 3x3 rotation matrix (axes as rows)
    Vec3 center_of_mass;      ///< Center of mass position
    std::vector<int> axis_atoms; ///< Atoms used for axis definition (for neighcrys method)
    AxisMethod method;        ///< Method used for calculation
    double determinant;       ///< Determinant of axes matrix (should be +1 for right-handed)
};

/**
 * @brief Structure for neighcrys-compatible axis file information
 */
struct NeighcrysAxisInfo {
    std::vector<std::string> atom_labels;  ///< Neighcrys-compatible atom labels
    std::vector<int> separations;          ///< Bond path separations
    std::vector<int> axis_atoms;           ///< Atoms defining the axis system
};

/**
 * @brief Molecular axis calculator class
 */
class MolecularAxisCalculator {
public:
    /**
     * @brief Construct a molecular axis calculator
     * @param wfn The wavefunction to analyze
     */
    explicit MolecularAxisCalculator(const occ::qm::Wavefunction& wfn);

    /**
     * @brief Calculate molecular axes using neighcrys method
     * @param axis_atoms Vector of 3 atom indices (0-based) defining the axis system
     * @return MolecularAxisResult containing axes and metadata
     */
    MolecularAxisResult calculate_neighcrys_axes(const std::vector<int>& axis_atoms) const;

    /**
     * @brief Calculate molecular axes using PCA method
     * @return MolecularAxisResult containing axes and metadata
     */
    MolecularAxisResult calculate_pca_axes() const;

    /**
     * @brief Calculate molecular axes using moment of inertia method
     * @return MolecularAxisResult containing axes and metadata
     */
    MolecularAxisResult calculate_moi_axes() const;

    /**
     * @brief Calculate molecular axes using specified method
     * @param method The axis calculation method to use
     * @param axis_atoms Atom indices for neighcrys method (ignored for other methods)
     * @return MolecularAxisResult containing axes and metadata
     */
    MolecularAxisResult calculate_axes(AxisMethod method, 
                                      const std::vector<int>& axis_atoms = {}) const;

    /**
     * @brief Get center of mass of the molecule
     * @return Center of mass position vector
     */
    Vec3 center_of_mass() const;

    /**
     * @brief Generate neighcrys-compatible atom labels
     * @return Vector of neighcrys-style atom labels
     */
    std::vector<std::string> generate_neighcrys_labels() const;

    /**
     * @brief Calculate bond path separation between two atoms
     * @param atom_i First atom index
     * @param atom_j Second atom index
     * @return Estimated bond path separation
     */
    int calculate_bond_separation(int atom_i, int atom_j) const;

    /**
     * @brief Generate neighcrys axis information
     * @param axis_atoms Atoms defining the axis system
     * @return NeighcrysAxisInfo structure
     */
    NeighcrysAxisInfo generate_neighcrys_info(const std::vector<int>& axis_atoms) const;

    /**
     * @brief Apply molecular transformation to wavefunction
     * @param wfn Wavefunction to transform (modified in place)
     * @param result Axis calculation result containing transformation
     */
    static void apply_molecular_transformation(occ::qm::Wavefunction& wfn, 
                                             const MolecularAxisResult& result);

    /**
     * @brief Write neighcrys-compatible axis file
     * @param filename Output filename
     * @param axis_info Neighcrys axis information
     */
    static void write_neighcrys_axis_file(const std::string& filename, 
                                        const NeighcrysAxisInfo& axis_info);

    /**
     * @brief Write oriented molecule in XYZ format
     * @param filename Output filename
     * @param wfn Wavefunction (should be oriented)
     * @param title Optional title for XYZ file
     */
    static void write_oriented_xyz(const std::string& filename, 
                                 const occ::qm::Wavefunction& wfn,
                                 const std::string& title = "Oriented molecule from OCC");

    /**
     * @brief Convert AxisMethod enum to string
     * @param method The axis method
     * @return String representation
     */
    static std::string axis_method_to_string(AxisMethod method);

    /**
     * @brief Convert string to AxisMethod enum
     * @param method_str String representation
     * @return AxisMethod enum value
     */
    static AxisMethod string_to_axis_method(const std::string& method_str);

private:
    const occ::qm::Wavefunction& m_wfn;
};

} // namespace occ::core