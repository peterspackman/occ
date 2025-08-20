#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <vector>
#include <string>

namespace occ::core {

/**
 * @brief Results from vibrational frequency analysis
 */
struct VibrationalModes {
    Vec frequencies_cm;        // Frequencies in cm⁻¹
    Vec frequencies_hartree;   // Frequencies in Hartree
    Mat normal_modes;          // Normal mode vectors (3N×3N)
    Mat mass_weighted_hessian; // Mass-weighted Hessian matrix
    Mat hessian;              // Original Hessian matrix (optional, for storage)
    
    size_t n_modes() const { return frequencies_cm.size(); }
    size_t n_atoms() const { return frequencies_cm.size() / 3; }
    
    /**
     * @brief Format a summary of vibrational analysis results
     * @return Formatted string with analysis summary
     */
    std::string summary_string() const;
    
    /**
     * @brief Format detailed vibrational frequencies
     * @return Formatted string with frequency table
     */
    std::string frequencies_string() const;
    
    /**
     * @brief Format normal mode vectors
     * @param threshold Only include components larger than this threshold
     * @return Formatted string with normal mode displacements
     */
    std::string normal_modes_string(double threshold = 0.1) const;
    
    /**
     * @brief Log vibrational analysis summary using occ::log
     */
    void log_summary() const;
    
    /**
     * @brief Get all frequencies as a sorted vector
     * @return Vector of all frequencies in cm⁻¹ (sorted)
     */
    Vec get_all_frequencies() const;
};

/**
 * @brief Perform vibrational frequency analysis on a Hessian matrix
 * 
 * @param hessian 3N×3N Hessian matrix in atomic units (Hartree/Bohr²)
 * @param masses Vector of atomic masses in AMU (length N for N atoms)
 * @param positions 3×N matrix of atomic positions in Bohr (optional, for projection)
 * @param project_tr_rot Project out translational and rotational modes (like ORCA's PROJECTTR)
 * @return VibrationalModes Complete vibrational analysis results
 */
VibrationalModes compute_vibrational_modes(const Mat &hessian, 
                                         const Vec &masses,
                                         const Mat3N &positions = Mat3N(),
                                         bool project_tr_rot = false);

/**
 * @brief Convenience function for molecular vibrational analysis
 * 
 * @param hessian 3N×3N Hessian matrix in atomic units (Hartree/Bohr²)
 * @param molecule Molecule object containing atomic masses and positions
 * @param project_tr_rot Project out translational and rotational modes (like ORCA's PROJECTTR)
 * @return VibrationalModes Complete vibrational analysis results
 */
VibrationalModes compute_vibrational_modes(const Mat &hessian, 
                                         const Molecule &molecule,
                                         bool project_tr_rot = false);

/**
 * @brief Construct mass-weighted Hessian matrix
 * 
 * The mass-weighted Hessian is constructed as:
 * H_mw[3i+a][3j+b] = H[3i+a][3j+b] / sqrt(m_i * m_j)
 * where m_i is the mass of atom i in AMU
 * 
 * @param hessian 3N×3N Hessian matrix in atomic units
 * @param masses Vector of atomic masses in AMU (length N for N atoms)
 * @return Mass-weighted Hessian matrix
 */
Mat mass_weighted_hessian(const Mat &hessian, const Vec &masses);

/**
 * @brief Convenience function for molecular mass-weighted Hessian
 * 
 * @param hessian 3N×3N Hessian matrix in atomic units
 * @param molecule Molecule object containing atomic masses
 * @return Mass-weighted Hessian matrix
 */
Mat mass_weighted_hessian(const Mat &hessian, const Molecule &molecule);

/**
 * @brief Convert frequency eigenvalues to cm⁻¹
 * 
 * For positive eigenvalues: ν = sqrt(λ) * conversion_factor
 * For negative eigenvalues: ν = -sqrt(-λ) * conversion_factor (imaginary frequencies)
 * 
 * @param eigenvalues Eigenvalues from mass-weighted Hessian (Hartree/AMU/Bohr²)
 * @return Frequencies in cm⁻¹
 */
Vec eigenvalues_to_frequencies_cm(const Vec &eigenvalues);

/**
 * @brief Convert frequencies from cm⁻¹ to Hartree
 * 
 * @param frequencies_cm Frequencies in cm⁻¹
 * @return Frequencies in Hartree
 */
Vec frequencies_cm_to_hartree(const Vec &frequencies_cm);


/**
 * @brief Construct translational projection vectors
 * 
 * @param masses Vector of atomic masses in AMU (length N for N atoms)
 * @return 3N×3 matrix with translation vectors as columns
 */
Mat construct_translation_vectors(const Vec &masses);

/**
 * @brief Construct rotational projection vectors
 * 
 * @param masses Vector of atomic masses in AMU (length N for N atoms)
 * @param positions 3×N matrix of atomic positions in Bohr
 * @return 3N×3 matrix with rotation vectors as columns
 */
Mat construct_rotation_vectors(const Vec &masses, const Mat3N &positions);

/**
 * @brief Project out translational and rotational modes from mass-weighted Hessian
 * 
 * @param mass_weighted_hessian Mass-weighted Hessian matrix
 * @param masses Vector of atomic masses in AMU (length N for N atoms)
 * @param positions 3×N matrix of atomic positions in Bohr
 * @return Projected Hessian matrix
 */
Mat project_tr_rot_modes(const Mat &mass_weighted_hessian, const Vec &masses, const Mat3N &positions);

/**
 * @brief Convenience functions for molecular systems
 */
Mat construct_translation_vectors(const Molecule &molecule);
Mat construct_rotation_vectors(const Molecule &molecule);
Mat project_tr_rot_modes(const Mat &mass_weighted_hessian, const Molecule &molecule);

} // namespace occ::core