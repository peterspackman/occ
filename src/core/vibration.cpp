#include <occ/core/vibration.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <Eigen/Eigenvalues>
#include <fmt/core.h>
#include <fmt/format.h>
#include <cmath>
#include <algorithm>

namespace occ::core {

VibrationalModes compute_vibrational_modes(const Mat &hessian, 
                                         const Vec &masses,
                                         const Mat3N &positions,
                                         bool project_tr_rot) {
    VibrationalModes result;
    
    // Store original Hessian
    result.hessian = hessian;
    
    // Construct mass-weighted Hessian
    result.mass_weighted_hessian = mass_weighted_hessian(hessian, masses);
    
    // Optionally project out translational and rotational modes
    Mat hessian_to_diagonalize = result.mass_weighted_hessian;
    if (project_tr_rot) {
        if (positions.size() == 0) {
            throw std::invalid_argument("Positions required for translational/rotational projection");
        }
        hessian_to_diagonalize = project_tr_rot_modes(result.mass_weighted_hessian, masses, positions);
    }
    
    // Diagonalize mass-weighted Hessian to get normal modes and frequencies
    Eigen::SelfAdjointEigenSolver<Mat> solver(hessian_to_diagonalize);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to diagonalize mass-weighted Hessian");
    }
    
    Vec eigenvalues = solver.eigenvalues();
    result.normal_modes = solver.eigenvectors();
    
    // Convert eigenvalues to frequencies
    result.frequencies_cm = eigenvalues_to_frequencies_cm(eigenvalues);
    result.frequencies_hartree = frequencies_cm_to_hartree(result.frequencies_cm);
    
    return result;
}

VibrationalModes compute_vibrational_modes(const Mat &hessian, 
                                         const Molecule &molecule,
                                         bool project_tr_rot) {
    // Extract masses from molecule
    const auto &atoms = molecule.atoms();
    Vec masses(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        masses[i] = Element(atoms[i].atomic_number).mass();
    }
    
    // Extract positions from molecule
    Mat3N positions = molecule.positions();
    
    return compute_vibrational_modes(hessian, masses, positions, project_tr_rot);
}

Mat mass_weighted_hessian(const Mat &hessian, const Vec &masses) {
    size_t natom = masses.size();
    size_t ndof = 3 * natom;
    
    if (hessian.rows() != ndof || hessian.cols() != ndof) {
        throw std::invalid_argument("Hessian matrix dimensions do not match number of atoms");
    }
    
    Mat result = Mat::Zero(ndof, ndof);
    
    // Construct mass-weighted Hessian: H_mw[3i+a][3j+b] = H[3i+a][3j+b] / sqrt(m_i * m_j)
    for (size_t i = 0; i < natom; i++) {
        double mass_i = masses[i]; // AMU
        
        for (size_t j = 0; j < natom; j++) {
            double mass_j = masses[j]; // AMU
            double mass_factor = 1.0 / std::sqrt(mass_i * mass_j);
            
            // Apply mass weighting to 3×3 block
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    size_t row = 3 * i + a;
                    size_t col = 3 * j + b;
                    result(row, col) = hessian(row, col) * mass_factor;
                }
            }
        }
    }
    
    return result;
}

Mat mass_weighted_hessian(const Mat &hessian, const Molecule &molecule) {
    // Extract masses from molecule
    const auto &atoms = molecule.atoms();
    Vec masses(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        masses[i] = Element(atoms[i].atomic_number).mass();
    }
    
    return mass_weighted_hessian(hessian, masses);
}

Vec eigenvalues_to_frequencies_cm(const Vec &eigenvalues) {
    Vec frequencies = Vec::Zero(eigenvalues.size());
    
    // Conversion factor from sqrt(Hartree/AMU/Bohr²) to cm⁻¹
    // eigenvalue units: Hartree/(AMU·Bohr²)
    // frequency units: 1/s = sqrt(Hartree/AMU)/Bohr
    // wavenumber units: 1/cm = frequency/(2πc)
    
    // sqrt(Hartree/AMU) / Bohr = sqrt(J/kg) / m = 1/s (angular frequency)
    // Then convert to wavenumber: ν̃ = ω / (2πc)
    const double conversion = std::sqrt(units::AU_TO_JOULE / units::AMU_TO_KG) / 
                             (units::BOHR_TO_METER * 2.0 * M_PI * units::SPEED_OF_LIGHT_CM_PER_S);
    
    for (int i = 0; i < eigenvalues.size(); i++) {
        double eigenval = eigenvalues[i];
        
        if (eigenval >= 0.0) {
            // Real frequency
            frequencies[i] = std::sqrt(eigenval) * conversion;
        } else {
            // Imaginary frequency (negative eigenvalue)
            frequencies[i] = -std::sqrt(-eigenval) * conversion;
        }
    }
    
    return frequencies;
}

Vec frequencies_cm_to_hartree(const Vec &frequencies_cm) {
    Vec frequencies_hartree = Vec::Zero(frequencies_cm.size());
    
    // Convert cm⁻¹ to Hartree: E = hc·ν̃
    // 1 cm⁻¹ = hc/cm in Joules = (hc/cm) / E_h in Hartree
    
    for (int i = 0; i < frequencies_cm.size(); i++) {
        frequencies_hartree[i] = frequencies_cm[i] * units::PER_CM_TO_AU;
    }
    
    return frequencies_hartree;
}


Mat construct_translation_vectors(const Vec &masses) {
    size_t natom = masses.size();
    size_t ndof = 3 * natom;
    
    Mat trans_vectors = Mat::Zero(ndof, 3);
    
    // Construct mass-weighted translation vectors
    for (size_t i = 0; i < natom; i++) {
        double mass_sqrt = std::sqrt(masses[i]);
        
        // Translation in x
        trans_vectors(3*i + 0, 0) = mass_sqrt;
        // Translation in y  
        trans_vectors(3*i + 1, 1) = mass_sqrt;
        // Translation in z
        trans_vectors(3*i + 2, 2) = mass_sqrt;
    }
    
    // Normalize translation vectors
    for (int col = 0; col < 3; col++) {
        double norm = trans_vectors.col(col).norm();
        if (norm > 1e-10) {
            trans_vectors.col(col) /= norm;
        }
    }
    
    return trans_vectors;
}

Mat construct_rotation_vectors(const Vec &masses, const Mat3N &positions) {
    size_t natom = masses.size();
    size_t ndof = 3 * natom;
    
    if (positions.cols() != natom || positions.rows() != 3) {
        throw std::invalid_argument("Positions matrix must be 3×N for N atoms");
    }
    
    Mat rot_vectors = Mat::Zero(ndof, 3);
    
    // Calculate center of mass
    Vec3 center_of_mass = Vec3::Zero();
    double total_mass = 0.0;
    for (size_t i = 0; i < natom; i++) {
        double mass = masses[i];
        center_of_mass += mass * positions.col(i);
        total_mass += mass;
    }
    center_of_mass /= total_mass;
    
    // Construct mass-weighted rotation vectors
    for (size_t i = 0; i < natom; i++) {
        double mass_sqrt = std::sqrt(masses[i]);
        Vec3 r = positions.col(i) - center_of_mass;
        
        // Rotation about x-axis: R × (1,0,0) = (0, z, -y)
        rot_vectors(3*i + 1, 0) = mass_sqrt * r[2];   // y component
        rot_vectors(3*i + 2, 0) = -mass_sqrt * r[1];  // z component
        
        // Rotation about y-axis: R × (0,1,0) = (-z, 0, x)
        rot_vectors(3*i + 0, 1) = -mass_sqrt * r[2];  // x component
        rot_vectors(3*i + 2, 1) = mass_sqrt * r[0];   // z component
        
        // Rotation about z-axis: R × (0,0,1) = (y, -x, 0)
        rot_vectors(3*i + 0, 2) = mass_sqrt * r[1];   // x component
        rot_vectors(3*i + 1, 2) = -mass_sqrt * r[0];  // y component
    }
    
    // Orthogonalize and normalize rotation vectors using Gram-Schmidt
    for (int col = 0; col < 3; col++) {
        // Remove components parallel to previous vectors
        for (int prev_col = 0; prev_col < col; prev_col++) {
            double dot = rot_vectors.col(col).dot(rot_vectors.col(prev_col));
            rot_vectors.col(col) -= dot * rot_vectors.col(prev_col);
        }
        
        // Normalize
        double norm = rot_vectors.col(col).norm();
        if (norm > 1e-10) {
            rot_vectors.col(col) /= norm;
        }
    }
    
    return rot_vectors;
}

Mat project_tr_rot_modes(const Mat &mass_weighted_hessian, const Vec &masses, const Mat3N &positions) {
    size_t ndof = mass_weighted_hessian.rows();
    
    // Construct translation and rotation vectors
    Mat trans_vectors = construct_translation_vectors(masses);
    Mat rot_vectors = construct_rotation_vectors(masses, positions);
    
    // Combine into projection matrix P (6 x ndof)
    Mat P = Mat::Zero(6, ndof);
    P.block(0, 0, 3, ndof) = trans_vectors.transpose();
    P.block(3, 0, 3, ndof) = rot_vectors.transpose();
    
    // Construct projection operator: (I - P^T P)
    Mat I = Mat::Identity(ndof, ndof);
    Mat projection = I - P.transpose() * P;
    
    // Project the Hessian: H_proj = P^T H P
    Mat projected_hessian = projection.transpose() * mass_weighted_hessian * projection;
    
    return projected_hessian;
}

// VibrationalModes member functions

std::string VibrationalModes::summary_string() const {
    std::string result;
    
    result += fmt::format("=== Vibrational Analysis Summary ===\n");
    result += fmt::format("Total modes: {}\n", frequencies_cm.size());
    result += fmt::format("\n=== All Frequencies (cm⁻¹) ===\n");
    
    for (int i = 0; i < frequencies_cm.size(); i++) {
        result += fmt::format("  Mode {:2d}: {:12.2f}\n", i + 1, frequencies_cm[i]);
    }
    
    return result;
}

std::string VibrationalModes::frequencies_string() const {
    std::string result;
    
    result += fmt::format("=== Vibrational Frequency Analysis ===\n");
    result += fmt::format("\n{:>6s} {:>14s} {:>14s}\n", 
                         "Mode", "Freq (cm⁻¹)", "Freq (meV)");
    result += fmt::format("{:-^40s}\n", "");
    
    for (int i = 0; i < frequencies_cm.size(); i++) {
        double freq_mev = frequencies_cm[i] * 0.1239841974;  // cm⁻¹ to meV
        result += fmt::format("{:6d} {:14.2f} {:14.2f}\n", 
                             i + 1, frequencies_cm[i], freq_mev);
    }
    
    return result;
}

std::string VibrationalModes::normal_modes_string(double threshold) const {
    std::string result;
    
    result += fmt::format("=== Normal Mode Displacements ===\n");
    result += fmt::format("(Components with magnitude > {:.2f})\n\n", threshold);
    
    for (int mode_idx = 0; mode_idx < frequencies_cm.size(); mode_idx++) {
        double freq = frequencies_cm[mode_idx];
        
        result += fmt::format("Mode {} - Frequency: {:.2f} cm⁻¹\n", mode_idx + 1, freq);
        
        // Extract and format significant displacements
        Vec mode_vec = normal_modes.col(mode_idx);
        bool has_significant = false;
        
        for (int atom = 0; atom < mode_vec.size() / 3; atom++) {
            double dx = mode_vec[3 * atom + 0];
            double dy = mode_vec[3 * atom + 1];
            double dz = mode_vec[3 * atom + 2];
            double magnitude = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (magnitude > threshold) {
                if (!has_significant) {
                    result += fmt::format("  {:>6s} {:>10s} {:>10s} {:>10s}\n", 
                                        "Atom", "dx", "dy", "dz");
                    has_significant = true;
                }
                result += fmt::format("  {:6d} {:10.4f} {:10.4f} {:10.4f}\n",
                                    atom + 1, dx, dy, dz);
            }
        }
        
        if (!has_significant) {
            result += "  (No displacements above threshold)\n";
        }
        result += "\n";
    }
    
    return result;
}

void VibrationalModes::log_summary() const {
    occ::log::info("Vibrational Analysis Summary:");
    occ::log::info("  Total modes: {}", frequencies_cm.size());
    
    auto all_freqs = get_all_frequencies();
    if (all_freqs.size() > 0) {
        occ::log::info("  Lowest freq: {:.2f} cm⁻¹", all_freqs[0]);
        occ::log::info("  Highest freq: {:.2f} cm⁻¹", all_freqs[all_freqs.size()-1]);
    }
}

Vec VibrationalModes::get_all_frequencies() const {
    Vec result = frequencies_cm;
    // Sort the frequencies
    std::sort(result.data(), result.data() + result.size());
    return result;
}

// Convenience functions for molecular systems
Mat construct_translation_vectors(const Molecule &molecule) {
    // Extract masses from molecule
    const auto &atoms = molecule.atoms();
    Vec masses(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        masses[i] = Element(atoms[i].atomic_number).mass();
    }
    
    return construct_translation_vectors(masses);
}

Mat construct_rotation_vectors(const Molecule &molecule) {
    // Extract masses and positions from molecule
    const auto &atoms = molecule.atoms();
    Vec masses(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        masses[i] = Element(atoms[i].atomic_number).mass();
    }
    Mat3N positions = molecule.positions();
    
    return construct_rotation_vectors(masses, positions);
}

Mat project_tr_rot_modes(const Mat &mass_weighted_hessian, const Molecule &molecule) {
    // Extract masses and positions from molecule
    const auto &atoms = molecule.atoms();
    Vec masses(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        masses[i] = Element(atoms[i].atomic_number).mass();
    }
    Mat3N positions = molecule.positions();
    
    return project_tr_rot_modes(mass_weighted_hessian, masses, positions);
}

} // namespace occ::core