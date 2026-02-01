#include <occ/core/molecular_axis.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <fstream>
#include <unordered_map>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>


namespace occ::core {

MolecularAxisCalculator::MolecularAxisCalculator(const occ::qm::Wavefunction& wfn) 
    : m_wfn(wfn) {}

Vec3 MolecularAxisCalculator::center_of_mass() const {
    Vec3 com = Vec3::Zero();
    double total_mass = 0.0;
    
    auto positions = m_wfn.positions();
    auto atomic_numbers = m_wfn.atomic_numbers();
    
    for (int i = 0; i < positions.cols(); i++) {
        double mass = Element(atomic_numbers[i]).mass();
        com += positions.col(i) * mass;
        total_mass += mass;
    }
    
    return com / total_mass;
}

MolecularAxisResult MolecularAxisCalculator::calculate_neighcrys_axes(const std::vector<int>& axis_atoms) const {
    if (axis_atoms.size() != 3) {
        throw std::runtime_error("Neighcrys axis method requires exactly 3 atom indices");
    }
    
    auto positions = m_wfn.positions();
    int atom_a = axis_atoms[0];
    int atom_b = axis_atoms[1]; 
    int atom_c = axis_atoms[2];
    
    if (atom_a >= positions.cols() || atom_b >= positions.cols() || atom_c >= positions.cols() ||
        atom_a < 0 || atom_b < 0 || atom_c < 0) {
        throw std::runtime_error("Invalid atom indices for neighcrys axes");
    }
    
    // X-axis: atom_b - atom_a (normalized)
    Vec3 x0 = positions.col(atom_b) - positions.col(atom_a);
    x0.normalize();
    
    // Y-axis: (atom_a - atom_c) projected perpendicular to X-axis  
    Vec3 x1 = positions.col(atom_a) - positions.col(atom_c);
    x1.normalize();
    x1 = x1 - x1.dot(x0) * x0;  // Remove component parallel to x0
    x1.normalize();
    
    // Z-axis: cross product to ensure right-handed system
    Vec3 x2 = x0.cross(x1);
    
    Mat3 axes;
    axes.row(0) = x0;
    axes.row(1) = x1; 
    axes.row(2) = x2;
    
    // Verify right-handed coordinate system
    double det = axes.determinant();
    if (det <= 0) {
        throw std::runtime_error("Coordinate system must be right-handed");
    }
    
    MolecularAxisResult result;
    result.axes = axes;
    result.center_of_mass = center_of_mass();
    result.axis_atoms = axis_atoms;
    result.method = AxisMethod::Neighcrys;
    result.determinant = det;
    
    return result;
}

MolecularAxisResult MolecularAxisCalculator::calculate_pca_axes() const {
    auto positions = m_wfn.positions();
    Vec3 com = center_of_mass();
    
    // Center positions at center of mass
    Mat3N centered_positions = positions.colwise() - com;
    
    // Compute SVD of transpose
    Eigen::JacobiSVD<Mat3N> svd(centered_positions, Eigen::ComputeFullU);
    Mat3 axes = svd.matrixU();
    
    // Ensure right-handed coordinate system
    if (axes.determinant() < 0) {
        axes.col(2) = -axes.col(2);
    }
    
    MolecularAxisResult result;
    result.axes = axes.transpose();  // Return as row vectors
    result.center_of_mass = com;
    result.axis_atoms = {};  // Not applicable for PCA
    result.method = AxisMethod::PCA;
    result.determinant = result.axes.determinant();
    
    return result;
}

MolecularAxisResult MolecularAxisCalculator::calculate_moi_axes() const {
    auto positions = m_wfn.positions();
    auto atomic_numbers = m_wfn.atomic_numbers();
    Vec3 com = center_of_mass();
    
    // Center positions at center of mass
    Mat3N centered_positions = positions.colwise() - com;
    
    // Compute moment of inertia tensor
    Mat3 moi = Mat3::Zero();
    for (int i = 0; i < positions.cols(); i++) {
        double mass = Element(atomic_numbers[i]).mass();
        Vec3 r = centered_positions.col(i);
        
        // Diagonal terms
        moi(0, 0) += mass * (r[1] * r[1] + r[2] * r[2]);
        moi(1, 1) += mass * (r[0] * r[0] + r[2] * r[2]);
        moi(2, 2) += mass * (r[0] * r[0] + r[1] * r[1]);
        
        // Off-diagonal terms
        moi(0, 1) -= mass * r[0] * r[1];
        moi(0, 2) -= mass * r[0] * r[2];
        moi(1, 2) -= mass * r[1] * r[2];
    }
    
    // Make symmetric
    moi(1, 0) = moi(0, 1);
    moi(2, 0) = moi(0, 2);
    moi(2, 1) = moi(1, 2);
    
    // Compute eigenvectors
    Eigen::SelfAdjointEigenSolver<Mat3> eigenSolver(moi);
    Mat3 axes = eigenSolver.eigenvectors().transpose();
    
    // Ensure right-handed coordinate system
    if (axes.determinant() < 0) {
        axes.row(2) = -axes.row(2);
    }
    
    MolecularAxisResult result;
    result.axes = axes;
    result.center_of_mass = com;
    result.axis_atoms = {};  // Not applicable for MOI
    result.method = AxisMethod::MOI;
    result.determinant = axes.determinant();
    
    return result;
}

MolecularAxisResult MolecularAxisCalculator::calculate_axes(AxisMethod method, 
                                                           const std::vector<int>& axis_atoms) const {
    switch (method) {
        case AxisMethod::Neighcrys:
            return calculate_neighcrys_axes(axis_atoms);
        case AxisMethod::PCA:
            return calculate_pca_axes();
        case AxisMethod::MOI:
            return calculate_moi_axes();
        case AxisMethod::None:
        default:
            throw std::runtime_error("Invalid or unsupported axis method");
    }
}

std::vector<std::string> MolecularAxisCalculator::generate_neighcrys_labels() const {
    std::vector<std::string> labels;
    auto atomic_numbers = m_wfn.atomic_numbers();
    
    // Count occurrences of each element
    std::unordered_map<int, int> element_counts;
    for (int i = 0; i < atomic_numbers.size(); i++) {
        element_counts[atomic_numbers[i]]++;
    }
    
    // Reset counts for labeling
    std::unordered_map<int, int> current_counts;
    
    for (int i = 0; i < atomic_numbers.size(); i++) {
        int atomic_num = atomic_numbers[i];
        current_counts[atomic_num]++;
        
        std::string symbol = Element(atomic_num).symbol();
        
        // Simple labeling scheme similar to neighcrys: ELEMENT_TYPE_NUMBER____
        std::string label = fmt::format("{}_F1_{:d}____", symbol, current_counts[atomic_num]);
        // Pad to 10 characters
        if (label.length() < 10) {
            label += std::string(10 - label.length(), '_');
        }
        labels.push_back(label);
    }
    
    return labels;
}

int MolecularAxisCalculator::calculate_bond_separation(int atom_i, int atom_j) const {
    auto positions = m_wfn.positions();
    Vec3 pos_i = positions.col(atom_i);
    Vec3 pos_j = positions.col(atom_j);
    double distance = (pos_i - pos_j).norm() * units::BOHR_TO_ANGSTROM;
    
    // Simple heuristic: if atoms are within bonding distance, separation is 1
    // Otherwise, estimate based on typical bond lengths
    if (distance < 2.0) return 1;
    if (distance < 3.5) return 2;
    return 3; // Default for longer distances
}

NeighcrysAxisInfo MolecularAxisCalculator::generate_neighcrys_info(const std::vector<int>& axis_atoms) const {
    NeighcrysAxisInfo info;
    info.atom_labels = generate_neighcrys_labels();
    info.axis_atoms = axis_atoms;
    
    if (axis_atoms.size() >= 3) {
        int atom_a = axis_atoms[0];
        int atom_b = axis_atoms[1];
        int atom_c = axis_atoms[2];
        
        info.separations = {
            calculate_bond_separation(atom_a, atom_b),
            calculate_bond_separation(atom_a, atom_c)
        };
    } else if (m_wfn.positions().cols() >= 3) {
        // Default to first 3 atoms
        info.axis_atoms = {0, 1, 2};
        info.separations = {
            calculate_bond_separation(0, 1),
            calculate_bond_separation(0, 2)
        };
    }
    
    return info;
}

void MolecularAxisCalculator::apply_molecular_transformation(occ::qm::Wavefunction& wfn, 
                                                           const MolecularAxisResult& result) {
    Vec3 translation = -result.center_of_mass;
    wfn.apply_transformation(result.axes, translation);
}

void MolecularAxisCalculator::write_neighcrys_axis_file(const std::string& filename, 
                                                      const NeighcrysAxisInfo& axis_info) {
    std::ofstream axis_file(filename);
    if (!axis_file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open axis file: {}", filename));
    }
    
    // Single molecule
    axis_file << "MOLX 1\n";
    
    if (axis_info.axis_atoms.size() >= 3 && axis_info.separations.size() >= 2) {
        int atom_a = axis_info.axis_atoms[0];
        int atom_b = axis_info.axis_atoms[1];
        int atom_c = axis_info.axis_atoms[2];
        
        // X-axis (LINE)
        axis_file << fmt::format("X LINE  {} {} {}\n", 
                                axis_info.atom_labels[atom_a], 
                                axis_info.atom_labels[atom_b], 
                                axis_info.separations[0]);
        
        // Y-axis (PLANE) 
        axis_file << fmt::format("Y PLANE {} {} {} {} {}\n",
                                axis_info.atom_labels[atom_a], 
                                axis_info.atom_labels[atom_b], 
                                axis_info.separations[0],
                                axis_info.atom_labels[atom_c], 
                                axis_info.separations[1]);
    }
    
    axis_file << "ENDS\n";
    axis_file.close();
}

void MolecularAxisCalculator::write_oriented_xyz(const std::string& filename, 
                                                const occ::qm::Wavefunction& wfn,
                                                const std::string& title) {
    std::ofstream xyz(filename);
    if (!xyz.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open XYZ file: {}", filename));
    }
    
    auto positions = wfn.positions() * units::BOHR_TO_ANGSTROM;
    auto atomic_numbers = wfn.atomic_numbers();
    
    xyz << positions.cols() << "\n";
    xyz << title << "\n";
    
    for (int i = 0; i < positions.cols(); i++) {
        Element element(atomic_numbers[i]);
        std::string symbol = element.symbol();
        Vec3 pos = positions.col(i);
        xyz << fmt::format("{:<4s} {:20.12f} {:20.12f} {:20.12f}\n", 
                          symbol, pos.x(), pos.y(), pos.z());
    }
    
    xyz.close();
}

std::string MolecularAxisCalculator::axis_method_to_string(AxisMethod method) {
    switch (method) {
        case AxisMethod::None: return "none";
        case AxisMethod::Neighcrys: return "nc";
        case AxisMethod::PCA: return "pca";
        case AxisMethod::MOI: return "moi";
        default: return "unknown";
    }
}

AxisMethod MolecularAxisCalculator::string_to_axis_method(const std::string& method_str) {
    if (method_str == "none") return AxisMethod::None;
    if (method_str == "nc") return AxisMethod::Neighcrys;
    if (method_str == "pca") return AxisMethod::PCA;
    if (method_str == "moi") return AxisMethod::MOI;
    throw std::runtime_error(fmt::format("Unknown axis method: {}", method_str));
}

} // namespace occ::core