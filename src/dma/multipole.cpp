#include <occ/dma/multipole.h>
#include <fmt/format.h>
#include <iomanip>
#include <sstream>

namespace occ::dma {

// Component names for pretty printing
const std::vector<std::string> Multipole::component_names = {
    "q",                                             // Monopole (1)
    "Dx", "Dy", "Dz",                                // Dipole (3)
    "Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "Qzz",        // Quadrupole (6)
    "Oxxx", "Oxxy", "Oxxz", "Oxyy", "Oxyz", "Oxzz",  // Octupole (10)
    "Oyyy", "Oyyz", "Oyzz", "Ozzz",
    "Hxxxx", "Hxxxy", "Hxxxz", "Hxxyy", "Hxxyz"      // Hexadecapole (15)
};

// Helper function to calculate the number of components for a specific rank
int Multipole::component_count(int rank) {
    switch (rank) {
        case 0: return 1;  // Monopole: q 
        case 1: return 4;  // + Dipole: dx, dy, dz
        case 2: return 10; // + Quadrupole: Qxx, Qxy, Qxz, Qyy, Qyz, Qzz
        case 3: return 20; // + Octupole: Oxxx, ... Ozzz
        case 4: return 35; // + Hexadecapole
        default:
            throw std::runtime_error(fmt::format("Unsupported multipole rank: {}", rank));
    }
}

Multipole::Multipole(int rank) : m_rank(rank) {
    if (rank < 0 || rank > 4) {
        throw std::runtime_error(fmt::format("Unsupported multipole rank: {}", rank));
    }
    
    // Allocate storage for all components up to the specified rank
    m_components = Vec::Zero(component_count(rank));
}

Vec3 Multipole::dipole() const {
    if (m_rank < 1) {
        throw std::runtime_error("Cannot get dipole from a multipole with rank < 1");
    }
    return Vec3(m_components(1), m_components(2), m_components(3));
}

void Multipole::set_dipole(const Vec3 &d) {
    if (m_rank < 1) {
        throw std::runtime_error("Cannot set dipole for a multipole with rank < 1");
    }
    m_components(1) = d(0);
    m_components(2) = d(1);
    m_components(3) = d(2);
}

Mat3 Multipole::quadrupole() const {
    if (m_rank < 2) {
        throw std::runtime_error("Cannot get quadrupole from a multipole with rank < 2");
    }
    
    Mat3 q = Mat3::Zero();
    // Fill the quadrupole tensor in the standard ordering
    q(0, 0) = m_components(4); // Qxx
    q(0, 1) = m_components(5); // Qxy
    q(0, 2) = m_components(6); // Qxz
    q(1, 0) = m_components(5); // Qxy (symmetric)
    q(1, 1) = m_components(7); // Qyy
    q(1, 2) = m_components(8); // Qyz
    q(2, 0) = m_components(6); // Qxz (symmetric)
    q(2, 1) = m_components(8); // Qyz (symmetric)
    q(2, 2) = m_components(9); // Qzz
    
    return q;
}

void Multipole::set_quadrupole(const Mat3 &q) {
    if (m_rank < 2) {
        throw std::runtime_error("Cannot set quadrupole for a multipole with rank < 2");
    }
    
    // Store the quadrupole tensor components
    m_components(4) = q(0, 0); // Qxx
    m_components(5) = q(0, 1); // Qxy
    m_components(6) = q(0, 2); // Qxz
    m_components(7) = q(1, 1); // Qyy
    m_components(8) = q(1, 2); // Qyz
    m_components(9) = q(2, 2); // Qzz
}

Multipole Multipole::operator+(const Multipole &other) const {
    int new_rank = std::max(m_rank, other.m_rank);
    Multipole result(new_rank);
    
    // Add components from both multipoles
    int my_components = m_components.size();
    int other_components = other.m_components.size();
    
    // Add my components
    for (int i = 0; i < my_components; i++) {
        result.m_components(i) += m_components(i);
    }
    
    // Add other's components
    for (int i = 0; i < other_components; i++) {
        result.m_components(i) += other.m_components(i);
    }
    
    return result;
}

Multipole& Multipole::operator+=(const Multipole &other) {
    // If other has higher rank, we need to resize
    if (other.m_rank > m_rank) {
        // Create a new multipole with the combined components
        Multipole result = *this + other;
        *this = result;
    } else {
        // Add other's components to mine
        for (int i = 0; i < other.m_components.size(); i++) {
            m_components(i) += other.m_components(i);
        }
    }
    
    return *this;
}

Multipole Multipole::operator-(const Multipole &other) const {
    int new_rank = std::max(m_rank, other.m_rank);
    Multipole result(new_rank);
    
    // Set all components to my values (or zero if I don't have them)
    for (int i = 0; i < result.m_components.size(); i++) {
        if (i < m_components.size()) {
            result.m_components(i) = m_components(i);
        }
    }
    
    // Subtract other's components
    for (int i = 0; i < other.m_components.size(); i++) {
        result.m_components(i) -= other.m_components(i);
    }
    
    return result;
}

Multipole Multipole::operator*(double scale) const {
    Multipole result(*this);
    result.m_components *= scale;
    return result;
}

std::string Multipole::to_string() const {
    std::ostringstream oss;
    
    // Format each component with name
    int idx = 0;
    oss << std::setw(5) << component_names[idx] << " " 
        << std::fixed << std::setprecision(6) << std::setw(12) << m_components(idx++) << "\n";
    
    if (m_rank >= 1) {
        oss << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "\n";
    }
    
    if (m_rank >= 2) {
        oss << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "\n"
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "\n";
    }
    
    // Add octupole and hexadecapole components if needed
    if (m_rank >= 3) {
        // Format octupole components (rank 3)
        for (int i = 0; i < 7; i++) {
            oss << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    ";
            if ((i + 1) % 3 == 0) {
                oss << "\n";
            }
        }
        // Handle the last elements of octupole
        oss << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    "
            << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "\n";
    }
    
    if (m_rank >= 4) {
        // Format hexadecapole components (rank 4)
        for (int i = 0; i < 15; i++) {
            oss << std::setw(5) << component_names[idx] << " " << std::setw(12) << m_components(idx++) << "    ";
            if ((i + 1) % 3 == 0) {
                oss << "\n";
            }
        }
    }
    
    return oss.str();
}

} // namespace occ::dma