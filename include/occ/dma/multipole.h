#pragma once
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::dma {

/**
 * @brief A simplified multipole moment class using Eigen vectors for storage
 * 
 * This class provides a simpler interface for working with multipole moments
 * compared to the templated std::array approach in occ::core::Multipole
 */
class Multipole {
public:
    /**
     * @brief Construct a multipole with specified maximum rank
     * 
     * @param rank Maximum rank of the multipole (0=charge, 1=dipole, 2=quadrupole, etc.)
     */
    Multipole(int rank = 0);
    
    /**
     * @brief Copy constructor
     */
    Multipole(const Multipole &other) = default;
    
    /**
     * @brief Assignment operator
     */
    Multipole& operator=(const Multipole &other) = default;
    
    /**
     * @brief Get the maximum rank of this multipole
     */
    int rank() const { return m_rank; }
    
    /**
     * @brief Get the charge (monopole)
     */
    double charge() const { return m_components(0); }
    
    /**
     * @brief Set the charge (monopole)
     */
    void set_charge(double q) { m_components(0) = q; }
    
    /**
     * @brief Get the dipole moment vector
     */
    Vec3 dipole() const;
    
    /**
     * @brief Set the dipole moment vector
     */
    void set_dipole(const Vec3 &d);
    
    /**
     * @brief Get the quadrupole tensor as a 3x3 matrix
     */
    Mat3 quadrupole() const;
    
    /**
     * @brief Set the quadrupole tensor from a 3x3 matrix
     */
    void set_quadrupole(const Mat3 &q);
    
    /**
     * @brief Get a specific multipole component by index
     * 
     * Component indices follow the standard ordering:
     * 0: q (charge)
     * 1-3: dx, dy, dz (dipole)
     * 4-9: Qxx, Qxy, Qxz, Qyy, Qyz, Qzz (quadrupole)
     * etc.
     * 
     * @param idx Index of the component
     * @return Component value
     */
    double get(int idx) const { return m_components(idx); }
    
    /**
     * @brief Set a specific multipole component by index
     * 
     * @param idx Index of the component
     * @param value Value to set
     */
    void set(int idx, double value) { m_components(idx) = value; }
    
    /**
     * @brief Get all components as an Eigen vector
     */
    const Vec& components() const { return m_components; }
    
    /**
     * @brief Get a mutable reference to all components
     */
    Vec& components() { return m_components; }
    
    /**
     * @brief Addition operator
     */
    Multipole operator+(const Multipole &other) const;
    
    /**
     * @brief In-place addition operator
     */
    Multipole& operator+=(const Multipole &other);
    
    /**
     * @brief Subtraction operator
     */
    Multipole operator-(const Multipole &other) const;
    
    /**
     * @brief Scalar multiplication
     */
    Multipole operator*(double scale) const;
    
    /**
     * @brief Get the number of components for a multipole of given rank
     */
    static int component_count(int rank);
    
    /**
     * @brief Convert to string representation
     */
    std::string to_string() const;
    
private:
    int m_rank;                // Maximum rank of multipole
    Vec m_components;          // Storage for all components
    
    // Component names for pretty printing
    static const std::vector<std::string> component_names;
};

// Allow scalar multiplication from the left
inline Multipole operator*(double scale, const Multipole &mp) {
    return mp * scale;
}

} // namespace occ::dma