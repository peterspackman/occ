#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/mults/coordinate_system.h>
#include <occ/dma/binomial.h>
#include <vector>
#include <cmath>

namespace occ::mults {

/**
 * S-functions for multipole interaction evaluation
 * 
 * This class computes the S-functions used in multipole electrostatic
 * interactions, following the formulation used in Orient.
 * 
 * The S-functions represent the geometric part of multipole interactions
 * for different combinations of multipole ranks and orientations.
 * 
 * Coordinate system:
 * - Site A coordinates: (rax, ray, raz) 
 * - Site B coordinates: (rbx, rby, rbz)
 * - Distance vector: rb - ra
 * 
 * S-function notation: S(t1, t2, j)
 * - t1: multipole index for site A
 * - t2: multipole index for site B  
 * - j: combined rank parameter
 */
class SFunctions {
public:
    // Maximum number of S-functions and derivative components
    static constexpr int MAX_S_FUNCTIONS = 400;
    static constexpr int NUM_FIRST_DERIVS = 15;   // 3 coords * 2 sites * 2 + 3 rotation params
    static constexpr int NUM_SECOND_DERIVS = 120; // Symmetric matrix storage

    // Orientation matrix derivative indices (row-major ordering)
    // Orientation matrix C = [cxx  cxy  cxz]
    //                        [cyx  cyy  cyz]
    //                        [czx  czy  czz]
    static constexpr int S1_CXX = 6;   // ∂/∂cxx - row 0, col 0
    static constexpr int S1_CXY = 7;   // ∂/∂cxy - row 0, col 1
    static constexpr int S1_CXZ = 8;   // ∂/∂cxz - row 0, col 2
    static constexpr int S1_CYX = 9;   // ∂/∂cyx - row 1, col 0
    static constexpr int S1_CYY = 10;  // ∂/∂cyy - row 1, col 1
    static constexpr int S1_CYZ = 11;  // ∂/∂cyz - row 1, col 2
    static constexpr int S1_CZX = 12;  // ∂/∂czx - row 2, col 0
    static constexpr int S1_CZY = 13;  // ∂/∂czy - row 2, col 1
    static constexpr int S1_CZZ = 14;  // ∂/∂czz - row 2, col 2
    
    struct SFunctionResult {
        double s0;                                    // Function value
        Vec s1 = Vec::Zero(NUM_FIRST_DERIVS);        // First derivatives 
        Vec s2 = Vec::Zero(NUM_SECOND_DERIVS);       // Second derivatives (packed)
        
        SFunctionResult() : s0(0.0) {}
    };
    
    /**
     * Constructor
     * @param max_rank Maximum multipole rank to handle
     */
    explicit SFunctions(int max_rank = 4);
    
    /**
     * Set the coordinate system
     * @param ra Position of site A
     * @param rb Position of site B
     */
    void set_coordinates(const Vec3& ra, const Vec3& rb);

    /**
     * Set the coordinate system directly
     * @param coords CoordinateSystem (may include body-frame transformations)
     */
    void set_coordinate_system(const CoordinateSystem& coords);

    /**
     * Get the current coordinate system
     */
    const CoordinateSystem& coordinate_system() const { return m_coords; }

    /**
     * @brief Create a new SFunctions object with swapped site coordinates
     *
     * Returns a new SFunctions where site A and site B are swapped.
     * This is used to compute S(t1@A, t2@B) as S(t2@B', t1@A') where
     * A' and B' are the swapped positions.
     *
     * @return SFunctions with ra and rb swapped
     */
    SFunctions swap_sites() const;

    /**
     * Compute S-function with automatic coordinate swapping
     *
     * This helper method automatically handles coordinate swapping when needed.
     * If swap=true, it creates a swapped SFunctions object and computes S(t2, t1)
     * instead of S(t1, t2), effectively reusing kernels for reverse orderings.
     *
     * @param t1 First multipole index
     * @param t2 Second multipole index
     * @param j Combined angular momentum
     * @param level Derivative level (0=S0, 1=S0+S1, 2=S0+S1+S2)
     * @param result Output result structure
     * @param swap If true, swap coordinates and call S(t2, t1)
     */
    void compute_with_swap(int t1, int t2, int j, int level, SFunctionResult& result, bool swap) const;

    /**
     * Compute S-function for given multipole indices
     * @param t1 Multipole index for site A
     * @param t2 Multipole index for site B
     * @param j Combined rank parameter
     * @param level Derivative level (0=value only, 1=first derivs, 2=second derivs)
     * @return S-function result with value and derivatives
     */
    SFunctionResult compute_s_function(int t1, int t2, int j, int level = 0) const;
    
    /**
     * Compute multiple S-functions in batch
     * @param indices Vector of (t1, t2, j) triplets
     * @param level Derivative level
     * @return Vector of S-function results
     */
    std::vector<SFunctionResult> compute_s_functions(
        const std::vector<std::tuple<int, int, int>>& indices, 
        int level = 0) const;
    
    /**
     * Convert linear multipole index to (l,m) quantum numbers
     * @param index Linear index
     * @return (l,m) pair
     */
    std::pair<int, int> index_to_lm(int index) const;
    
    // Accessors matching Orient's variable names exactly (delegate to coordinate system)
    inline double rax() const { return m_coords.rax(); }
    inline double ray() const { return m_coords.ray(); }
    inline double raz() const { return m_coords.raz(); }
    inline double rbx() const { return m_coords.rbx(); }
    inline double rby() const { return m_coords.rby(); }
    inline double rbz() const { return m_coords.rbz(); }
    inline double r() const { return m_coords.r; }
    
    // Raw coordinate accessors
    inline double raw_rax() const { return m_coords.raw_rax(); }
    inline double raw_ray() const { return m_coords.raw_ray(); }
    inline double raw_raz() const { return m_coords.raw_raz(); }
    inline double raw_rbx() const { return m_coords.raw_rbx(); }
    inline double raw_rby() const { return m_coords.raw_rby(); }
    inline double raw_rbz() const { return m_coords.raw_rbz(); }
    
    // Distance vector components
    inline double dx() const { return m_coords.dx(); }
    inline double dy() const { return m_coords.dy(); }
    inline double dz() const { return m_coords.dz(); }

    // Orientation matrix accessors (match Orient's naming)
    inline double cxx() const { return m_coords.cxx; }
    inline double cxy() const { return m_coords.cxy; }
    inline double cxz() const { return m_coords.cxz; }
    inline double cyx() const { return m_coords.cyx; }
    inline double cyy() const { return m_coords.cyy; }
    inline double cyz() const { return m_coords.cyz; }
    inline double czx() const { return m_coords.czx; }
    inline double czy() const { return m_coords.czy; }
    inline double czz() const { return m_coords.czz; }

private:
    int m_max_rank{0};
    CoordinateSystem m_coords;  // Coordinate system handling all transformations
    occ::dma::BinomialCoefficients m_binomial;  // For Orient-style fac() coefficients
    
    // Constants commonly used in S-functions (following Orient's sqrts.f90)
    static constexpr double rt2 = 1.4142135623730950488;
    static constexpr double rt3 = 1.7320508075688772935; 
    static constexpr double rt5 = 2.2360679774997896964;
    static constexpr double rt7 = 2.6457513110645905905;
    static constexpr double rt6 = rt2 * rt3;
    static constexpr double rt10 = rt2 * rt5;
    static constexpr double rt14 = rt2 * rt7;
    static constexpr double rt15 = rt3 * rt5;
    static constexpr double rt21 = rt3 * rt7;
    static constexpr double rt30 = rt2 * rt3 * rt5;
    static constexpr double rt35 = rt5 * rt7;
    static constexpr double rt42 = rt2 * rt3 * rt7;
    static constexpr double rt70 = rt2 * rt5 * rt7;
    static constexpr double rt105 = rt3 * rt5 * rt7;
    
    // Core S-function computation methods (following Orient structure)
    void compute_s012(int t1, int t2, int j, int level, SFunctionResult& result) const;
    void compute_s3(int t1, int t2, int j, int level, SFunctionResult& result) const;
    void compute_s4(int t1, int t2, int j, int level, SFunctionResult& result) const;
    void compute_s5(int t1, int t2, int j, int level, SFunctionResult& result) const;
    
    // Helper methods for specific multipole combinations
    void compute_charge_charge(int level, SFunctionResult& result) const;
    void compute_charge_dipole(int component, int level, SFunctionResult& result) const;
    void compute_charge_quadrupole(int component, int level, SFunctionResult& result) const;
    void compute_dipole_charge(int component, int level, SFunctionResult& result) const;
    void compute_quadrupole_charge(int component, int level, SFunctionResult& result) const;
    void compute_dipole_dipole(int comp1, int comp2, int level, SFunctionResult& result) const;
    void compute_dipole_quadrupole(int dip_comp, int quad_comp, int level, SFunctionResult& result) const;
    void compute_quadrupole_dipole(int quad_comp, int dip_comp, int level, SFunctionResult& result) const;
    void compute_quadrupole_quadrupole(int comp1, int comp2, int level, SFunctionResult& result) const;
};

} // namespace occ::mults
