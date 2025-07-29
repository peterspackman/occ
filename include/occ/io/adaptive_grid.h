#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/geometry/volume_grid.h>

namespace occ::io {

struct AdaptiveBounds {
    Vec3 origin;
    Mat3 basis;  // Column vectors are the grid axes
    IVec3 steps;
    
    Vec3 min_corner() const { return origin; }
    Vec3 max_corner() const { return origin + basis * steps.cast<double>(); }
    
    double volume() const {
        return std::abs(basis.determinant()) * steps.prod();
    }
};

// Template class for computing adaptive grid bounds
template <typename PropertyFunctor>
class AdaptiveGridBounds {
public:
    struct Parameters {
        double value_threshold{1e-6};    // Stop when property drops below this
        double initial_radius{5.0};      // Initial search radius in Bohr
        double max_radius{30.0};         // Maximum search radius
        double search_tolerance{0.1};    // Tolerance for boundary search in Bohr
        double extra_buffer{2.0};        // Extra buffer after finding boundary
        double grid_spacing{0.2};        // Target grid spacing in Bohr
    };
    
    AdaptiveGridBounds(PropertyFunctor& func, const Parameters& params = {})
        : m_func(func), m_params(params) {}
    
    // Compute bounds for a molecule
    AdaptiveBounds compute(const core::Molecule& mol) {
        // Get molecule extent (convert from Angstroms to Bohr)
        Mat3N positions = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
        Vec3 min_pos = positions.rowwise().minCoeff();
        Vec3 max_pos = positions.rowwise().maxCoeff();
        Vec3 mol_center = 0.5 * (min_pos + max_pos);
        
        occ::log::debug("Molecule center: [{:.3f}, {:.3f}, {:.3f}] Bohr", 
                        mol_center(0), mol_center(1), mol_center(2));
        occ::log::debug("Molecule extent: min=[{:.3f}, {:.3f}, {:.3f}], max=[{:.3f}, {:.3f}, {:.3f}]",
                        min_pos(0), min_pos(1), min_pos(2),
                        max_pos(0), max_pos(1), max_pos(2));
        
        // Search along the 6 cardinal directions
        Vec3 bounds_min = mol_center;
        Vec3 bounds_max = mol_center;
        
        // +X, -X, +Y, -Y, +Z, -Z
        std::vector<Vec3> directions = {
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, -1, 0),
            Vec3(0, 0, 1), Vec3(0, 0, -1)
        };
        
        for (size_t i = 0; i < directions.size(); ++i) {
            const auto& dir = directions[i];
            double distance = find_boundary_distance(mol_center, dir);
            Vec3 boundary_point = mol_center + distance * dir;
            
            const char* dir_names[] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
            occ::log::debug("Direction {}: distance = {:.3f} Bohr", dir_names[i], distance);
            
            // Update bounds
            bounds_min = bounds_min.cwiseMin(boundary_point);
            bounds_max = bounds_max.cwiseMax(boundary_point);
        }
        
        // Add buffer
        bounds_min -= Vec3::Constant(m_params.extra_buffer);
        bounds_max += Vec3::Constant(m_params.extra_buffer);
        
        AdaptiveBounds bounds;
        bounds.origin = bounds_min;
        bounds.basis = Mat3::Identity();
        Vec3 box_size = bounds_max - bounds_min;
        
        // Set step sizes based on desired resolution
        bounds.steps = (box_size / m_params.grid_spacing).template cast<int>() + IVec3::Constant(1);
        bounds.basis.diagonal() = box_size.array() / bounds.steps.cast<double>().array();
        
        return bounds;
    }
    
    // Compute bounds with custom center point
    AdaptiveBounds compute(const Vec3& center, double initial_guess = 10.0) {
        // Search along the 6 cardinal directions
        Vec3 bounds_min = center;
        Vec3 bounds_max = center;
        
        // +X, -X, +Y, -Y, +Z, -Z
        std::vector<Vec3> directions = {
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, -1, 0),
            Vec3(0, 0, 1), Vec3(0, 0, -1)
        };
        
        for (const auto& dir : directions) {
            double distance = find_boundary_distance(center, dir, initial_guess);
            Vec3 boundary_point = center + distance * dir;
            
            // Update bounds
            bounds_min = bounds_min.cwiseMin(boundary_point);
            bounds_max = bounds_max.cwiseMax(boundary_point);
        }
        
        // Add buffer
        bounds_min -= Vec3::Constant(m_params.extra_buffer);
        bounds_max += Vec3::Constant(m_params.extra_buffer);
        
        AdaptiveBounds bounds;
        bounds.origin = bounds_min;
        bounds.basis = Mat3::Identity();
        Vec3 box_size = bounds_max - bounds_min;
        bounds.steps = (box_size / m_params.grid_spacing).template cast<int>() + IVec3::Constant(1);
        bounds.basis.diagonal() = box_size.array() / bounds.steps.cast<double>().array();
        
        return bounds;
    }
    
private:
    PropertyFunctor& m_func;
    Parameters m_params;
    
    double find_boundary_distance(const Vec3& start, const Vec3& direction, 
                                  double initial_guess = -1.0) {
        // Binary search for the boundary where property drops below threshold
        double r_min = 0.0;
        double r_max = (initial_guess > 0) ? initial_guess : m_params.initial_radius;
        
        // First, find a point where the property is below threshold
        Mat3N test_point(3, 1);
        Vec value = Vec::Zero(1);
        
        test_point.col(0) = start + r_max * direction;
        m_func(test_point, value);
        
        occ::log::trace("Initial probe at r={:.3f}: value={:.3e}, point=[{:.3f}, {:.3f}, {:.3f}]", 
                        r_max, value(0), test_point(0, 0), test_point(1, 0), test_point(2, 0));
        
        while (std::abs(value(0)) > m_params.value_threshold && r_max < m_params.max_radius) {
            r_max += m_params.grid_spacing;
            test_point.col(0) = start + r_max * direction;
            value.setZero();
            m_func(test_point, value);
            occ::log::trace("Probe at r={:.3f}: value={:.3e}, point=[{:.3f}, {:.3f}, {:.3f}]", 
                            r_max, value(0), test_point(0, 0), test_point(1, 0), test_point(2, 0));
        }
        
        if (r_max >= m_params.max_radius) {
            occ::log::debug("Hit max radius, returning {:.3f}", m_params.max_radius);
            return m_params.max_radius;
        }
        
        // If initial value was already below threshold, r_min should be 0
        // Binary search for exact boundary
        occ::log::trace("Starting binary search between {:.3f} and {:.3f}", r_min, r_max);
        while (r_max - r_min > m_params.search_tolerance) {
            double r_mid = 0.5 * (r_min + r_max);
            test_point.col(0) = start + r_mid * direction;
            value.setZero();
            m_func(test_point, value);
            
            if (std::abs(value(0)) > m_params.value_threshold) {
                r_min = r_mid;
            } else {
                r_max = r_mid;
            }
        }
        
        occ::log::trace("Binary search complete, returning {:.3f}", r_max);
        return r_max;
    }
};

// Convenience function for creating adaptive bounds calculator
template <typename PropertyFunctor>
auto make_adaptive_bounds(PropertyFunctor& func, 
                         const typename AdaptiveGridBounds<PropertyFunctor>::Parameters& params = {}) {
    return AdaptiveGridBounds<PropertyFunctor>(func, params);
}

} // namespace occ::io