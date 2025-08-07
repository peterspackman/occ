#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/atom.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <cassert>

namespace occ::isosurface {

// Property types for volume data
enum class VolumePropertyKind {
    ElectronDensity,      // "electron_density", "density", "rho" 
    ElectronDensityAlpha, // "rho_alpha"
    ElectronDensityBeta,  // "rho_beta"
    ElectricPotential,    // "esp"
    EEQ_ESP,              // "eeqesp"
    PromoleculeDensity,   // "promolecule"
    DeformationDensity,   // "deformation_density"
    XCDensity,            // "xc"
    CrystalVoid           // "void" (alias for promolecule with crystal)
};

// Spin constraint for properties that support it
enum class SpinConstraint {
    Total,  // "both" (default)
    Alpha,  // "alpha"
    Beta    // "beta"
};

/**
 * @brief Clean volumetric data representation using Eigen::Tensor
 * 
 * This struct holds all the data needed to represent a 3D scalar field
 * on a regular grid, with proper linear algebra types.
 */
struct VolumeData {
    Vec3 origin;                                    // Grid origin in Bohr
    Mat3 basis;                                     // Grid basis vectors (columns)
    IVec3 steps;                                    // Number of steps in each direction
    std::vector<core::Atom> atoms;                  // Atoms for this calculation
    Eigen::Tensor<double, 3> data;                  // Volumetric data [nx, ny, nz]
    std::string name;                               // Volume name/description
    VolumePropertyKind property;                    // What property this represents
    
    // Convenience accessors
    int nx() const { return steps(0); }
    int ny() const { return steps(1); }
    int nz() const { return steps(2); }
    size_t total_points() const { return static_cast<size_t>(nx()) * ny() * nz(); }
    
    // Grid spacing (diagonal elements of basis matrix)
    Vec3 spacing() const { return basis.diagonal(); }
    
    // Physical dimensions of the grid
    Vec3 dimensions() const { 
        return Vec3(nx() * spacing()(0), ny() * spacing()(1), nz() * spacing()(2)); 
    }
    
    // Convert grid indices to physical coordinates
    Vec3 grid_to_coords(int i, int j, int k) const {
        return origin + i * basis.col(0) + j * basis.col(1) + k * basis.col(2);
    }
    
    // Access data with bounds checking in debug mode
    double& operator()(int i, int j, int k) {
        assert(i >= 0 && i < nx() && j >= 0 && j < ny() && k >= 0 && k < nz());
        return data(i, j, k);
    }
    
    const double& operator()(int i, int j, int k) const {
        assert(i >= 0 && i < nx() && j >= 0 && j < ny() && k >= 0 && k < nz());
        return data(i, j, k);
    }
};

} // namespace occ::isosurface