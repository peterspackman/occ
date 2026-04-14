#pragma once
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/cartesian_multipole.h>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>

namespace occ::mults {

/// High-level API for Cartesian T-tensor multipole interaction calculations.
///
/// This engine computes electrostatic interaction energies between
/// multipole distributions using the Cartesian interaction tensor approach:
///
///   E = sum_{tuv,t'u'v'} m^A_{tuv} * T_{t+t',u+u',v+v'} * m^B_{t'u'v'}
///
/// where m^A absorbs the (-1)^l sign and 1/(t!u!v!) factorial,
/// and m^B absorbs only the 1/(t'!u'!v'!) factorial.
///
/// Intended to run side-by-side with the S-function engine for
/// cross-validation and benchmarking.
class CartesianInteractions {
public:
    struct Config {
        int max_rank = 4;
    };

    CartesianInteractions() = default;
    explicit CartesianInteractions(const Config &config);

    /// Compute electrostatic interaction energy between two multipoles.
    ///
    /// @param mult1  First multipole distribution (spherical, Stone convention)
    /// @param pos1   Position of first multipole
    /// @param mult2  Second multipole distribution
    /// @param pos2   Position of second multipole
    /// @return Interaction energy in atomic units
    double compute_interaction_energy(
        const occ::dma::Mult &mult1, const Vec3 &pos1,
        const occ::dma::Mult &mult2, const Vec3 &pos2) const;

    /// Body-frame variant: multipoles defined in body frame, rotated to lab.
    ///
    /// @param rot1  Rotation matrix for molecule 1 (body to lab)
    /// @param rot2  Rotation matrix for molecule 2 (body to lab)
    double compute_interaction_energy(
        const occ::dma::Mult &mult1, const Vec3 &pos1, const Mat3 &rot1,
        const occ::dma::Mult &mult2, const Vec3 &pos2, const Mat3 &rot2) const;

    const Config &config() const { return m_config; }

private:
    Config m_config;
};

} // namespace occ::mults
