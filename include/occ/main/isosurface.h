#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/interpolator.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/slater/slaterbasis.h>
#include <vector>

namespace occ::main {

struct AxisAlignedBoundingBox {
    Eigen::Vector3f lower;
    Eigen::Vector3f upper;

    inline bool inside(const Eigen::Vector3f &point) const {
        return (lower.array() <= point.array()).all() &&
               (upper.array() >= point.array()).all();
    }
};

using LinearInterpolatorFloat =
    occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;

struct AtomInterpolator {
    LinearInterpolatorFloat interpolator;
    Eigen::Matrix<float, 3, Eigen::Dynamic> positions;
    float threshold{144.0};
    int interior{0};
};

struct InterpolatorParams {
    int num_points{8192};
    float domain_lower{0.04};
    float domain_upper{144.0};
};

class StockholderWeightFunctor {
  public:
    StockholderWeightFunctor(const occ::core::Molecule &in,
                             occ::core::Molecule &ext, float sep,
                             const InterpolatorParams & = {});

    OCC_ALWAYS_INLINE float operator()(float x, float y, float z) const {
        double tot_i{0.0}, tot_e{0.0};
        Eigen::Vector3f pos{x * m_cube_side_length + m_origin(0),
                            y * m_cube_side_length + m_origin(1),
                            z * m_cube_side_length + m_origin(2)};

        if (!m_bounding_box.inside(pos))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;
        occ::timing::start(occ::timing::category::isosurface_function);

        for (const auto &[interp, interp_positions, threshold, interior] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                float r = (interp_positions.col(i) - pos).squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                if (i < interior) {
                    tot_i += rho;
                } else {
                    tot_e += rho;
                }
            }
        }

        occ::timing::stop(occ::timing::category::isosurface_function);
        return m_diagonal_scale_factor * (m_isovalue - tot_i / (tot_i + tot_e));
    }

    inline float isovalue() const { return m_isovalue; }
    inline void set_isovalue(float iso) { m_isovalue = iso; }
    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int subdivisions() const { return m_subdivisions; }
    inline int num_calls() const { return m_num_calls; }

  private:
    float m_diagonal_scale_factor{0.5f};
    float m_buffer{6.0};
    float m_cube_side_length{0.0};
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin;
    float m_isovalue{0.5};
    mutable int m_num_calls{0};
    int m_subdivisions{1};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};
    size_t m_num_interior{0};

    AxisAlignedBoundingBox m_bounding_box;

    std::vector<AtomInterpolator> m_atom_interpolators;
    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};

class PromoleculeDensityFunctor {
  public:
    PromoleculeDensityFunctor(const occ::core::Molecule &mol, float sep,
                              const InterpolatorParams &params = {});

    float operator()(float x, float y, float z) const {
        float result{0.0};
        Eigen::Vector3f pos{x * m_cube_side_length + m_origin(0),
                            y * m_cube_side_length + m_origin(1),
                            z * m_cube_side_length + m_origin(2)};

        if (!m_bounding_box.inside(pos))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;

        occ::timing::start(occ::timing::category::isosurface_function);
        for (const auto &[interp, interp_positions, threshold, interior] :
             m_atom_interpolators) {
            Eigen::VectorXf r =
                (interp_positions.colwise() - pos).colwise().squaredNorm();
            for (int i = 0; i < r.rows(); i++) {
                if (r(i) > threshold)
                    continue;
                float rho = interp(r(i));
                result += rho;
            }
        }

        float normalized = result / m_isovalue;
        // this works as a kind of logistic function, makes the behaviour
        // near linear near the critical point
        occ::timing::stop(occ::timing::category::isosurface_function);
        return m_diagonal_scale_factor * (0.5 - result / (result + m_isovalue));
    }

    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int subdivisions() const { return m_subdivisions; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();
    float m_diagonal_scale_factor{0.5f};

    float m_buffer{2.0};
    float m_cube_side_length{0.0};
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    int m_subdivisions{1};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
    std::vector<AtomInterpolator> m_atom_interpolators;

    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};

} // namespace occ::main
