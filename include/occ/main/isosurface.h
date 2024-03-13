#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/interpolator.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/slater/slaterbasis.h>
#include <occ/gto/density.h>
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

inline float smoothstep(float x, float l, float u) {
    float x2 = x * x;
    if (x < 0.0)
        return l;
    if (x > 1.0)
        return u;
    return (u - l) * (x2 - 2.0f * x2 * x) + l;
}

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
        float v = (m_isovalue - tot_i / (tot_i + tot_e + m_background_density));

        return m_diagonal_scale_factor * v;
    }

    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double tot_i{0.0}, tot_e{0.0};
        Eigen::Vector3f tot_i_g(0.0, 0.0, 0.0), tot_e_g(0.0, 0.0, 0.0);
        Eigen::Vector3f pos{x * m_cube_side_length + m_origin(0),
                            y * m_cube_side_length + m_origin(1),
                            z * m_cube_side_length + m_origin(2)};

        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;
        occ::timing::start(occ::timing::category::isosurface_function);

        float min_r = std::numeric_limits<float>::max();
        Eigen::Vector3f min_v;

        for (const auto &[interp, interp_positions, threshold, interior] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                Eigen::Vector3f v = interp_positions.col(i) - pos;
                float r = v.squaredNorm();
                if (r > threshold)
                    continue;
                else if (r < min_r) {
                    min_r = r;
                    min_v = v;
                }
                float rho = interp(r);
                float grad_rho = interp.gradient(r);
                if (i < interior) {
                    tot_i += rho;
                    tot_i_g.array() += 2 * v.array() * grad_rho;
                } else {
                    tot_e += rho;
                    tot_e_g.array() += 2 * v.array() * grad_rho;
                }
            }
        }

        double tot = tot_i + tot_e + m_background_density;
        Eigen::Vector3f result =
            ((tot_i_g.array() * tot_e - tot_e_g.array() * tot_i) / (tot * tot));
        occ::timing::stop(occ::timing::category::isosurface_function);
        if (result.squaredNorm() < 1e-6)
            return -min_v.normalized();
        return result.normalized();
    }

    inline float isovalue() const { return m_isovalue; }
    inline void set_isovalue(float iso) { m_isovalue = iso; }
    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int subdivisions() const { return m_subdivisions; }
    inline int num_calls() const { return m_num_calls; }

    inline void set_background_density(float rho) {
        m_background_density = rho;
    }
    inline float background_density() const { return m_background_density; }

  private:
    float m_diagonal_scale_factor{0.5f};
    float m_buffer{8.0};
    float m_cube_side_length{0.0};
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin;
    float m_isovalue{0.5};
    float m_background_density{0.0};
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
            for (int i = 0; i < interp_positions.cols(); i++) {
                float r = (interp_positions.col(i) - pos).squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                result += rho;
            }
        }

        occ::timing::stop(occ::timing::category::isosurface_function);
        float v = 0.5 - (result / (m_isovalue + result));
        return m_diagonal_scale_factor * v;
    }

    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double result{0.0};
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        Eigen::Vector3f pos{x * m_cube_side_length + m_origin(0),
                            y * m_cube_side_length + m_origin(1),
                            z * m_cube_side_length + m_origin(2)};

        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;
        occ::timing::start(occ::timing::category::isosurface_function);

        for (const auto &[interp, interp_positions, threshold, interior] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                Eigen::Vector3f v = interp_positions.col(i) - pos;
                float r = v.squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                float grad_rho = interp.gradient(r);
                result += rho;
                grad.array() += 2 * v.array() * grad_rho;
            }
        }

        occ::timing::stop(occ::timing::category::isosurface_function);
        return grad.normalized();
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

    float m_buffer{8.0};
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


class ElectronDensityFunctor {
  public:
    ElectronDensityFunctor(const occ::qm::Wavefunction &wfn, float sep);

    float operator()(float x, float y, float z) const {
        float result{0.0};
        Mat3N pos(3, 1);
	pos(0, 0) = x * m_cube_side_length + m_origin(0);
	pos(1, 0) = y * m_cube_side_length + m_origin(1);
	pos(2, 0) = z * m_cube_side_length + m_origin(2);

	Eigen::Vector3f posf = pos.col(0).cast<float>();
        if (!m_bounding_box.inside(posf))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;

        occ::timing::start(occ::timing::category::isosurface_function);

	auto rho = occ::density::evaluate_density_on_grid<0>(m_wfn, pos);
	result = rho(0);
        occ::timing::stop(occ::timing::category::isosurface_function);
        float v = 0.5 - (result / (m_isovalue + result));
        return m_diagonal_scale_factor * v;
    }

    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double result{0.0};
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        Mat3N pos(3, 1);
	pos(0, 0) = x * m_cube_side_length + m_origin(0);
	pos(1, 0) = y * m_cube_side_length + m_origin(1);
	pos(2, 0) = z * m_cube_side_length + m_origin(2);

	Eigen::Vector3f posf = pos.col(0).cast<float>();
        if (!m_bounding_box.inside(posf))
            return posf.normalized();

        m_num_calls++;
        occ::timing::start(occ::timing::category::isosurface_function);
	auto rho = occ::density::evaluate_density_on_grid<1>(m_wfn, pos);
	grad(0) = -rho(0, 1);
	grad(1) = -rho(0, 2);
	grad(2) = -rho(0, 3);
        occ::timing::stop(occ::timing::category::isosurface_function);
        return grad.normalized();
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

    float m_buffer{8.0};
    float m_cube_side_length{0.0};
    qm::Wavefunction m_wfn;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    int m_subdivisions{1};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
};


} // namespace occ::main
