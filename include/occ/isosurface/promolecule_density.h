#pragma once
#include <occ/isosurface/common.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>

namespace occ::isosurface {

class PromoleculeDensityFunctor {
  public:
    PromoleculeDensityFunctor(const occ::core::Molecule &mol, float sep,
                              const InterpolatorParams &params = {});

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

    float operator()(float x, float y, float z) const {
        float result{0.0};
        auto pos = impl::remap_point(x, y, z, m_cube_side_length, m_origin);

        if (!m_bounding_box.inside(pos))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;

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

        return m_isovalue - result;
    }

    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double result{0.0};
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        auto pos = impl::remap_point(x, y, z, m_cube_side_length, m_origin);

        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;

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

        return grad.normalized();
    }

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }

    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();

    float m_buffer{8.0};
    Eigen::Vector3f m_cube_side_length;
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
    std::vector<AtomInterpolator> m_atom_interpolators;

    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};

}
