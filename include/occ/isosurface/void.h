#pragma once
#include <occ/isosurface/common.h>
#include <occ/crystal/crystal.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/parallel.h>

namespace occ::isosurface {

class VoidSurfaceFunctor {
  public:
    VoidSurfaceFunctor(const crystal::Crystal &crystal, float sep,
                       const InterpolatorParams &params = {});

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

    inline float operator()(const FVec3 &pos) const {
        float result{0.0};

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

        return -result;
    }

    inline void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {

	Mat3N pos_frac = m_crystal.to_fractional(pos.cast<double>().array() * occ::units::BOHR_TO_ANGSTROM);
	int num_threads = occ::parallel::get_num_threads();
	auto inner_func = [&](int thread_id) {
	    int total_elements = pos.cols();
	    int block_size = total_elements / num_threads;
	    int start_index = thread_id * block_size;
	    int end_index = start_index + block_size;
	    if(thread_id == num_threads - 1) {
		end_index = total_elements;
	    }
	    for(int pt = start_index; pt < end_index; pt++) {
		if((pos_frac.col(pt).array() > 1.0).any() || (pos_frac.col(pt).array() < 0.0).any()) {
		    layer(pt) = -10;
		    continue;
		}
		Eigen::Vector3f p = pos.col(pt);
		m_num_calls++;
		float tot = 0.0;

		for (const auto &[interp, interp_positions, threshold, interior] :
		     m_atom_interpolators) {
		    for (int i = 0; i < interp_positions.cols(); i++) {
			float r = (interp_positions.col(i) - p).squaredNorm();
			if (r > threshold)
			    continue;
			float rho = interp(r);
			tot += rho;
		    }
		}
		layer(pt) = -tot;
	    }
	};
	occ::parallel::parallel_do(inner_func);
    }


    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        Eigen::Vector3f pos = occ::units::ANGSTROM_TO_BOHR * m_crystal.to_cartesian(Vec3(x, y, z)).cast<float>();
	
        m_num_calls++;

        for (const auto &[interp, interp_positions, threshold, interior] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                Eigen::Vector3f v = interp_positions.col(i) - pos;
                float r = v.squaredNorm();
                if (r > threshold)
                    continue;
                float grad_rho = interp.gradient(r);
                grad.array() += 2 * v.array() * grad_rho;
            }
        }

        return -grad.normalized();
    }

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }


    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

    inline const auto &molecule() const { return m_molecule; }

  private:
    void update_region();
    occ::crystal::Crystal m_crystal;

    float m_buffer{8.0};
    Eigen::Vector3f m_cube_side_length;
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
    std::vector<AtomInterpolator> m_atom_interpolators;
    occ::core::Molecule m_molecule;

    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};

}
