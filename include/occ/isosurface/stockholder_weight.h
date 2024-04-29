#pragma once
#include <occ/isosurface/common.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>

namespace occ::isosurface {

class StockholderWeightFunctor {
  public:
    StockholderWeightFunctor(const occ::core::Molecule &in,
                             occ::core::Molecule &ext, float sep,
                             const InterpolatorParams & = {});

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

    OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
        double tot_i{0.0}, tot_e{m_background_density};

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
                if (i < interior) {
                    tot_i += rho;
                } else {
                    tot_e += rho;
                }
            }
        }

        return tot_i / (tot_i + tot_e);
    }

    void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {
        m_num_calls += layer.size();

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
		Eigen::Vector3f p = pos.col(pt);
		if (!m_bounding_box.inside(p)) {
		    layer(pt) = 0.0;
		    continue;
		}
		m_num_calls++;
		float tot_i = 0.0;
		float tot_e = m_background_density;

		for (const auto &[interp, interp_positions, threshold, interior] :
		     m_atom_interpolators) {
		    for (int i = 0; i < interp_positions.cols(); i++) {
			float r = (interp_positions.col(i) - p).squaredNorm();
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
		layer(pt) = tot_i / (tot_i + tot_e);
	    }
	};
	occ::parallel::parallel_do(inner_func);
    }


    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double tot_i{0.0}, tot_e{0.0};
        Eigen::Vector3f tot_i_g(0.0, 0.0, 0.0), tot_e_g(0.0, 0.0, 0.0);
        auto pos = Eigen::Vector3f(x, y, z);

        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;

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
        if (result.squaredNorm() < 1e-6)
            return -min_v.normalized();
        return result.normalized();
    }

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }

    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

    inline void set_background_density(float rho) {
        m_background_density = rho;
    }
    inline float background_density() const { return m_background_density; }

  private:
    float m_buffer{8.0};
    Eigen::Vector3f m_cube_side_length;
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin;
    float m_background_density{0};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};
    size_t m_num_interior{0};

    AxisAlignedBoundingBox m_bounding_box;

    std::vector<AtomInterpolator> m_atom_interpolators;
    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};


}
