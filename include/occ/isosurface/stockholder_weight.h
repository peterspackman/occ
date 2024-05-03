#pragma once
#include <occ/isosurface/common.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/slater/hirshfeld.h>
#include <occ/core/parallel.h>

namespace occ::isosurface {

class StockholderWeightFunctor {
  public:
    StockholderWeightFunctor(const occ::core::Molecule &in,
                             occ::core::Molecule &ext, float sep,
                             const occ::slater::InterpolatorParams & = {});

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

    OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
        if (!m_bounding_box.inside(pos))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;
	return m_hirshfeld(pos);
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
		layer(pt) = m_hirshfeld(p);
	    }
	};
	occ::parallel::parallel_do(inner_func);
    }


    OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;
	return m_hirshfeld.gradient(pos);
    }

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }

    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

    inline void set_background_density(float rho) {
        m_hirshfeld.set_background_density(rho);
    }
    inline float background_density() const { return m_hirshfeld.background_density(); }

  private:
    float m_buffer{8.0};
    Eigen::Vector3f m_cube_side_length;
    Eigen::Vector3f m_origin;
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    occ::slater::StockholderWeight m_hirshfeld;
    AxisAlignedBoundingBox m_bounding_box;


};


}
