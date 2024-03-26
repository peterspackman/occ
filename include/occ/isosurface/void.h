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

    bool inside_cell(const Eigen::Vector3f &p) const;

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	dest.resize(v.size());
	Eigen::Map<const Eigen::Matrix3Xf> vertices(v.data(), 3, v.size() / 3);
	Eigen::Map<Eigen::Matrix3Xf> dest_vertices(dest.data(), 3, v.size() / 3);

	dest_vertices = m_crystal.to_cartesian(vertices.cast<double>()).cast<float>();
    }

    inline float operator()(float x, float y, float z) const {
        float result{0.0};
        Eigen::Vector3f pos = occ::units::ANGSTROM_TO_BOHR * m_crystal.to_cartesian(Vec3(x, y, z)).cast<float>();

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

        return result - m_isovalue;
    }

    inline void fill_layer(float offset, Eigen::Ref<Eigen::MatrixXf> layer) const {

	Mat3N pos_frac(3, layer.size());
	for(int x = 0, idx = 0; x < layer.rows(); x++) {
	    for(int y = 0; y < layer.cols(); y++) {
		pos_frac(0, idx) = x;
		pos_frac(1, idx) = y;
		idx++;
	    }
	}
	pos_frac.row(2).setConstant(offset);
	fmt::print("pos_frac\n{}\n", pos_frac.leftCols(5));

	Eigen::Matrix3Xf positions = m_crystal.to_cartesian(pos_frac).cast<float>();
	fmt::print("pos \n{}\n", positions.leftCols(5));

	auto f = [&](const Eigen::Matrix3Xf &pos) {
	    Eigen::VectorXf rho(pos.cols());

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
		    rho(pt) = tot;
		}
	    };
	    occ::parallel::parallel_do(inner_func);
	    return rho;
	};


	auto values = f(positions);
	fmt::print("Rho:\n{}\n", values.topRows(5));

	for(int x = 0, idx = 0; x < layer.rows(); x++) {
	    for(int y = 0; y < layer.cols(); y++) {
		layer(x, y) = values(idx) - m_isovalue;
		idx++;
	    }
	}
    }


    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        Eigen::Vector3f pos = occ::units::ANGSTROM_TO_BOHR * m_crystal.to_cartesian(Vec3(x, y, z)).cast<float>();
	

	Vec3 pangs = pos.cast<double>() * occ::units::BOHR_TO_ANGSTROM;
	Vec3 v = m_crystal.to_fractional(pangs);

	/*
	const double eps = 1e-6;

	int ilower, iupper;
	double dlower = v.minCoeff(&ilower);
	double dupper = v.maxCoeff(&iupper);

	if(dlower < eps) {
	    Vec3 normal = Vec3::Zero();
	    normal(ilower) = -1.0;
	    normal = m_crystal.unit_cell().inverse() * normal;
	    return normal.normalized().cast<float>();
	}
	else if(dupper > (1-eps)) {
	    Vec3 normal = Vec3::Zero();
	    normal(iupper) = 1.0;
	    normal = m_crystal.unit_cell().inverse() * normal;
	    return normal.normalized().cast<float>();
	}
	*/
        m_num_calls++;

        double result{0.0};
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

        return -grad.normalized();
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
    occ::crystal::Crystal m_crystal;

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
