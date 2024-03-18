#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/interpolator.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/slater/slaterbasis.h>
#include <occ/gto/density.h>
#include <occ/core/parallel.h>
#include <occ/qm/hf.h>
#include <vector>

namespace occ::main {

namespace impl {
struct FillParams {
    Eigen::Vector3f origin;
    float side_length{0.0};
    float separation{0.0};
    float isovalue{0.0};
};

template<class Func>
void fill_layer(Func &f, const FillParams &params, 
		float offset, Eigen::Ref<Eigen::MatrixXf> layer) {
    Mat3N pos(3, layer.size());
    for(int x = 0, idx = 0; x < layer.rows(); x++) {
	for(int y = 0; y < layer.cols(); y++) {
	    pos(0, idx) = x * params.separation + params.origin(0);
	    pos(1, idx) = y * params.separation + params.origin(1);
	    idx++;
	}
    }
    pos.row(2).setConstant(offset * params.side_length + params.origin(2));

    auto values = f(pos);

    for(int x = 0, idx = 0; x < layer.rows(); x++) {
	for(int y = 0; y < layer.cols(); y++) {
	    layer(x, y) = params.isovalue - values(idx);
	    idx++;
	}
    }
}

template<class Func>
void fill_normals(Func &f, const FillParams &params,
		  const std::vector<float> &vertices,
		  std::vector<float> &normals) {
    auto cube_pos = Eigen::Map<const Eigen::Matrix3Xf>(vertices.data(), 3, vertices.size() / 3);

    Mat3N pos(cube_pos.rows(), cube_pos.cols());
    for(int i = 0; i < cube_pos.cols(); i++) {
	pos(0, i) = cube_pos(0, i) * params.side_length + params.origin(0);
	pos(1, i) = cube_pos(1, i) * params.side_length + params.origin(1);
	pos(2, i) = cube_pos(2, i) * params.side_length + params.origin(2);
    }

    Mat3N grad = f(pos);
    for(int i = 0; i < grad.cols(); i++) {
	Vec3 normal = -grad.col(i).normalized();
	normals.push_back(normal(0));
	normals.push_back(normal(1));
	normals.push_back(normal(2));
    }
}

}

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

        return m_isovalue - tot_i / (tot_i + tot_e + m_background_density);
    }

    void fill_layer(float offset, Eigen::Ref<Eigen::MatrixXf> layer) const {

        m_num_calls += layer.size();
	auto func = [&](const Mat3N &pos) {
	    Eigen::VectorXf inside(pos.cols());
	    Eigen::VectorXf outside(pos.cols());

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
		    Eigen::Vector3f p = pos.col(pt).cast<float>();
		    if (!m_bounding_box.inside(p)) {
			inside(pt) = 0.0;
			outside(pt) = 1e8;
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
		    inside(pt) = tot_i;
		    outside(pt) = tot_e;
		}
	    };
	    occ::parallel::parallel_do(inner_func);
	    return inside.array() / (inside.array() + outside.array());
	};
	impl::FillParams params{
	    m_origin,
	    m_cube_side_length,
	    m_target_separation,
	    m_isovalue
	};

	impl::fill_layer(func, params, offset, layer);
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

    inline int cubes_per_side() const { 
	return std::ceil(side_length() / m_target_separation); 
    }

    inline float isovalue() const { return m_isovalue; }
    inline void set_isovalue(float iso) { m_isovalue = iso; }
    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

    inline void set_background_density(float rho) {
        m_background_density = rho;
    }
    inline float background_density() const { return m_background_density; }

  private:
    float m_buffer{8.0};
    float m_cube_side_length{0.0};
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin;
    float m_isovalue{0.5};
    float m_background_density{0};
    mutable int m_num_calls{0};
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
        Eigen::Vector3f pos{x * m_cube_side_length + m_origin(0),
                            y * m_cube_side_length + m_origin(1),
                            z * m_cube_side_length + m_origin(2)};

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

    inline int cubes_per_side() const { 
	return std::ceil(side_length() / m_target_separation); 
    }
    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();

    float m_buffer{8.0};
    float m_cube_side_length{0.0};
    InterpolatorParams m_interpolator_params;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
    std::vector<AtomInterpolator> m_atom_interpolators;

    ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};


class ElectronDensityFunctor {
  public:
    ElectronDensityFunctor(const occ::qm::Wavefunction &wfn, float sep);

    void fill_layer(float offset, Eigen::Ref<Eigen::MatrixXf> layer) const {

        m_num_calls += layer.size();
	auto func = [&](const Mat3N &pos) {
	    if(m_mo_index >= 0) {
		return m_wfn.electron_density_mo(pos, m_mo_index);
	    }
	    else {
		return m_wfn.electron_density(pos);
	    }
	};
	impl::FillParams params{
	    m_origin,
	    m_cube_side_length,
	    m_target_separation,
	    m_isovalue
	};

	impl::fill_layer(func, params, offset, layer);
    }

    void fill_normals(const std::vector<float> &vertices, std::vector<float> &normals) const {
	m_num_calls += vertices.size() / 3;
	auto func = [&](const Mat3N &pos) {
	    if(m_mo_index >= 0) {
		return m_wfn.electron_density_mo_gradient(pos, m_mo_index);
	    }
	    else {
		return m_wfn.electron_density_gradient(pos);
	    }
	};

	impl::FillParams params{
	    m_origin,
	    m_cube_side_length,
	    m_target_separation,
	    m_isovalue
	};
	impl::fill_normals(func, params, vertices, normals);
    }

    inline int cubes_per_side() const { 
	return std::ceil(side_length() / m_target_separation); 
    }

    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();

    int m_mo_index{-1};
    float m_buffer{5.0};
    float m_cube_side_length{0.0};
    qm::Wavefunction m_wfn;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
};

class ElectricPotentialFunctor {
  public:
    ElectricPotentialFunctor(const occ::qm::Wavefunction &wfn, float sep);

    void fill_layer(float offset, Eigen::Ref<Eigen::MatrixXf> layer) const {
        m_num_calls += layer.size();
	auto func = [&](const Mat3N &pos) {
	    Vec esp = m_hf.electronic_electric_potential_contribution(m_wfn.mo, pos);
	    esp += m_hf.nuclear_electric_potential_contribution(pos);
	    esp.array() = esp.array().abs();
	    return esp;
	};
	impl::FillParams params{
	    m_origin,
	    m_cube_side_length,
	    m_target_separation,
	    m_isovalue
	};

	impl::fill_layer(func, params, offset, layer);
    }

    void fill_normals(const std::vector<float> &vertices, std::vector<float> &normals) const {
	m_num_calls += vertices.size() / 3;
	auto func = [&](const Mat3N &pos) {
	    return -(m_hf.electronic_electric_field_contribution(m_wfn.mo, pos) +
		   m_hf.nuclear_electric_field_contribution(pos));
	};

	impl::FillParams params{
	    m_origin,
	    m_cube_side_length,
	    m_target_separation,
	    m_isovalue
	};
	impl::fill_normals(func, params, vertices, normals);
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
	Mat3N efield = m_hf.electronic_electric_field_contribution(m_wfn.mo, pos);
	efield += m_hf.nuclear_electric_field_contribution(pos);

	grad(0) = -efield(0, 0);
	grad(1) = -efield(1, 0);
	grad(2) = -efield(2, 0);
        return grad.normalized();
    }

    inline int cubes_per_side() const { 
	return std::ceil(side_length() / m_target_separation); 
    }

    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline float side_length() const { return m_cube_side_length; }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();
    qm::HartreeFock m_hf;
    float m_buffer{5.0};
    float m_cube_side_length{0.0};
    qm::Wavefunction m_wfn;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
};


} // namespace occ::main
