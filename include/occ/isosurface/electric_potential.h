#pragma once
#include <occ/isosurface/common.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace occ::isosurface {

class ElectricPotentialFunctor {
  public:
    ElectricPotentialFunctor(const occ::qm::Wavefunction &wfn, float sep);

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

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
	    m_cube_side_length(2),
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
	    m_cube_side_length(2),
	    m_target_separation,
	    m_isovalue
	};
	impl::fill_normals(func, params, vertices, normals);
    }


    OCC_ALWAYS_INLINE Eigen::Vector3f normal(float x, float y, float z) const {
        double result{0.0};
        Eigen::Vector3f grad(0.0, 0.0, 0.0);
        auto posf = impl::remap_point(x, y, z, m_cube_side_length, m_origin);

        if (!m_bounding_box.inside(posf))
            return posf.normalized();

	Mat3N pos = posf.cast<double>();

        m_num_calls++;
	Mat3N efield = m_hf.electronic_electric_field_contribution(m_wfn.mo, pos);
	efield += m_hf.nuclear_electric_field_contribution(pos);

	grad(0) = -efield(0, 0);
	grad(1) = -efield(1, 0);
	grad(2) = -efield(2, 0);
        return grad.normalized();
    }

    inline float isovalue() const { return m_isovalue; }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }
    
    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();
    qm::HartreeFock m_hf;
    float m_buffer{5.0};
    Eigen::Vector3f m_cube_side_length;
    qm::Wavefunction m_wfn;
    Eigen::Vector3f m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
};

}
