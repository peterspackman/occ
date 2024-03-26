#pragma once
#include <occ/isosurface/common.h>
#include <occ/qm/wavefunction.h>

namespace occ::isosurface {

class ElectronDensityFunctor {
  public:
    ElectronDensityFunctor(const occ::qm::Wavefunction &wfn, float sep);

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

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
	    m_cube_side_length(2),
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
	    m_cube_side_length(2),
	    m_target_separation,
	    m_isovalue
	};
	impl::fill_normals(func, params, vertices, normals);
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

    int m_mo_index{-1};
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
