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

    void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const {
        m_num_calls += layer.size();
	Mat3N dpos = pos.cast<double>();
	Vec esp = m_hf.electronic_electric_potential_contribution(m_wfn.mo, dpos);
	esp += m_hf.nuclear_electric_potential_contribution(dpos);
	esp.array() = esp.array().abs();
	layer = esp.cast<float>();
    }


    OCC_ALWAYS_INLINE FVec3 normal(const FVec3 &posf) const {
        FVec3 grad(0.0, 0.0, 0.0);

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

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }
    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region();
    qm::HartreeFock m_hf;
    float m_buffer{5.0};
    FVec3 m_cube_side_length;
    qm::Wavefunction m_wfn;
    FVec3 m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
};

}
