#pragma once
#include <occ/isosurface/common.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/slater/promolecule.h>

namespace occ::isosurface {

class PromoleculeDensityFunctor {
  public:
    PromoleculeDensityFunctor(const occ::core::Molecule &mol, float sep,
                              const occ::slater::InterpolatorParams &params = {});

    inline void remap_vertices(const std::vector<float> &v, std::vector<float> &dest) const {
	impl::remap_vertices(*this, v, dest);
    }

    OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
        if (!m_bounding_box.inside(pos))
            return 1.0e8; // return an arbitrary large distance
        m_num_calls++;
	return m_promol(pos);
    }

    OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
        if (!m_bounding_box.inside(pos))
            return pos.normalized(); // zero normal
        m_num_calls++;

	return m_promol.gradient(pos);
    }

    inline const auto &side_length() const { return m_cube_side_length; }

    inline Eigen::Vector3i cubes_per_side() const { 
	return (side_length().array() / m_target_separation).ceil().cast<int>();
    }

    inline void set_isovalue(float iso) {
        m_isovalue = iso;
        update_region_for_isovalue();
    }

    inline const auto &origin() const { return m_origin; }
    inline int num_calls() const { return m_num_calls; }

  private:
    void update_region_for_isovalue();

    float m_buffer{8.0};
    FVec3 m_cube_side_length;
    FVec3 m_origin, m_minimum_atom_pos, m_maximum_atom_pos;
    float m_isovalue{0.002};
    mutable int m_num_calls{0};
    float m_target_separation{0.2 * occ::units::ANGSTROM_TO_BOHR};

    AxisAlignedBoundingBox m_bounding_box;
    occ::slater::PromoleculeDensity m_promol;

};

}
