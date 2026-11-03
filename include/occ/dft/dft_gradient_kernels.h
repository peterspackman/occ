#pragma once
#include <occ/dft/dft_kernels.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

namespace occ::dft::kernels {

// Adds the XC nuclear gradient of one batch of grid points to `gradient`
// (3 x natom). `bf_to_atom` maps each column of
// `gto_vals.phi` / each index of `D` to its global atom (so the same kernel
// serves the dense path, where it is basis.bf_to_atom(), and the block-sparse
// screened path, where it is a packed local->global atom map).
// `point_parent`, when given, is each point's parent atom: the grid response
// of points moving with their atom is added there. `energy_density`, when
// given, receives the unweighted XC energy density per point.
template <int derivative_order, SpinorbitalKind spinorbital_kind>
void process_grid_block_gradient(const BatchDensity &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const std::vector<int> &bf_to_atom,
                                 double *energy_density = nullptr,
                                 const int *point_parent = nullptr);

} // namespace occ::dft::kernels
