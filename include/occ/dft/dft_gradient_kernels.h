#pragma once
#include <occ/dft/dft_kernels.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

namespace occ::dft::kernels {

// Helper function to process a block of grid points for gradient calculation
// specialized/implemented in cpp file. `bf_to_atom` maps each column of
// `gto_vals.phi` / each index of `D` to its global atom (so the same kernel
// serves the dense path, where it is basis.bf_to_atom(), and the block-sparse
// screened path, where it is a packed local->global atom map).
template <int derivative_order, SpinorbitalKind spinorbital_kind>
void process_grid_block_gradient(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const std::vector<int> &bf_to_atom);

} // namespace occ::dft::kernels
