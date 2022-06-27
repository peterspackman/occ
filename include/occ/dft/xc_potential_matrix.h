#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dft/functional.h>
#include <occ/gto/gto.h>
#include <occ/qm/spinorbital.h>

namespace occ::dft {

using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;

template <occ::qm::SpinorbitalKind spinorbital_kind, int derivative_order>
void xc_potential_matrix(const DensityFunctional::Result &res, MatConstRef rho,
                         const occ::gto::GTOValues &gto_vals, MatRef KK,
                         double &energy) = delete;

template <>
void xc_potential_matrix<Restricted, 0>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef KK, double &energy);

template <>
void xc_potential_matrix<Restricted, 1>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef KK, double &energy);

template <>
void xc_potential_matrix<Restricted, 2>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef KK, double &energy);

template <>
void xc_potential_matrix<Unrestricted, 0>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef KK, double &energy);

template <>
void xc_potential_matrix<Unrestricted, 1>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef KK, double &energy);

template <>
void xc_potential_matrix<Unrestricted, 2>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef KK, double &energy);

} // namespace occ::dft
