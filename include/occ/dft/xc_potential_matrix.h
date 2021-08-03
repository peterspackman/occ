#pragma once
#include <occ/qm/spinorbital.h>
#include <occ/dft/functional.h>
#include <occ/gto/gto.h>
#include <occ/core/linear_algebra.h>


namespace occ::dft {

using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;

template<occ::qm::SpinorbitalKind spinorbital_kind, int derivative_order>
void xc_potential_matrix(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy) = delete;

template<>
void xc_potential_matrix<Restricted, 0>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

template<>
void xc_potential_matrix<Restricted, 1>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

template<>
void xc_potential_matrix<Restricted, 2>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

template<>
void xc_potential_matrix<Unrestricted, 0>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

template<>
void xc_potential_matrix<Unrestricted, 1>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

template<>
void xc_potential_matrix<Unrestricted, 2>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& KK, double &energy);

}