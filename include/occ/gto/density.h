#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

namespace occ::density {
using occ::qm::SpinorbitalKind;

constexpr int num_components(int deriv_order) {
    switch (deriv_order) {
    case 0:
        return 1;
    case 1:
        return 4;
    case 2:
        return 6;
    }
    return 1;
}

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
void evaluate_density(MatConstRef D, const occ::gto::GTOValues &gto_values,
                      MatRef rho);

template <>
void evaluate_density<0, SpinorbitalKind::Restricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <>
void evaluate_density<1, SpinorbitalKind::Restricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <>
void evaluate_density<2, SpinorbitalKind::Restricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <>
void evaluate_density<0, SpinorbitalKind::Unrestricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <>
void evaluate_density<1, SpinorbitalKind::Unrestricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <>
void evaluate_density<2, SpinorbitalKind::Unrestricted>(
    MatConstRef D, const occ::gto::GTOValues &gto_values, MatRef rho);

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
Mat evaluate_density(const Mat &D, const occ::gto::GTOValues &gto_values) {
    const auto npt = gto_values.phi.rows();
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
        occ::Mat rho(npt, num_components(max_derivative));
        evaluate_density<max_derivative, spinorbital_kind>(D, gto_values, rho);
        return rho;
    } else {
        occ::Mat rho(2 * npt, num_components(max_derivative));
        evaluate_density<max_derivative, spinorbital_kind>(D, gto_values, rho);
        return rho;
    }
}

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
Mat evaluate_density_on_grid(const qm::AOBasis &basis, const Mat &D,
                             const occ::Mat &grid_pts) {
    auto gto_values = occ::gto::evaluate_basis(basis, grid_pts, max_derivative);
    return evaluate_density<max_derivative, spinorbital_kind>(D, gto_values);
}

template <size_t max_derivative>
Mat evaluate_density_on_grid(const qm::Wavefunction &wfn, const occ::Mat &grid_pts) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    auto gto_values = occ::gto::evaluate_basis(wfn.basis, grid_pts, max_derivative);
    switch(wfn.mo.kind) {
	case R:
	    return evaluate_density<max_derivative, R>(wfn.mo.D * 2, gto_values);
	case U:
	    return evaluate_density<max_derivative, U>(wfn.mo.D * 2, gto_values);
	case G:
	    throw std::runtime_error("Invalid spinorbital kind (general)");
    }
}


} // namespace occ::density
