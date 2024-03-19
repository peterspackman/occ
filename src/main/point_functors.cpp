#include <occ/main/point_functors.h>
#include <occ/core/units.h>
#include <occ/gto/density.h>
#include <occ/core/eeq.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/element.h>

namespace occ::main {


std::pair<IVec, Mat3N> atom_nums_positions(const AtomList &atoms) {
    IVec n(atoms.size());
    Mat3N p(3, atoms.size());
    for(int i = 0; i < atoms.size(); i++) {
	n(i) = atoms[i].atomic_number;
	p(0, i) = atoms[i].x * occ::units::BOHR_TO_ANGSTROM;
	p(1, i) = atoms[i].y * occ::units::BOHR_TO_ANGSTROM;
	p(2, i) = atoms[i].z * occ::units::BOHR_TO_ANGSTROM;
    }
    return {n, p};
}


EspFunctor::EspFunctor(const Wavefunction &w) : wfn(w), hf(w.basis) {}

void EspFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> result) {
    result += hf.electronic_electric_potential_contribution(wfn.mo, points);
    result += hf.nuclear_electric_potential_contribution(points);
}

EEQEspFunctor::EEQEspFunctor(const AtomList &a, double charge) :
    atoms(a) {
    auto [p, n] = atom_nums_positions(a);
    charges = occ::core::charges::eeq_partial_charges(p, n, 0.0);
}

void EEQEspFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> result) {
    for(int i = 0; i < atoms.size(); i++) {
	Vec3 pos(atoms[i].x, atoms[i].y, atoms[i].z);
	result.array() += 
	    charges(i) / (points.colwise() - pos).colwise().norm().array();
    }
}

PromolDensityFunctor::PromolDensityFunctor(const AtomList &a) : atoms(a) {

    auto basis = occ::slater::load_slaterbasis("thakkar");
    ankerl::unordered_dense::map<int, std::vector<int>> tmp_map;
    occ::log::debug("Loaded slater basis");

    // TODO handle charges
    Eigen::Matrix3Xf coordinates(3, atoms.size());
    for(int i = 0; i < atoms.size(); i++) {
	coordinates(0, i) = atoms[i].x;
	coordinates(1, i) = atoms[i].y;
	coordinates(2, i) = atoms[i].z;
    }

    occ::log::debug("Built coordinates");

    for (size_t i = 0; i < atoms.size(); i++) {
        int el = atoms[i].atomic_number;
        tmp_map[el].push_back(i);
    }
    occ::log::debug("Built interpolators");

    for (const auto &[el, idxs]: tmp_map) {
	auto b = basis[occ::core::Element(el).symbol()];
	auto func = [&b](float x) { return b.rho(std::sqrt(x)); };
	atom_interpolators.push_back(pfimpl::AtomInterpolator{
		pfimpl::LinearInterpolatorFloat(
		func, interpolator_params.domain_lower,
		interpolator_params.domain_upper,
		interpolator_params.num_points),
	    coordinates(Eigen::all, idxs)
	});
    }

    for(auto &ai: atom_interpolators) {
	ai.threshold = ai.interpolator.find_threshold(1e-8);
    }
    occ::log::debug("Built atom interpolators");
}

void PromolDensityFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest) {
    for(int pt = 0; pt < points.cols(); pt++) {
        float result{0.0};
	Eigen::Vector3f pos = points.col(pt).cast<float>();
        for (const auto &[interp, interp_positions, threshold] :
             atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                float r = (interp_positions.col(i) - pos).squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                result += rho;
            }
        }
	dest(pt) += result;
    }
}

ElectronDensityFunctor::ElectronDensityFunctor(const Wavefunction &w, Spin s) : wfn(w), spin(s) {}

void ElectronDensityFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest) {

    constexpr auto R = qm::SpinorbitalKind::Restricted;
    constexpr auto U = qm::SpinorbitalKind::Unrestricted;
    constexpr auto G = qm::SpinorbitalKind::General;

    auto gto_values = occ::gto::evaluate_basis(wfn.basis, points, 0);

    Mat D = 2 * wfn.mo.D;
    if(mo_index >= 0) {
	D = wfn.mo.density_matrix_single_mo(mo_index);
    }

    switch(wfn.mo.kind) {
	case U: {
	    Vec tmp = occ::density::evaluate_density<0, U>(D, gto_values);
	    switch(spin) {
		case Spin::Total:
		    dest += qm::block::a(tmp);
		    dest += qm::block::b(tmp);
		    break;
		case Spin::Alpha:
		    dest += qm::block::a(tmp);
		    break;
		case Spin::Beta:
		    dest += qm::block::b(tmp);
		    break;
	    }
	    break;
	}
	case G: {
	    throw std::runtime_error("General case not implemented");
	    break;
	}
	default: {
	    Vec tmp = occ::density::evaluate_density<0, R>(D, gto_values);
	    dest += tmp;
	    break;
	}
    }
}

DeformationDensityFunctor::DeformationDensityFunctor(const Wavefunction &wfn,
	ElectronDensityFunctor::Spin spin) : rho_func(wfn, spin), pro_func(wfn.atoms) {}

void DeformationDensityFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest) {
    Vec tmp = Vec::Zero(dest.rows(), dest.cols());
    rho_func(points, dest);
    pro_func(points, tmp);
    if(rho_func.spin != ElectronDensityFunctor::Spin::Total) {
	dest -= 0.5 * tmp;
    }
    else {
	dest -= tmp;
    }
}


}
