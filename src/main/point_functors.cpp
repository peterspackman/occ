#include <occ/main/point_functors.h>
#include <occ/core/units.h>
#include <occ/gto/density.h>
#include <occ/core/eeq.h>

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


ElectronDensityFunctor::ElectronDensityFunctor(const Wavefunction &w, Spin s) : wfn(w), spin(s) {}

void ElectronDensityFunctor::operator()(Eigen::Ref<const Mat3N> points, Eigen::Ref<Vec> dest) {

    constexpr auto R = qm::SpinorbitalKind::Restricted;
    constexpr auto U = qm::SpinorbitalKind::Restricted;

    auto gto_values = occ::gto::evaluate_basis(wfn.basis, points, 0);

    switch(wfn.mo.kind) {
	case U: {
	    Vec tmp = occ::density::evaluate_density<0, U>(wfn.mo.D, gto_values);
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
	    
	}
	default:
	    occ::density::evaluate_density<0, R>(wfn.mo.D, gto_values, dest);
    }
}



}
