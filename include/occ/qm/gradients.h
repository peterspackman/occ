#pragma once
#include <cwchar>
#include <occ/qm/mo.h>
#include <occ/qm/expectation.h>

namespace occ::qm {

template<typename Proc>
class GradientEvaluator {

public:
    explicit GradientEvaluator(Proc &p) : m_proc(p),
	m_gradients(Mat3N::Zero(3, p.atoms().size())) {}


    inline Mat3N nuclear_repulsion() const {
	return m_proc.nuclear_repulsion_gradient();
    }

    inline Mat3N electronic(const MolecularOrbitals &mo) const {
	const auto &atoms = m_proc.atoms();
	const auto &basis = m_proc.aobasis();
	const auto &bf_to_atom = basis.bf_to_atom();
	const auto &first_bf = basis.first_bf();
	const auto &atom_to_shell = basis.atom_to_shell();

	Mat3N result = Mat3N::Zero(3, atoms.size());
	auto ovlp = m_proc.compute_overlap_gradient();
	auto en = m_proc.compute_nuclear_attraction_gradient();
	auto kin = m_proc.compute_kinetic_gradient();
	auto f = m_proc.compute_fock_gradient(mo);
	auto hcore = en + kin;

	auto Dweighted = mo.energy_weighted_density_matrix();

	for(size_t atom = 0; atom < atoms.size(); atom++) {
	    auto grad_rinv = m_proc.compute_rinv_gradient_for_atom(atom);

	    grad_rinv.scale_by(-1.0 * atoms[atom].atomic_number);


	    double x = 0.0, y = 0.0, z = 0.0;

	    for(int s : atom_to_shell[atom]) {
		const auto &sh = basis[s];
		for(int bf0 = first_bf[s]; bf0 < first_bf[s] + sh.size(); bf0++) {
		    grad_rinv.x.row(bf0) -= hcore.x.row(bf0);
		    grad_rinv.y.row(bf0) -= hcore.y.row(bf0);
		    grad_rinv.z.row(bf0) -= hcore.z.row(bf0);

		    x += 4 * f.x.row(bf0).dot(mo.D.row(bf0));
		    y += 4 * f.y.row(bf0).dot(mo.D.row(bf0));
		    z += 4 * f.z.row(bf0).dot(mo.D.row(bf0));

		    x -= 4 * ovlp.x.col(bf0).dot(Dweighted.col(bf0));
		    y -= 4 * ovlp.y.col(bf0).dot(Dweighted.col(bf0));
		    z -= 4 * ovlp.z.col(bf0).dot(Dweighted.col(bf0));
		}
	    }
	    grad_rinv.symmetrize();
	    grad_rinv.scale_by(2.0);

	    x += 2 * occ::qm::expectation(mo.kind, grad_rinv.x, mo.D);
	    y += 2 * occ::qm::expectation(mo.kind, grad_rinv.y, mo.D);
	    z += 2 * occ::qm::expectation(mo.kind, grad_rinv.z, mo.D);

	    result(0, atom) += x;
	    result(1, atom) += y;
	    result(2, atom) += z;

	}
	return result;
    }


    inline const Mat3N& operator()(const MolecularOrbitals &mo) {

	m_gradients = nuclear_repulsion();
	m_gradients += electronic(mo);
	return m_gradients;
    }

private:
    Proc &m_proc;
    Mat3N m_gradients;

};

}
