#pragma once
#include <occ/core/log.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>

namespace occ::qm {

namespace impl {

double accumulate1(SpinorbitalKind sk, int r, Mat op, Mat D) {
  double result = 0.0;
  switch (sk) {
  case SpinorbitalKind::Unrestricted: {
    result += 2 * op.col(r).dot(block::a(D).col(r) + block::b(D).col(r));
    break;
  }
  case SpinorbitalKind::General: {
    result += 2 * op.col(r).dot(block::aa(D).col(r) + block::bb(D).col(r));
    break;
  }
  case SpinorbitalKind::Restricted:
    result += 4 * op.col(r).dot(D.col(r));
    break;
  }
  return result;
}

double accumulate2(SpinorbitalKind sk, int r, Mat op, Mat D) {
  double result = 0.0;
  switch (sk) {
  case SpinorbitalKind::Unrestricted: {
    result += 4 * block::a(op).row(r).dot(block::a(D).row(r));
    result += 4 * block::b(op).row(r).dot(block::b(D).row(r));
    break;
  }
  case SpinorbitalKind::General: {
    result += 16 * op.row(r).dot(D.row(r));
    break;
  }
  default: {
    result += 4 * op.row(r).dot(D.row(r));
    break;
  }
  }
  return result;
}

} // namespace impl

template <typename Proc> class GradientEvaluator {

public:
  explicit GradientEvaluator(Proc &p)
      : m_proc(p), m_gradients(Mat3N::Zero(3, p.atoms().size())) {}

  inline Mat3N nuclear_repulsion() const {
    return m_proc.nuclear_repulsion_gradient();
  }

  inline Mat3N electronic(const MolecularOrbitals &mo) const {
    const auto &atoms = m_proc.atoms();
    const auto &basis = m_proc.aobasis();
    const auto &first_bf = basis.first_bf();
    const auto &atom_to_shell = basis.atom_to_shell();
    occ::log::info("computing atomic gradients");

    Mat3N result = Mat3N::Zero(3, atoms.size());
    auto ovlp = m_proc.compute_overlap_gradient();
    auto en = m_proc.compute_nuclear_attraction_gradient();
    auto kin = m_proc.compute_kinetic_gradient();
    auto f = m_proc.compute_fock_gradient(mo);
    auto hcore = en + kin;

    auto Dweighted = mo.energy_weighted_density_matrix();

    for (size_t atom = 0; atom < atoms.size(); atom++) {
      auto grad_rinv = m_proc.compute_rinv_gradient_for_atom(atom);

      grad_rinv.scale_by(-1.0 * atoms[atom].atomic_number);

      double x = 0.0, y = 0.0, z = 0.0;

      for (int s : atom_to_shell[atom]) {
        const auto &sh = basis[s];
        for (int bf0 = first_bf[s]; bf0 < first_bf[s] + sh.size(); bf0++) {

          grad_rinv.x.row(bf0) -= hcore.x.row(bf0);
          grad_rinv.y.row(bf0) -= hcore.y.row(bf0);
          grad_rinv.z.row(bf0) -= hcore.z.row(bf0);

          x += impl::accumulate2(mo.kind, bf0, f.x, mo.D);
          y += impl::accumulate2(mo.kind, bf0, f.y, mo.D);
          z += impl::accumulate2(mo.kind, bf0, f.z, mo.D);

          x -= impl::accumulate1(mo.kind, bf0, ovlp.x, Dweighted);
          y -= impl::accumulate1(mo.kind, bf0, ovlp.y, Dweighted);
          z -= impl::accumulate1(mo.kind, bf0, ovlp.z, Dweighted);
        }
      }
      // this term checked
      grad_rinv.symmetrize();
      grad_rinv.scale_by(2.0);

      x += 2 * occ::qm::expectation(mo.kind, mo.D, grad_rinv.x);
      y += 2 * occ::qm::expectation(mo.kind, mo.D, grad_rinv.y);
      z += 2 * occ::qm::expectation(mo.kind, mo.D, grad_rinv.z);

      result(0, atom) += x;
      result(1, atom) += y;
      result(2, atom) += z;
    }
    return result;
  }

  inline const Mat3N &operator()(const MolecularOrbitals &mo) {

    m_gradients = nuclear_repulsion();
    m_gradients += electronic(mo);
    return m_gradients;
  }

private:
  Proc &m_proc;
  Mat3N m_gradients;
};

} // namespace occ::qm
