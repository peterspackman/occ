#pragma once
#include <occ/core/atom.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/xdm/xdm.h>
#include <string>
#include <optional>

namespace occ::qm {

enum class DispersionType {
  None,
  D4,
  XDM
};

namespace impl {

// Helper to compute D4 dispersion energy and gradient
std::pair<double, Mat3N> compute_d4_dispersion(
    const std::vector<core::Atom> &atoms,
    int charge,
    const std::string &functional);

// Helper to compute XDM dispersion energy and gradient
// If params is provided, uses those; otherwise looks up functional-specific params
std::pair<double, Mat3N> compute_xdm_dispersion(
    const AOBasis &basis,
    const MolecularOrbitals &mo,
    int charge,
    const std::string &functional,
    const std::optional<occ::xdm::XDM::Parameters> &params = std::nullopt);

inline double accumulate1(SpinorbitalKind sk, int r, Mat op, Mat D) {
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

inline double accumulate2(SpinorbitalKind sk, int r, Mat op, Mat D) {
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
      : m_proc(p), m_gradients(Mat3N::Zero(3, p.atoms().size())), m_schwarz_computed(false) {}

  /**
   * @brief Enable D4 dispersion correction
   * @param functional DFT functional name for D4 parameters (e.g., "pbe", "b3lyp")
   */
  inline void set_dispersion_d4(const std::string &functional) {
    m_dispersion_type = DispersionType::D4;
    m_dispersion_functional = functional;
  }

  /**
   * @brief Enable XDM dispersion correction
   * @param functional DFT functional name for XDM parameters (e.g., "pbe", "b3lyp")
   * @param params Optional XDM parameters to override functional defaults
   */
  inline void set_dispersion_xdm(const std::string &functional,
                                  const std::optional<occ::xdm::XDM::Parameters> &params = std::nullopt) {
    m_dispersion_type = DispersionType::XDM;
    m_dispersion_functional = functional;
    m_xdm_params = params;
  }

  inline Mat3N nuclear_repulsion() const {
    Mat3N result = m_proc.nuclear_repulsion_gradient();
    return result;
  }

  inline Mat3N electronic(const MolecularOrbitals &mo) {
    const auto &atoms = m_proc.atoms();
    const auto &basis = m_proc.aobasis();
    const auto &first_bf = basis.first_bf();
    const auto &atom_to_shell = basis.atom_to_shell();
    
    // Compute Schwarz matrix once and cache it
    if (!m_schwarz_computed) {
      occ::log::debug("Computing Schwarz screening matrix for gradients");
      m_schwarz = m_proc.compute_schwarz_ints();
      m_schwarz_computed = true;
    }
    
    occ::log::info("computing atomic gradients");

    Mat3N result = m_proc.additional_atomic_gradients(mo);
    auto ovlp = m_proc.compute_overlap_gradient();
    auto en = m_proc.compute_nuclear_attraction_gradient();
    auto kin = m_proc.compute_kinetic_gradient();
    occ::log::info("computing fock gradient with Schwarz screening");
    auto f = m_proc.compute_fock_gradient(mo, m_schwarz);
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

    occ::timing::start(occ::timing::gradient);

    m_gradients = nuclear_repulsion();
    m_gradients += electronic(mo);

    // Add dispersion gradient if enabled
    if (m_dispersion_functional.has_value()) {
      const std::string &functional = m_dispersion_functional.value();
      int charge = 0; // TODO: Get charge from proc if available

      switch (m_dispersion_type) {
      case DispersionType::D4: {
        occ::log::debug("Computing D4 dispersion gradient");
        const auto &atoms = m_proc.atoms();
        auto [e_disp, grad_disp] = impl::compute_d4_dispersion(atoms, charge, functional);
        occ::log::info("D4 dispersion energy: {:20.12f} Ha", e_disp);
        m_gradients += grad_disp;
        break;
      }
      case DispersionType::XDM: {
        occ::log::debug("Computing XDM dispersion gradient");
        const auto &basis = m_proc.aobasis();
        auto [e_disp, grad_disp] = impl::compute_xdm_dispersion(basis, mo, charge, functional, m_xdm_params);
        occ::log::info("XDM dispersion energy: {:20.12f} Ha", e_disp);
        m_gradients += grad_disp;
        break;
      }
      default:
        break;
      }
    }

    occ::timing::stop(occ::timing::gradient);

    // Log Cartesian gradients with atom labels
    const auto &atoms = m_proc.atoms();
    for (int atom = 0; atom < m_gradients.cols(); atom++) {
      occ::log::info("{:2s}{:3d}: {:12.8f} {:12.8f} {:12.8f}",
                     core::Element(atoms[atom].atomic_number).symbol(), atom,
                     m_gradients(0, atom), m_gradients(1, atom),
                     m_gradients(2, atom));
    }
    return m_gradients;
  }

private:
  Proc &m_proc;
  Mat3N m_gradients;
  mutable Mat m_schwarz;
  mutable bool m_schwarz_computed;
  DispersionType m_dispersion_type{DispersionType::None};
  std::optional<std::string> m_dispersion_functional;
  std::optional<occ::xdm::XDM::Parameters> m_xdm_params;
};

} // namespace occ::qm
