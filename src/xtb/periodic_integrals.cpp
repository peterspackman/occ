#include <cmath>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/h0.h>
#include <occ/xtb/periodic_integrals.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Helper: translate a copy of the central-cell atoms by T (Bohr).
std::vector<core::Atom> translated_atoms(const std::vector<core::Atom> &atoms,
                                         const Vec3 &t) {
  std::vector<core::Atom> out = atoms;
  for (auto &a : out) {
    a.x += t.x();
    a.y += t.y();
    a.z += t.z();
  }
  return out;
}

bool is_central(const LatticeImage &im) {
  return im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0;
}

} // namespace

std::vector<Mat>
periodic_overlap_blocks(const PeriodicSystem &sys,
                        const Gfn2Parameters &params,
                        const std::vector<LatticeImage> &translations) {
  // Central-cell basis (used as the reference for nbf and as the T=0 block).
  auto central_basis = build_aobasis(sys.atoms, params);
  const int nbf = static_cast<int>(central_basis.nbf());

  // T=0: molecular overlap.
  qm::IntegralEngine central_engine(central_basis);
  Mat S0 = central_engine.one_electron_operator(
      qm::IntegralEngine::Op::overlap);

  std::vector<Mat> result;
  result.reserve(translations.size());
  for (const auto &im : translations) {
    if (is_central(im)) {
      result.push_back(S0);
      continue;
    }
    // Build a "two-cell" basis: central atoms + central atoms translated.
    auto translated = translated_atoms(sys.atoms, im.t_bohr);
    auto translated_basis = build_aobasis(translated, params);
    // Merged basis: central first, translated second (so the (0:nbf, nbf:2nbf)
    // block of the resulting overlap is S^T).
    gto::AOBasis merged = central_basis;
    merged.merge(translated_basis);
    qm::IntegralEngine engine(merged);
    Mat S_merged = engine.one_electron_operator(
        qm::IntegralEngine::Op::overlap);
    result.push_back(S_merged.block(0, nbf, nbf, nbf));
  }
  return result;
}

std::vector<Mat>
periodic_h0_blocks(const PeriodicSystem &sys, const Gfn2Parameters &params,
                   const std::vector<LatticeImage> &translations,
                   const std::vector<Mat> &S_per_T, const Vec &cn) {
  if (S_per_T.size() != translations.size()) {
    throw std::runtime_error("periodic_h0_blocks: S/T size mismatch");
  }
  auto central_basis = build_aobasis(sys.atoms, params);
  auto central_shells = build_shell_table(sys.atoms, params);
  const int nbf = static_cast<int>(central_basis.nbf());

  // T=0: molecular H0.
  Mat H0_central = build_h0(sys.atoms, params, central_shells, central_basis,
                             S_per_T[0], cn);

  std::vector<Mat> result;
  result.reserve(translations.size());
  for (size_t ti = 0; ti < translations.size(); ++ti) {
    const auto &im = translations[ti];
    if (is_central(im)) {
      result.push_back(H0_central);
      continue;
    }
    // Two-cell: central + translated. By lattice translation symmetry the
    // CN at every cell is the same, so concatenate cn with itself.
    auto translated = translated_atoms(sys.atoms, im.t_bohr);
    std::vector<core::Atom> merged_atoms = sys.atoms;
    merged_atoms.insert(merged_atoms.end(), translated.begin(),
                        translated.end());

    auto merged_basis = central_basis;
    auto translated_basis = build_aobasis(translated, params);
    merged_basis.merge(translated_basis);

    auto merged_shells = build_shell_table(merged_atoms, params);
    Vec cn_merged(2 * cn.size());
    cn_merged.head(cn.size()) = cn;
    cn_merged.tail(cn.size()) = cn;

    // Recompute the merged overlap from blocks we already have:
    //   S_merged = [[S0,           S^T],
    //               [S^(-T) = S^T^T, S0]]
    Mat S_merged(2 * nbf, 2 * nbf);
    S_merged.block(0, 0, nbf, nbf) = S_per_T[0];
    S_merged.block(nbf, nbf, nbf, nbf) = S_per_T[0];
    S_merged.block(0, nbf, nbf, nbf) = S_per_T[ti];
    S_merged.block(nbf, 0, nbf, nbf) = S_per_T[ti].transpose();

    Mat H0_merged = build_h0(merged_atoms, params, merged_shells, merged_basis,
                              S_merged, cn_merged);
    result.push_back(H0_merged.block(0, nbf, nbf, nbf));
  }
  return result;
}

CMat bloch_sum(const std::vector<Mat> &M_per_T,
               const std::vector<LatticeImage> &translations, const Vec3 &k) {
  if (M_per_T.empty()) {
    throw std::runtime_error("bloch_sum: empty input");
  }
  CMat result = CMat::Zero(M_per_T[0].rows(), M_per_T[0].cols());
  for (size_t i = 0; i < M_per_T.size(); ++i) {
    const double phase = k.dot(translations[i].t_bohr);
    const std::complex<double> w(std::cos(phase), std::sin(phase));
    result.array() += w * M_per_T[i].cast<std::complex<double>>().array();
  }
  return result;
}

Mat bloch_sum_gamma(const std::vector<Mat> &M_per_T) {
  if (M_per_T.empty()) {
    throw std::runtime_error("bloch_sum_gamma: empty input");
  }
  Mat result = Mat::Zero(M_per_T[0].rows(), M_per_T[0].cols());
  for (const auto &m : M_per_T)
    result += m;
  return result;
}

} // namespace occ::xtb
