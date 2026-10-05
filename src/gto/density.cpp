#include <fmt/core.h>
#include <occ/core/parallel.h>
#include <occ/gto/density.h>

namespace occ::density {

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;

Mat evaluate_orbitals_on_grid(const occ::gto::AOBasis &basis, MatConstRef C,
                              const occ::Mat3N &points) {
  if (C.rows() != static_cast<Eigen::Index>(basis.nbf())) {
    throw std::runtime_error(fmt::format(
        "evaluate_orbitals_on_grid: {} coefficient rows for {} basis "
        "functions",
        C.rows(), basis.nbf()));
  }
  constexpr Eigen::Index block_size = 4096;
  const Eigen::Index npts = points.cols();
  Mat result(npts, C.cols());
  const size_t num_blocks = (npts + block_size - 1) / block_size;

  occ::parallel::thread_local_storage<occ::gto::GTOValues> gto_vals_local;
  occ::parallel::parallel_for(size_t(0), num_blocks, [&](size_t block) {
    auto &gto_vals = gto_vals_local.local();
    const Eigen::Index l = block * block_size;
    const Eigen::Index n = std::min(block_size, npts - l);
    occ::gto::evaluate_basis(basis, points.middleCols(l, n), gto_vals, 0);
    // phi is (n x nbf), so this is psi_i at each point
    result.middleRows(l, n).noalias() = gto_vals.phi * C;
  });
  return result;
}

template <>
void evaluate_density<0, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  // use a MatRM as a row major temporary, selfadjointView also speeds things
  // up a little.
  // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
  MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
  rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<0, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  // alpha part first
  auto Da = occ::qm::block::a(D);
  MatRM Dphi = gto_values.phi * Da;
  auto rho_a = occ::qm::block::a(rho);
  rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  auto Db = occ::qm::block::b(D);
  Dphi = gto_values.phi * Db;
  auto rho_b = occ::qm::block::b(rho);
  rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<1, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  // use a MatRM as a row major temporary, selfadjointView also speeds things
  // up a little.
  // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
  MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
  rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  rho.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<1, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  // alpha part first
  auto Da = occ::qm::block::a(D);
  MatRM Dphi = gto_values.phi * Da;
  auto rho_a = occ::qm::block::a(rho);
  rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  /*
   * If we wish to get the values interleaved for say libxc, use an
   * Eigen::Map as follows: Map<occ::Mat, 0, Stride<Dynamic,
   * 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic,
   * 2>(2*Dphi.rows(), 2)) = RHS
   */
  rho_a.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho_a.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho_a.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  // beta part
  auto Db = occ::qm::block::b(D);
  Dphi = gto_values.phi * Db;
  auto rho_b = occ::qm::block::b(rho);
  rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  rho_b.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho_b.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho_b.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<2, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  // use a MatRM as a row major temporary, selfadjointView also speeds things
  // up a little.
  // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
  MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
  rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  rho.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  // laplacian
  rho.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                     gto_values.phi_zz.array()) *
                    Dphi.array())
                       .rowwise()
                       .sum();
  // tau
  Dphi = gto_values.phi_x * D;
  rho.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_y * D;
  rho.col(5).array() +=
      (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_z * D;
  rho.col(5).array() +=
      (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();

  rho.col(4).array() += 2 * rho.col(5).array();
  rho.col(5).array() *= 0.5;
}

template <>
void evaluate_density<2, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
  occ::timing::start(occ::timing::category::fft);
  // use a MatRM as a row major temporary, selfadjointView also speeds things
  // up a little.
  // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
  // alpha part first
  auto Da = occ::qm::block::a(D);
  MatRM Dphi = gto_values.phi * Da;
  auto rho_a = occ::qm::block::a(rho);
  rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  /*
   * If we wish to get the values interleaved for say libxc, use an
   * Eigen::Map as follows: Map<occ::Mat, 0, Stride<Dynamic,
   * 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic,
   * 2>(2*Dphi.rows(), 2)) = RHS
   */
  rho_a.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho_a.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho_a.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  // laplacian
  rho_a.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                       gto_values.phi_zz.array()) *
                      Dphi.array())
                         .rowwise()
                         .sum();
  // tau
  Dphi = gto_values.phi_x * Da;
  rho_a.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_y * Da;
  rho_a.col(5).array() +=
      (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_z * Da;
  rho_a.col(5).array() +=
      (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  rho_a.col(4).array() += 2 * rho_a.col(5).array();
  rho_a.col(5).array() *= 0.5;
  // beta part
  auto Db = occ::qm::block::b(D);
  Dphi = gto_values.phi * Db;
  auto rho_b = occ::qm::block::b(rho);
  rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
  rho_b.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  rho_b.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  rho_b.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  // laplacian
  rho_b.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                       gto_values.phi_zz.array()) *
                      Dphi.array())
                         .rowwise()
                         .sum();
  // tau
  Dphi = gto_values.phi_x * Db;
  rho_b.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_y * Db;
  rho_b.col(5).array() +=
      (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
  Dphi = gto_values.phi_z * Db;
  rho_b.col(5).array() +=
      (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
  rho_b.col(4).array() += 2 * rho_b.col(5).array();
  rho_b.col(5).array() *= 0.5;
}

} // namespace occ::density
