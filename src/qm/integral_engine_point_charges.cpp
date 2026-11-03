// Point-charge (1/|r - s|) integrals contracted inside the shell-pair loop,
// with the charges' positions free to move with an owning atom: what a
// continuum solvation model needs for its nuclear derivatives and response,
// where the apparent surface charges sit on a cavity attached to the atoms.
//
// L^k_{mn} = <m| 1/|r - s_k| |n>. Derivatives with respect to the centre of a
// basis function are minus the libcint nabla on it; the derivative with
// respect to the charge position follows from translational invariance
// (d/dA + d/dB + d/ds = 0), and is credited to the charge's owner atom when it
// has one (owner < 0: the charge is fixed in space).
//
// Every kernel loops over shell pairs in parallel and over charges inside, so
// the outputs that are nbf x nbf are written block by block without races.

#include "detail/libcint_derivatives.h"
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

using IntEnv = cint::IntegralEnvironment;

namespace {

using detail::call;
using detail::Intor;
using detail::unique_pairs;

inline void set_origin(IntEnv &env, const Mat3N &points, Eigen::Index k) {
  env.set_rinv_origin({points(0, k), points(1, k), points(2, k)});
}

// sum_{m in p, n in q} X_{mn} buf(m, n) for each of `ncomp` components.
template <int ncomp>
inline Eigen::Matrix<double, ncomp, 1>
contract(const Mat &X, int bp, int bq, int np, int nq, const double *buf) {
  Eigen::Matrix<double, ncomp, 1> out;
  const size_t n = static_cast<size_t>(np) * nq;
  for (int c = 0; c < ncomp; c++) {
    Eigen::Map<const Mat> blk(buf + c * n, np, nq);
    out(c) = X.block(bp, bq, np, nq).cwiseProduct(blk).sum();
  }
  return out;
}

} // namespace

Mat IntegralEngine::point_charge_potentials(const std::vector<Mat> &D,
                                            const Mat3N &points) const {
  occ::timing::start(occ::timing::category::ints1e);
  const auto &basis = m_aobasis;
  const auto &first_bf = basis.first_bf();
  const Eigen::Index npts = points.cols();
  const Eigen::Index nd = static_cast<Eigen::Index>(D.size());
  Intor *rinv = is_spherical() ? libcint::int1e_rinv_sph
                               : libcint::int1e_rinv_cart;
  const auto pairs = unique_pairs(m_shellpairs);
  const size_t bufsize = m_env.buffer_size_1e();

  struct ThreadData {
    IntEnv env;
    std::vector<double> buf;
    Mat V;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize),
                      Mat::Zero(npts, nd)};
  });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t ip) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ip];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    // Symmetric densities: the (q, p) block doubles the (p, q) one.
    const double f = (p == q) ? 1.0 : 2.0;
    int shls[2] = {p, q};
    for (Eigen::Index k = 0; k < npts; k++) {
      set_origin(t.env, points, k);
      if (!call(rinv, t.env, t.buf.data(), shls))
        continue;
      Eigen::Map<const Mat> blk(t.buf.data(), np, nq);
      for (Eigen::Index d = 0; d < nd; d++)
        t.V(k, d) += f * D[d].block(bp, bq, np, nq).cwiseProduct(blk).sum();
    }
  });
  Mat V = Mat::Zero(npts, nd);
  for (const auto &t : tls)
    V += t.V;
  occ::timing::stop(occ::timing::category::ints1e);
  return V;
}

std::vector<Mat> IntegralEngine::point_charge_matrices(
    const Mat &Q, const Mat3N &points) const {
  occ::timing::start(occ::timing::category::ints1e);
  const auto &basis = m_aobasis;
  const auto &first_bf = basis.first_bf();
  const auto nbf = basis.nbf();
  const Eigen::Index npts = points.cols();
  const Eigen::Index nv = Q.cols();
  Intor *rinv = is_spherical() ? libcint::int1e_rinv_sph
                               : libcint::int1e_rinv_cart;
  const auto pairs = unique_pairs(m_shellpairs);
  const size_t bufsize = m_env.buffer_size_1e();

  std::vector<Mat> result(nv, Mat::Zero(nbf, nbf));
  struct ThreadData {
    IntEnv env;
    std::vector<double> buf;
    std::vector<Mat> blocks;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize), {}};
  });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t ip) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ip];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    t.blocks.assign(nv, Mat::Zero(np, nq));
    int shls[2] = {p, q};
    for (Eigen::Index k = 0; k < npts; k++) {
      set_origin(t.env, points, k);
      if (!call(rinv, t.env, t.buf.data(), shls))
        continue;
      Eigen::Map<const Mat> blk(t.buf.data(), np, nq);
      for (Eigen::Index v = 0; v < nv; v++)
        t.blocks[v] += Q(k, v) * blk;
    }
    for (Eigen::Index v = 0; v < nv; v++) {
      result[v].block(bp, bq, np, nq) = t.blocks[v];
      if (p != q)
        result[v].block(bq, bp, nq, np) = t.blocks[v].transpose();
    }
  });
  occ::timing::stop(occ::timing::category::ints1e);
  return result;
}

Mat IntegralEngine::point_charge_potential_derivatives(
    const Mat &D, const Mat3N &points, const IVec &owner) const {
  occ::timing::start(occ::timing::category::ints1e);
  const auto &basis = m_aobasis;
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const Eigen::Index n3 = 3 * static_cast<Eigen::Index>(basis.atoms().size());
  const Eigen::Index npts = points.cols();
  Intor *iprinv = is_spherical() ? libcint::int1e_iprinv_sph
                                 : libcint::int1e_iprinv_cart;
  const auto pairs = unique_pairs(m_shellpairs);
  const size_t bufsize = 3 * m_env.buffer_size_1e();

  struct ThreadData {
    IntEnv env;
    std::vector<double> b0, b1;
    Mat G;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize),
                      std::vector<double>(bufsize), Mat::Zero(n3, npts)};
  });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t ip) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ip];
    const int M = shell_atom[p], N = shell_atom[q];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    const double f = (p == q) ? 1.0 : 2.0;
    int pq[2] = {p, q}, qp[2] = {q, p};
    for (Eigen::Index k = 0; k < npts; k++) {
      const int c = owner(k);
      if (M == N && M == c)
        continue;
      set_origin(t.env, points, k);
      Vec3 gM = Vec3::Zero(), gN = Vec3::Zero();
      if (call(iprinv, t.env, t.b0.data(), pq))
        gM = -f * contract<3>(D, bp, bq, np, nq, t.b0.data());
      if (p == q)
        gN = gM;
      else if (call(iprinv, t.env, t.b1.data(), qp))
        gN = -f * contract<3>(D, bq, bp, nq, np, t.b1.data());
      t.G.block<3, 1>(3 * M, k) += gM;
      t.G.block<3, 1>(3 * N, k) += gN;
      if (c >= 0)
        t.G.block<3, 1>(3 * c, k) -= gM + gN;
    }
  });
  Mat G = Mat::Zero(n3, npts);
  for (const auto &t : tls)
    G += t.G;
  occ::timing::stop(occ::timing::category::ints1e);
  return G;
}

Mat IntegralEngine::point_charge_hessian(const Mat &D, const Vec &charges,
                                         const Mat3N &points,
                                         const IVec &owner) const {
  occ::timing::start(occ::timing::category::ints1e);
  const bool sph = is_spherical();
  const auto &basis = m_aobasis;
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const Eigen::Index n3 = 3 * static_cast<Eigen::Index>(basis.atoms().size());
  const Eigen::Index npts = points.cols();
  Intor *ipiprinv =
      sph ? libcint::int1e_ipiprinv_sph : libcint::int1e_ipiprinv_cart;
  Intor *iprinvip =
      sph ? libcint::int1e_iprinvip_sph : libcint::int1e_iprinvip_cart;
  const auto pairs = unique_pairs(m_shellpairs);
  const size_t bufsize = 9 * m_env.buffer_size_1e();

  struct ThreadData {
    IntEnv env;
    std::vector<double> b0, b1, b2;
    Mat H;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize),
                      std::vector<double>(bufsize),
                      std::vector<double>(bufsize), Mat::Zero(n3, n3)};
  });
  auto as_mat3 = [](const Eigen::Matrix<double, 9, 1> &v) {
    Mat3 m;
    for (int c = 0; c < 9; c++)
      m(c / 3, c % 3) = v(c);
    return m;
  };
  // As the nuclear-attraction part of one_electron_hessian, with the
  // operator centre at each charge and its blocks credited to the owner.
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t ip) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ip];
    const int M = shell_atom[p], N = shell_atom[q];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    int pq[2] = {p, q}, qp[2] = {q, p};
    for (Eigen::Index k = 0; k < npts; k++) {
      const int C = owner(k);
      if (M == N && M == C)
        continue;
      const double f = charges(k) * ((p == q) ? 1.0 : 2.0);
      if (f == 0.0)
        continue;
      set_origin(t.env, points, k);
      Mat3 MM = Mat3::Zero(), NN = Mat3::Zero(), MN = Mat3::Zero();
      if (call(ipiprinv, t.env, t.b0.data(), pq))
        MM = f * as_mat3(contract<9>(D, bp, bq, np, nq, t.b0.data()));
      if (call(ipiprinv, t.env, t.b1.data(), qp))
        NN = f * as_mat3(contract<9>(D, bq, bp, nq, np, t.b1.data()));
      if (call(iprinvip, t.env, t.b2.data(), pq))
        MN = f * as_mat3(contract<9>(D, bp, bq, np, nq, t.b2.data()));
      detail::add_operator_centre_blocks(t.H, M, N, C, MM, NN, MN);
    }
  });
  Mat H = Mat::Zero(n3, n3);
  for (const auto &t : tls)
    H += t.H;
  occ::timing::stop(occ::timing::category::ints1e);
  return H;
}

std::vector<Mat> IntegralEngine::point_charge_derivative_matrices(
    const Vec &charges, const Mat3N &points, const IVec &owner) const {
  occ::timing::start(occ::timing::category::ints1e);
  const auto &basis = m_aobasis;
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const auto nbf = basis.nbf();
  const Eigen::Index n3 = 3 * static_cast<Eigen::Index>(basis.atoms().size());
  const Eigen::Index npts = points.cols();
  Intor *iprinv = is_spherical() ? libcint::int1e_iprinv_sph
                                 : libcint::int1e_iprinv_cart;
  const auto pairs = unique_pairs(m_shellpairs);
  const size_t bufsize = 3 * m_env.buffer_size_1e();

  std::vector<Mat> result(n3, Mat::Zero(nbf, nbf));
  struct ThreadData {
    IntEnv env;
    std::vector<double> b0, b1;
    std::vector<Mat> blocks;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize),
                      std::vector<double>(bufsize), {}};
  });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t ip) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ip];
    const int M = shell_atom[p], N = shell_atom[q];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    const size_t n = static_cast<size_t>(np) * nq;
    t.blocks.assign(n3, Mat::Zero(np, nq));
    int pq[2] = {p, q}, qp[2] = {q, p};
    for (Eigen::Index k = 0; k < npts; k++) {
      const int c = owner(k);
      if (M == N && M == c)
        continue;
      const double w = charges(k);
      if (w == 0.0)
        continue;
      set_origin(t.env, points, k);
      // d/dM <p|L|q> = -<nabla p|L|q>, d/dN <p|L|q> = -<p|L|nabla q>
      if (call(iprinv, t.env, t.b0.data(), pq)) {
        for (int x = 0; x < 3; x++) {
          Eigen::Map<const Mat> d(t.b0.data() + x * n, np, nq);
          t.blocks[3 * M + x] -= w * d;
          if (c >= 0)
            t.blocks[3 * c + x] += w * d;
        }
      }
      if (call(iprinv, t.env, t.b1.data(), qp)) {
        for (int x = 0; x < 3; x++) {
          Eigen::Map<const Mat> d(t.b1.data() + x * n, nq, np);
          t.blocks[3 * N + x] -= w * d.transpose();
          if (c >= 0)
            t.blocks[3 * c + x] += w * d.transpose();
        }
      }
    }
    for (Eigen::Index x = 0; x < n3; x++) {
      result[x].block(bp, bq, np, nq) = t.blocks[x];
      if (p != q)
        result[x].block(bq, bp, nq, np) = t.blocks[x].transpose();
    }
  });
  occ::timing::stop(occ::timing::category::ints1e);
  return result;
}

} // namespace occ::qm
