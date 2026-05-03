#include <occ/core/parallel.h>
#include <occ/qm/cint_interface.h>
#include <occ/xtb/multipole_ints.h>

namespace occ::xtb {

using qm::cint::Operator;
using ShellKind = gto::Shell::Kind;
using IntEnv = qm::cint::IntegralEnvironment;

namespace {

// Generic per-shell-pair multipole integral evaluator. Returns a flat array
// of `n_components` nbf×nbf matrices stacked in cint buffer order. The cint
// buffer for op contains `n_components` blocks, each (dims[0]*dims[1]) doubles.
template <Operator op, ShellKind kind>
std::vector<Mat> compute_multipole_blocks(const gto::AOBasis &basis,
                                          IntEnv &env, int n_components) {
  const auto nbf = basis.nbf();
  const auto nsh = basis.size();

  std::vector<Mat> result(n_components, Mat::Zero(nbf, nbf));

  occ::qm::cint::Optimizer opt(env, op, 2);
  std::unique_ptr<double[]> buffer(new double[env.buffer_size_1e(op)]);
  const auto &first_bf = basis.first_bf();

  for (int p = 0; p < static_cast<int>(nsh); ++p) {
    const int bf_p = first_bf[p];
    const int n_p = static_cast<int>(basis[p].size());
    for (int q = 0; q <= p; ++q) {
      const int bf_q = first_bf[q];
      const int n_q = static_cast<int>(basis[q].size());
      std::array<int, 2> idxs{p, q};
      auto dims = env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                  buffer.get(), nullptr);
      if (dims[0] < 0)
        continue;
      const size_t block_size =
          static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]);
      for (int n = 0; n < n_components; ++n) {
        Eigen::Map<const Mat> tmp(buffer.get() + n * block_size, dims[0],
                                  dims[1]);
        result[n].block(bf_p, bf_q, n_p, n_q) = tmp;
        if (p != q) {
          result[n].block(bf_q, bf_p, n_q, n_p) = tmp.transpose();
        }
      }
    }
  }
  return result;
}

} // namespace

MatTriple dipole_ao_matrices(qm::IntegralEngine &engine, const Vec3 &origin) {
  auto &env = engine.env();
  env.set_common_origin({origin.x(), origin.y(), origin.z()});

  std::vector<Mat> blocks;
  if (engine.is_spherical()) {
    blocks = compute_multipole_blocks<Operator::dipole, ShellKind::Spherical>(
        engine.aobasis(), env, 3);
  } else {
    blocks = compute_multipole_blocks<Operator::dipole, ShellKind::Cartesian>(
        engine.aobasis(), env, 3);
  }
  // restore default origin
  env.set_common_origin({0.0, 0.0, 0.0});

  MatTriple D;
  D.x = std::move(blocks[0]);
  D.y = std::move(blocks[1]);
  D.z = std::move(blocks[2]);
  return D;
}

std::array<Mat, 6> quadrupole_ao_matrices(qm::IntegralEngine &engine,
                                          const Vec3 &origin) {
  auto &env = engine.env();
  env.set_common_origin({origin.x(), origin.y(), origin.z()});

  std::vector<Mat> blocks;
  if (engine.is_spherical()) {
    blocks =
        compute_multipole_blocks<Operator::quadrupole, ShellKind::Spherical>(
            engine.aobasis(), env, 9);
  } else {
    blocks =
        compute_multipole_blocks<Operator::quadrupole, ShellKind::Cartesian>(
            engine.aobasis(), env, 9);
  }
  env.set_common_origin({0.0, 0.0, 0.0});

  // cint orders the 9 quadrupole components row-major as {xx, xy, xz, yx,
  // yy, yz, zx, zy, zz}; we return only the 6 unique upper-triangle entries.
  return {std::move(blocks[0]), std::move(blocks[1]), std::move(blocks[2]),
          std::move(blocks[4]), std::move(blocks[5]), std::move(blocks[8])};
}

} // namespace occ::xtb
