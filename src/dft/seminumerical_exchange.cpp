#include <occ/core/parallel.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/gto/gto.h>

namespace occ::dft::cosx {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellList = std::vector<qm::Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellKind = qm::Shell::Kind;
using Op = qm::cint::Operator;
using Buffer = std::vector<double>;
using IntegralResult = qm::IntegralEngine::IntegralResult<3>;

SemiNumericalExchange::SemiNumericalExchange(const qm::AOBasis &basis,
                                             const BeckeGridSettings &settings)
    : m_atoms(basis.atoms()), m_basis(basis), m_grid(m_basis, settings),
      m_engine(basis.atoms(), basis.shells()) {
  for (size_t i = 0; i < m_atoms.size(); i++) {
    m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
  }
  m_overlap = m_engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  m_numerical_overlap = compute_overlap_matrix();
  fmt::print("Max error |Sn - S|: {:12.8f}\n",
             (m_numerical_overlap - m_overlap).array().cwiseAbs().maxCoeff());
  m_overlap_projector = m_numerical_overlap.ldlt().solve(m_overlap);
}

Mat SemiNumericalExchange::compute_overlap_matrix() const {
  const auto &basis = m_engine.aobasis();
  size_t nbf = basis.nbf();
  using occ::parallel::nthreads;
  constexpr size_t BLOCKSIZE = 64;

  std::vector<Mat> SS(nthreads);
  for (size_t i = 0; i < nthreads; i++) {
    SS[i] = Mat::Zero(nbf, nbf);
  }
  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

    auto lambda = [&](int thread_id) {
      auto &S = SS[thread_id];
      Mat rho(BLOCKSIZE, 1);
      for (size_t block = 0; block < num_blocks; block++) {
        if (block % nthreads != thread_id)
          continue;
        Eigen::Index l = block * BLOCKSIZE;
        Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
        Eigen::Index npt = u - l;
        if (npt <= 0)
          continue;

        const auto &pts_block = atom_pts.middleCols(l, npt);
        const auto &weights_block = atom_weights.segment(l, npt);
        occ::gto::GTOValues ao;
        occ::gto::evaluate_basis(basis, pts_block, ao, 0);
        S.noalias() +=
            ao.phi.transpose() *
            (ao.phi.array().colwise() * weights_block.array()).matrix();
      }
    };
    occ::parallel::parallel_do(lambda);
  }

  for (size_t i = 1; i < nthreads; i++) {
    SS[0].noalias() += SS[i];
  }
  return SS[0];
}

template <ShellKind kind, typename Lambda>
void three_center_screened_aux_kernel(
    Lambda &f, qm::cint::IntegralEnvironment &env, const qm::AOBasis &aobasis,
    const qm::AOBasis &auxbasis, const ShellPairList &shellpairs,

    occ::qm::cint::Optimizer &opt, int thread_id = 0) noexcept {
  auto nthreads = occ::parallel::get_num_threads();
  size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                   auxbasis.max_shell_size();
  auto buffer = std::make_unique<double[]>(bufsize);
  IntegralResult args;
  args.thread = thread_id;
  args.buffer = buffer.get();
  std::array<int, 3> shell_idx;
  const auto &first_bf_ao = aobasis.first_bf();
  const auto &first_bf_aux = auxbasis.first_bf();
  for (int auxP = 0; auxP < auxbasis.size(); auxP++) {
    if (auxP % nthreads != thread_id)
      continue;
    const auto &shauxP = auxbasis[auxP];
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &shp = aobasis[p];
      const auto &plist = shellpairs[p];
      if ((shp.extent > 0.0) &&
          (shp.origin - shauxP.origin).norm() > shp.extent) {
        continue;
      }
      for (const int q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        const auto &shq = aobasis[q];
        shell_idx = {p, q, auxP + static_cast<int>(aobasis.size())};
        if ((shq.extent > 0.0) &&
            (shq.origin - shauxP.origin).norm() > shq.extent) {
          continue;
        }
        args.dims = env.three_center_helper<Op::coulomb, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  }
}

Mat SemiNumericalExchange::compute_K(const qm::MolecularOrbitals &mo,
                                     double precision,
                                     const Mat &Schwarz) const {
  size_t nbf = m_basis.nbf();
  const auto &D = mo.D;
  Mat D2 = 2 * D;
  constexpr size_t BLOCKSIZE = 128;
  using occ::parallel::nthreads;

  Mat Sn = Mat::Zero(nbf, nbf);

  Mat D2q = m_overlap_projector * D2;
  Mat K = Mat::Zero(nbf, nbf);
  const auto &basis = m_engine.aobasis();
  const auto shell2bf = basis.first_bf();
  occ::qm::cint::Optimizer opt(m_engine.env(), Op::coulomb, 3);

  Mat Fg(BLOCKSIZE, nbf);
  Mat Gg(BLOCKSIZE, nbf);
  Mat rho(BLOCKSIZE, 1);

  auto f = [&Gg, &Fg](const qm::IntegralEngine::IntegralResult<3> &args) {
    int n = args.shell[2];
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    Gg.block(n, args.bf[1], 1, args.dims[1]) +=
        Fg.block(n, args.bf[0], 1, args.dims[0]) * tmp;
    if (args.shell[0] != args.shell[1]) {
      Gg.block(n, args.bf[0], 1, args.dims[0]) +=
          Fg.block(n, args.bf[1], 1, args.dims[1]) * tmp.transpose();
    }
  };
  auto lambda = [&](int thread_id) {
    if (m_engine.is_spherical()) {
      three_center_screened_aux_kernel<qm::Shell::Kind::Spherical>(
          f, m_engine.env(), m_engine.aobasis(), m_engine.auxbasis(),
          m_engine.shellpairs(), opt, thread_id);
    } else {
      three_center_screened_aux_kernel<qm::Shell::Kind::Cartesian>(
          f, m_engine.env(), m_engine.aobasis(), m_engine.auxbasis(),
          m_engine.shellpairs(), opt, thread_id);
    }
  };

  std::vector<qm::Atom> dummy_atoms;
  std::vector<qm::Shell> aux_shells;

  // compute J, K
  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;
    occ::gto::GTOValues ao;
    for (size_t block = 0; block < num_blocks; block++) {
      Eigen::Index l = block * BLOCKSIZE;
      Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
      Eigen::Index npt = u - l;
      if (npt <= 0)
        continue;

      const auto &pts_block = atom_pts.middleCols(l, npt);
      const auto &weights_block = atom_weights.segment(l, npt);
      occ::gto::evaluate_basis(basis, pts_block, ao, 0);
      if (ao.phi.maxCoeff() < precision)
        continue;

      dummy_atoms.resize(npt);
      aux_shells.resize(npt);
      for (size_t pt = 0; pt < npt; pt++) {
        dummy_atoms[pt] = {0, pts_block(0, pt), pts_block(1, pt),
                           pts_block(2, pt)};
        aux_shells[pt].origin = pts_block.col(pt);
      }

      m_engine.set_dummy_basis(dummy_atoms, aux_shells);

      Mat wao = ao.phi.array().colwise() * weights_block.array();
      Fg = wao * D2q;
      Gg.setZero();

      occ::parallel::parallel_do(lambda);
      K.noalias() -= ao.phi.transpose() * Gg.block(0, 0, npt, nbf);
    }
  }

  return 0.25 * (K + K.transpose());
}

} // namespace occ::dft::cosx
