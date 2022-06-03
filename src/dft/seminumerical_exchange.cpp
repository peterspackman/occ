#include <occ/core/parallel.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/gto/gto.h>
#include <occ/qm/ints.h>

namespace occ::dft::cosx {

using libint2::Operator;

SemiNumericalExchange::SemiNumericalExchange(
    const std::vector<occ::core::Atom> &atoms, const qm::BasisSet &basis)
    : m_atoms(atoms), m_basis(basis),
      m_engine(atoms, occ::qm::from_libint2_basis(basis)),
      m_grid(m_basis, m_atoms) {
    for (size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    if (m_engine.is_spherical()) {
        m_overlap =
            m_engine.one_electron_operator<qm::IntegralEngine::Op::overlap,
                                           qm::OccShell::Kind::Spherical>();
    } else {
        m_overlap =
            m_engine.one_electron_operator<qm::IntegralEngine::Op::overlap,
                                           qm::OccShell::Kind::Cartesian>();
    }
    m_numerical_overlap = compute_overlap_matrix();
    fmt::print("Max error |Sn - S|: {:12.8f}\n",
               (m_numerical_overlap - m_overlap).array().cwiseAbs().maxCoeff());
    std::cout << "S\n" << m_overlap.block(0, 0, 5, 5) << '\n';
    std::cout << "S\n" << m_numerical_overlap.block(0, 0, 5, 5) << '\n';
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
                Eigen::Index u =
                    std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
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

Mat SemiNumericalExchange::compute_K(qm::SpinorbitalKind kind,
                                     const qm::MolecularOrbitals &mo,
                                     double precision,
                                     const Mat &Schwarz) const {
    size_t nbf = m_basis.nbf();
    const auto &D = mo.D;
    Mat D2 = 2 * D;
    constexpr size_t BLOCKSIZE = 64;
    using occ::parallel::nthreads;

    Mat Sn = Mat::Zero(nbf, nbf);

    libint2::Engine engine = libint2::Engine(
        Operator::nuclear, m_basis.max_nprim(), m_basis.max_l(), 0);

    Mat D2q = m_overlap_projector * D2;
    Mat K = Mat::Zero(nbf, nbf);
    const auto &basis = m_engine.aobasis();
    const auto nshells = basis.nsh();
    const auto shell2bf = basis.first_bf();

    std::vector<qm::Atom> dummy_atoms;
    dummy_atoms.reserve(BLOCKSIZE);
    dummy_atoms.push_back({0, 0.0, 0.0, 0.0});
    std::vector<qm::OccShell> aux_shells;
    aux_shells.reserve(BLOCKSIZE);
    aux_shells.emplace_back(occ::core::PointCharge{1.0, {0, 0, 0}});
    for (size_t i = 1; i < BLOCKSIZE; i++) {
        aux_shells.push_back(aux_shells[0]);
        dummy_atoms.push_back(dummy_atoms[0]);
    }

    // compute J, K
    for (const auto &atom_grid : m_atom_grids) {
        const auto &atom_pts = atom_grid.points;
        const auto &atom_weights = atom_grid.weights;
        const size_t npt_total = atom_pts.cols();
        const size_t num_blocks = npt_total / BLOCKSIZE + 1;
        std::vector<occ::core::PointCharge> chgs(1);
        Mat rho(BLOCKSIZE, 1);
        occ::gto::GTOValues ao;
        for (size_t block = 0; block < num_blocks; block++) {
            Eigen::Index l = block * BLOCKSIZE;
            Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
            Eigen::Index npt = u - l;
            if (npt <= 0)
                continue;

            if (npt < BLOCKSIZE) {
                aux_shells.resize(npt);
                dummy_atoms.resize(npt);
            }

            const auto &pts_block = atom_pts.middleCols(l, npt);
            const auto &weights_block = atom_weights.segment(l, npt);
            occ::gto::evaluate_basis(basis, pts_block, ao, 0);
            if (ao.phi.maxCoeff() < precision)
                continue;

            for (size_t pt = 0; pt < npt; pt++) {
                dummy_atoms[pt] = {0, pts_block(0, pt), pts_block(1, pt),
                                   pts_block(2, pt)};
                aux_shells[pt].origin = pts_block.col(pt);
            }

            m_engine.set_dummy_basis(dummy_atoms, aux_shells);

            Mat wao = ao.phi.array().colwise() * weights_block.array();
            Mat Fg = wao * D2q;

            Mat Gg = Mat::Zero(Fg.rows(), Fg.cols());
            auto f = [&Gg,
                      &Fg](const qm::IntegralEngine::IntegralResult<3> &args) {
                int n = args.shell[2];
                Eigen::Map<const Mat> tmp(args.buffer, args.dims[0],
                                          args.dims[1]);
                Gg.block(n, args.bf[1], 1, args.dims[1]) +=
                    Fg.block(n, args.bf[0], 1, args.dims[0]) * tmp;
                if (args.shell[0] != args.shell[1]) {
                    Gg.block(n, args.bf[0], 1, args.dims[0]) +=
                        Fg.block(n, args.bf[1], 1, args.dims[1]) *
                        tmp.transpose();
                }
            };
            auto lambda = [&](int thread_id) {
                if (m_engine.is_spherical()) {
                    m_engine.template evaluate_three_center_aux<
                        qm::OccShell::Kind::Spherical>(f, thread_id);
                } else {
                    m_engine.template evaluate_three_center_aux<
                        qm::OccShell::Kind::Cartesian>(f, thread_id);
                }
            };
            occ::parallel::parallel_do(lambda);
            K.noalias() -= ao.phi.transpose() * Gg;
        }
    }

    return 0.25 * (K + K.transpose());
}

} // namespace occ::dft::cosx
