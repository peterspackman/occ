#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/multipole.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/qm/occshell.h>
#include <occ/qm/shellblock_norm.h>
#include <vector>

namespace occ::qm {

class IntegralEngine {
  public:
    template <size_t num_centers> struct IntegralResult {
        int thread{0};
        std::array<int, num_centers> shell{0};
        std::array<int, num_centers> bf{0};
        std::array<int, num_centers> dims{0};
        const double *buffer{nullptr};
    };

    using ShellList = std::vector<OccShell>;
    using AtomList = std::vector<occ::core::Atom>;
    using ShellPairList = std::vector<std::vector<size_t>>;
    using IntEnv = cint::IntegralEnvironment;
    using ShellKind = OccShell::Kind;
    using Op = cint::Operator;

    IntegralEngine(const AtomList &at, const ShellList &sh)
        : m_aobasis(at, sh), m_env(at, sh) {

        if (is_spherical()) {
            compute_shellpairs<ShellKind::Spherical>();
        } else {
            compute_shellpairs<ShellKind::Cartesian>();
        }
    }

    inline auto nbf() const noexcept { return m_aobasis.nbf(); }
    inline auto nbf_aux() const noexcept { return m_auxbasis.nbf(); }
    inline auto nsh() const noexcept { return m_aobasis.nsh(); }
    inline auto nsh_aux() const noexcept { return m_auxbasis.nsh(); }
    inline const AOBasis &aobasis() const { return m_aobasis; }
    inline const AOBasis &auxbasis() const { return m_auxbasis; }

    inline const auto &first_bf() const noexcept {
        return m_aobasis.first_bf();
    }
    inline const auto &first_bf_aux() const noexcept {
        return m_auxbasis.first_bf();
    }
    inline const auto &shellpairs() const noexcept { return m_shellpairs; }
    inline const auto &shells() const noexcept { return m_aobasis.shells(); }

    inline void set_auxiliary_basis(const ShellList &bs, bool dummy = false) {
        if (!dummy) {
            clear_auxiliary_basis();
            m_auxbasis = AOBasis(m_aobasis.atoms(), bs);
            ShellList combined = m_aobasis.shells();
            combined.insert(combined.end(), m_auxbasis.shells().begin(),
                            m_auxbasis.shells().end());
            m_env = IntEnv(m_aobasis.atoms(), combined);
        } else {
            AtomList dummy_atoms;
            dummy_atoms.reserve(bs.size());
            for (const auto &shell : bs) {
                dummy_atoms.push_back(
                    {0, shell.origin(0), shell.origin(1), shell.origin(2)});
            }
            set_dummy_basis(dummy_atoms, bs);
        }
    }

    inline void set_dummy_basis(const AtomList &dummy_atoms,
                                const ShellList &bs) {
        clear_auxiliary_basis();
        m_auxbasis = AOBasis(dummy_atoms, bs);
        AtomList combined_sites = m_aobasis.atoms();
        combined_sites.insert(combined_sites.end(), dummy_atoms.begin(),
                              dummy_atoms.end());
        ShellList combined = m_aobasis.shells();
        combined.insert(combined.end(), m_auxbasis.shells().begin(),
                        m_auxbasis.shells().end());
        m_env = IntEnv(combined_sites, combined);
    }

    inline void clear_auxiliary_basis() {
        m_auxbasis = AOBasis();
        m_env = IntEnv(m_aobasis.atoms(), m_aobasis.shells());
    }

    inline bool have_auxiliary_basis() const noexcept {
        return m_auxbasis.nsh() > 0;
    }

    template <ShellKind kind, typename Lambda>
    void evaluate_three_center_aux(Lambda &f,
                                   int thread_id = 0) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 3);
        auto buffer = std::make_unique<double[]>(buffer_size_3e());
        IntegralResult<3> args;
        args.thread = thread_id;
        args.buffer = buffer.get();
        std::array<int, 3> shell_idx;
        const auto &first_bf_ao = m_aobasis.first_bf();
        const auto &first_bf_aux = m_auxbasis.first_bf();
        for (int auxP = 0; auxP < m_auxbasis.size(); auxP++) {
            if (auxP % nthreads != thread_id)
                continue;
            const auto &shauxP = m_auxbasis[auxP];
            args.bf[2] = first_bf_aux[auxP];
            args.shell[2] = auxP;
            for (int p = 0; p < m_aobasis.size(); p++) {
                args.bf[0] = first_bf_ao[p];
                args.shell[0] = p;
                const auto &shp = m_aobasis[p];
                const auto &plist = m_shellpairs.at(p);
                if ((shp.extent > 0.0) &&
                    (shp.origin - shauxP.origin).norm() > shp.extent) {
                    continue;
                }
                for (const int &q : plist) {
                    args.bf[1] = first_bf_ao[q];
                    args.shell[1] = q;
                    const auto &shq = m_aobasis[q];
                    shell_idx = {p, q,
                                 auxP + static_cast<int>(m_aobasis.size())};
                    if ((shq.extent > 0.0) &&
                        (shq.origin - shauxP.origin).norm() > shq.extent) {
                        continue;
                    }
                    args.dims = m_env.three_center_helper<Op::coulomb, kind>(
                        shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
                    if (args.dims[0] > -1) {
                        f(args);
                    }
                }
            }
        }
    }

    Mat one_electron_operator(Op op) const;

    Mat fock_operator(SpinorbitalKind, const MolecularOrbitals &mo,
                      const Mat &Schwarz = Mat()) const;

    Mat fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                  bool is_shell_diagonal);

    Mat coulomb(SpinorbitalKind, const MolecularOrbitals &mo,
                const Mat &Schwarz = Mat()) const;

    std::pair<Mat, Mat> coulomb_and_exchange(SpinorbitalKind,
                                             const MolecularOrbitals &mo,
                                             const Mat &Schwarz = Mat()) const;

    template <ShellKind kind = ShellKind::Cartesian>
    Mat
    point_charge_potential(const std::vector<occ::core::PointCharge> &charges) {
        const auto nbf = m_aobasis.nbf();
        ShellList dummy_shells;
        dummy_shells.reserve(charges.size());
        for (size_t i = 0; i < charges.size(); i++) {
            dummy_shells.push_back(OccShell(charges[i]));
        }
        set_auxiliary_basis(dummy_shells, true);
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nbf, nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        size_t nsh = m_aobasis.size();
        auto f = [nsh, &results](const IntegralResult<3> &args) {
            auto &result = results[args.thread];
            Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
            result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                tmp;
            if (args.shell[0] != args.shell[1]) {
                result.block(args.bf[1], args.bf[0], args.dims[1],
                             args.dims[0]) += tmp.transpose();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_three_center_aux<kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; i++) {
            results[0] += results[i];
        }
        return results[0];
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    Vec electric_potential(const MolecularOrbitals &mo, const Mat3N &points) {
        ShellList dummy_shells;
        dummy_shells.reserve(points.cols());
        for (size_t i = 0; i < points.cols(); i++) {
            dummy_shells.push_back(
                OccShell({1.0, {points(0, i), points(1, i), points(2, i)}}));
        }
        set_auxiliary_basis(dummy_shells, true);
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<Vec> results;
        results.emplace_back(Vec::Zero(points.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        const auto &D = mo.D;
        auto f = [&D, &results](const IntegralResult<3> &args) {
            auto &v = results[args.thread];
            auto scale = (args.shell[0] == args.shell[1]) ? 1 : 2;
            Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
            if constexpr (sk == SpinorbitalKind::Restricted) {
                v(args.shell[2]) +=
                    (D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                         .array() *
                     tmp.array())
                        .sum() *
                    scale;
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                const auto alpha = occ::qm::block::a(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto beta = occ::qm::block::a(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                v(args.shell[2]) +=
                    ((alpha.array() + beta.array()) * tmp.array()).sum() *
                    scale;
            } else if constexpr (sk == SpinorbitalKind::General) {
                const auto aa = occ::qm::block::aa(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto ab = occ::qm::block::ab(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto ba = occ::qm::block::ba(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto bb = occ::qm::block::bb(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                v(args.shell[2]) +=
                    ((aa.array() + ab.array() + ba.array() + bb.array()) *
                     tmp.array())
                        .sum() *
                    scale;
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_three_center_aux<kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; i++) {
            results[0] += results[i];
        }
        return 2 * results[0];
    }

    template <ShellKind kind>
    inline void compute_shellpairs(double threshold = 1e-12) {
        constexpr auto op = Op::overlap;
        const auto nsh = m_aobasis.size();
        m_shellpairs.resize(nsh);
        auto buffer = std::make_unique<double[]>(buffer_size_1e());
        for (int p = 0; p < nsh; p++) {
            auto &plist = m_shellpairs[p];
            const auto &sh1 = m_aobasis[p];
            for (int q = 0; q <= p; q++) {
                if (m_aobasis.shells_share_origin(p, q)) {
                    plist.push_back(q);
                    continue;
                }
                const auto &sh2 = m_aobasis[q];
                std::array<int, 2> idxs{p, q};
                std::array<int, 2> dims = m_env.two_center_helper<op, kind>(
                    idxs, nullptr, buffer.get(), nullptr);
                Eigen::Map<const occ::Mat> tmp(buffer.get(), dims[0], dims[1]);
                if (tmp.norm() >= threshold) {
                    plist.push_back(q);
                }
            }
        }
    }

    Vec multipole(SpinorbitalKind, int order, const MolecularOrbitals &mo,
                  const Vec3 &origin = {0, 0, 0}) const;

    template <ShellKind kind = ShellKind::Cartesian> Mat schwarz() const {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        constexpr bool use_euclidean_norm{false};
        const auto nsh = m_aobasis.size();
        const auto &first_bf = m_aobasis.first_bf();
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nsh, nsh));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        auto f = [&results](const IntegralResult<4> &args) {
            auto &result = results[args.thread];
            auto N = args.dims[0] * args.dims[1];
            Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
            double sq_norm =
                use_euclidean_norm ? tmp.norm() : tmp.lpNorm<Eigen::Infinity>();
            double norm = std::sqrt(sq_norm);
            result(args.shell[0], args.shell[1]) = norm;
            result(args.shell[1], args.shell[0]) = norm;
        };

        auto lambda = [&](int thread_id) {
            auto buffer = std::make_unique<double[]>(buffer_size_2e());
            for (int p = 0, pq = 0; p < nsh; p++) {
                int bf1 = first_bf[p];
                const auto &sh1 = m_aobasis[p];
                for (const int &q : m_shellpairs.at(p)) {
                    if (pq++ % nthreads != thread_id)
                        continue;
                    int bf2 = first_bf[q];
                    const auto &sh2 = m_aobasis[q];
                    std::array<int, 4> idxs{p, q, p, q};
                    IntegralResult<4> args{
                        thread_id,
                        idxs,
                        {bf1, bf2, bf1, bf2},
                        m_env.four_center_helper<op, kind>(
                            idxs, nullptr, buffer.get(), nullptr),
                        buffer.get()};
                    if (args.dims[0] > -1)
                        f(args);
                }
            }
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        return results[0];
    }

    inline bool is_spherical() const {
        return m_aobasis.kind() == OccShell::Kind::Spherical;
    }

  private:
    double m_precision{1e-12};
    AOBasis m_aobasis, m_auxbasis;
    ShellPairList m_shellpairs;
    // TODO remove mutable
    mutable IntEnv m_env;

    inline size_t buffer_size_1e(const Op op = Op::overlap) const {
        auto bufsize = m_aobasis.max_shell_size() * m_aobasis.max_shell_size();
        switch (op) {
        case Op::dipole:
            bufsize *= occ::core::num_unique_multipole_components(1);
            break;
        case Op::quadrupole:
            bufsize *= occ::core::num_unique_multipole_components(2);
            break;
        case Op::octapole:
            bufsize *= occ::core::num_unique_multipole_components(3);
            break;
        case Op::hexadecapole:
            bufsize *= occ::core::num_unique_multipole_components(4);
            break;
        default:
            break;
        }
        return bufsize;
    }

    inline size_t buffer_size_3e() const {
        return m_auxbasis.max_shell_size() * buffer_size_1e();
    }

    inline size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};

} // namespace occ::qm
