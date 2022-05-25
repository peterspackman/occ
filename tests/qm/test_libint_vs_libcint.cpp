#include "catch.hpp"
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/parallel.h>
#include <occ/io/json_basis.h>
#include <occ/qm/basisset.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/fock.h>
#include <occ/qm/hf.h>
#include <vector>

using occ::qm::cint::Operator;
using Kind = occ::qm::OccShell::Kind;
using occ::Mat;

struct IntegralEngine {
    struct TwoCenterArgs {
        int thread{0};
        std::array<int, 2> shell{0};
        std::array<int, 2> bf{0};
        std::array<int, 2> dims{0};
        const double *buffer{nullptr};
    };

    struct FourCenterArgs {
        int thread{0};
        std::array<int, 4> shell{0};
        std::array<int, 4> bf{0};
        std::array<int, 4> dims{0};
        const double *buffer{nullptr};
    };

    using ShellPairList = std::vector<std::vector<size_t>>;
    IntegralEngine(const std::vector<occ::core::Atom> &at,
                   const std::vector<occ::qm::OccShell> &sh)
        : atoms(at), shells(sh), env(at, sh) {
        for (const auto &shell : shells) {
            first_bf.push_back(nbf);
            nbf += shell.size();
            nsh += 1;
            max_shell_size = std::max(max_shell_size, shell.size());
        }
        compute_shellpairs();
    }

    template <Operator op, Kind kind, typename Lambda>
    void evaluate_two_center(Lambda &f, int thread_id = 0) const {
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<double> buffer(buffer_size_1e());
        for (int p = 0, pq = 0; p < nsh; p++) {
            int bf1 = first_bf[p];
            const auto &sh1 = shells[p];
            for (const int &q : shellpairs.at(p)) {
                if (pq++ % nthreads != thread_id)
                    continue;
                int bf2 = first_bf[q];
                const auto &sh2 = shells[q];
                std::array<int, 2> idxs{p, q};
                TwoCenterArgs args{thread_id,
                                   idxs,
                                   {bf1, bf2},
                                   env.two_center_helper<op, kind>(
                                       idxs, nullptr, buffer.data(), nullptr),
                                   buffer.data()};
                f(args);
            }
        }
    }

    template <Operator op, Kind kind, typename Lambda>
    void evaluate_four_center(Lambda &f, int thread_id = 0) const {
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<double> buffer(buffer_size_2e());
        std::array<int, 4> shell_idx;
        std::array<int, 4> bf;
        for (int p = 0, pqrs = 0; p < nsh; p++) {
            const auto &sh1 = shells[p];
            bf[0] = first_bf[p];
            const auto &plist = shellpairs.at(p);
            for (const int &q : plist) {
                bf[1] = first_bf[q];
                const auto &sh2 = shells[q];
                for (int r = 0; r <= p; r++) {
                    const auto &sh3 = shells[r];
                    bf[2] = first_bf[r];
                    const auto s_max = (p == r) ? q : r;
                    for (const int s : shellpairs.at(r)) {
                        if (s > s_max)
                            break;
                        if (pqrs++ % nthreads != thread_id)
                            continue;
                        bf[3] = first_bf[s];
                        const auto &sh4 = shells[s];
                        shell_idx = {p, q, r, s};

                        FourCenterArgs args{
                            thread_id, shell_idx, bf,
                            env.four_center_helper<Operator::coulomb, kind>(
                                shell_idx, nullptr, buffer.data(), nullptr),
                            buffer.data()};
                        f(args);
                    }
                }
            }
        }
    }

    template <Operator op, Kind kind = Kind::Cartesian>
    Mat one_electron_operator() const {
        auto nthreads = occ::parallel::get_num_threads();
        Mat result = Mat::Zero(nbf, nbf);
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nbf, nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        auto f = [&results](const TwoCenterArgs &args) {
            auto &result = results[args.thread];
            Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0],
                                           args.dims[1]);
            result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) =
                tmp;
            if (args.shell[0] != args.shell[1]) {
                result.block(args.bf[1], args.bf[0], args.dims[1],
                             args.dims[0]) = tmp.transpose();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_two_center<op, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }
        return results[0];
    }

    template <Kind kind = Kind::Cartesian>
    Mat fock_operator(const Mat &D) const {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Operator op = Operator::coulomb;
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nbf, nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        auto f = [&D, &results](const FourCenterArgs &args) {
            auto &g = results[args.thread];
            auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
            auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                    ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                    : 2;
            auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
            auto scale = pq_degree * rs_degree * pr_qs_degree;

            for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
                const auto bf3 = f3 + args.bf[3];
                for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
                    const auto bf2 = f2 + args.bf[2];
                    for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
                        const auto bf1 = f1 + args.bf[1];
                        for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
                            const auto bf0 = f0 + args.bf[0];
                            const auto value = args.buffer[f0123] * scale;
                            // J
                            g(bf0, bf1) += D(bf2, bf3) * value;
                            g(bf2, bf3) += D(bf0, bf1) * value;
                            // K
                            g(bf0, bf2) -= 0.25 * D(bf1, bf3) * value;
                            g(bf1, bf3) -= 0.25 * D(bf0, bf2) * value;
                            g(bf0, bf3) -= 0.25 * D(bf1, bf2) * value;
                            g(bf1, bf2) -= 0.25 * D(bf0, bf3) * value;
                        }
                    }
                }
            }
        };
        auto lambda = [&](int thread_id) {
            evaluate_four_center<Operator::coulomb, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        return 0.5 * (results[0] + results[0].transpose());
    }

    void compute_shellpairs(double threshold = 1e-12) {
        constexpr auto op = Operator::overlap;
        shellpairs.resize(nsh);
        std::vector<double> buffer(buffer_size_1e());
        for (int p = 0; p < nsh; p++) {
            auto &plist = shellpairs[p];
            const auto &sh1 = shells[p];
            for (int q = 0; q <= p; q++) {
                if (shells[p].origin == shells[q].origin) {
                    plist.push_back(q);
                    continue;
                }
                const auto &sh2 = shells[q];
                std::array<int, 2> idxs{p, q};
                std::array<int, 2> dims =
                    env.two_center_helper<op, Kind::Spherical>(
                        idxs, nullptr, buffer.data(), nullptr);
                Eigen::Map<const occ::Mat> tmp(buffer.data(), dims[0], dims[1]);
                if (tmp.norm() >= threshold) {
                    plist.push_back(q);
                }
            }
        }
    }

    template <Kind kind = Kind::Cartesian> Mat schwarz() const {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Operator op = Operator::coulomb;
        constexpr bool use_euclidean_norm{false};
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nsh, nsh));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        auto f = [&results](const FourCenterArgs &args) {
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
            std::vector<double> buffer(buffer_size_2e());
            for (int p = 0, pq = 0; p < nsh; p++) {
                int bf1 = first_bf[p];
                const auto &sh1 = shells[p];
                for (const int &q : shellpairs.at(p)) {
                    if (pq++ % nthreads != thread_id)
                        continue;
                    int bf2 = first_bf[q];
                    const auto &sh2 = shells[q];
                    std::array<int, 4> idxs{p, q, p, q};
                    FourCenterArgs args{
                        thread_id,
                        idxs,
                        {bf1, bf2, bf1, bf2},
                        env.four_center_helper<op, kind>(
                            idxs, nullptr, buffer.data(), nullptr),
                        buffer.data()};
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

    size_t nbf{0};
    size_t nsh{0};
    size_t max_shell_size{0};
    std::vector<occ::core::Atom> atoms;
    std::vector<occ::qm::OccShell> shells;
    std::vector<size_t> first_bf;
    ShellPairList shellpairs;
    mutable occ::qm::cint::IntegralEnvironment env;

    size_t buffer_size_1e() const { return max_shell_size * max_shell_size; }
    size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};

IntegralEngine
read_atomic_orbital_basis(const std::vector<occ::core::Atom> &atoms,
                          std::string name, bool cartesian = true) {
    occ::util::to_lower(name); // make name lowercase
    occ::io::JsonBasisReader reader(name);
    std::vector<occ::qm::OccShell> shells;
    for (const auto &atom : atoms) {
        const auto &element_basis = reader.element_basis(atom.atomic_number);
        for (const auto &electron_shell : element_basis.electron_shells) {
            for (size_t n = 0; n < electron_shell.angular_momentum.size();
                 n++) {
                shells.push_back(occ::qm::OccShell(
                    electron_shell.angular_momentum[n],
                    electron_shell.exponents, {electron_shell.coefficients[n]},
                    {atom.x, atom.y, atom.z}));
            }
        }
    }

    // normalize the shells
    for (auto &shell : shells) {
        if (!cartesian) {
            shell.kind = Kind::Spherical;
        }
        shell.incorporate_shell_norm();
    }
    return IntegralEngine(atoms, shells);
}

std::vector<occ::qm::OccShell> libint_to_occ(const occ::qm::BasisSet &basis) {
    std::vector<occ::qm::OccShell> result;
    result.reserve(basis.size());
    for (const auto &sh : basis) {
        result.emplace_back(occ::qm::OccShell(sh));
        result.back().incorporate_shell_norm();
        fmt::print("coeff_normalized(0, 0) difference: {}\n",
                   result.back().coeff_normalized(0, 0) -
                       sh.coeff_normalized(0, 0));
    }
    return result;
}

TEST_CASE("Water nuclear attraction", "[cint]") {
    using occ::qm::OccShell;
    libint2::Shell::do_enforce_unit_normalization(true);
    if (!libint2::initialized())
        libint2::initialize();
    occ::parallel::set_num_threads(1);

    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::BasisSet basis("def2-tzvp", atoms);

    constexpr Kind kind = Kind::Cartesian;
    basis.set_pure(kind == Kind::Spherical);

    occ::hf::HartreeFock hf(atoms, basis);
    occ::ints::FockBuilder fock(basis.max_nprim(), basis.max_l());

    IntegralEngine basis2(atoms, libint_to_occ(basis));

    auto o1 = hf.compute_overlap_matrix();
    auto o2 = basis2.one_electron_operator<Operator::overlap, kind>();
    Eigen::Index i, j;
    fmt::print("Overlap max err: {}\n", (o2 - o1).cwiseAbs().maxCoeff(&i, &j));

    auto n1 = hf.compute_nuclear_attraction_matrix();
    auto n2 = basis2.one_electron_operator<Operator::nuclear, kind>();
    fmt::print("Nuclear max err: {}\n", (n2 - n1).cwiseAbs().maxCoeff());

    auto k1 = hf.compute_kinetic_matrix();
    auto k2 = basis2.one_electron_operator<Operator::kinetic, kind>();
    fmt::print("Kinetic max err: {}\n", (k2 - k1).cwiseAbs().maxCoeff());

    occ::Mat D = occ::Mat::Random(basis2.nbf, basis2.nbf);
    occ::qm::MolecularOrbitals mo;
    mo.D = D;
    occ::Mat f1 = fock.compute_fock<occ::qm::SpinorbitalKind::Restricted>(
        basis, hf.shellpair_list(), hf.shellpair_data(), mo);
    occ::Mat f2 = basis2.fock_operator<kind>(D);
    fmt::print("Fock max err: {}\n", (f2 - f1).cwiseAbs().maxCoeff());

    Mat schw1 = hf.compute_schwarz_ints();
    Mat schw2 = basis2.schwarz<kind>();
    fmt::print("Schwarz max err: {}\n",
               (schw2 - schw1).cwiseAbs().maxCoeff(&i, &j));
}
