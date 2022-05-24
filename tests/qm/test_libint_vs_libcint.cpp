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

struct AOBasis {
    AOBasis(const std::vector<occ::core::Atom> &at,
            const std::vector<occ::qm::OccShell> &sh)
        : atoms(at), shells(sh), env(at, sh) {
        for (const auto &shell : shells) {
            first_bf.push_back(nbf);
            nbf += shell.size();
            nsh += 1;
            max_shell_size = std::max(max_shell_size, shell.size());
        }
    }
    size_t nbf{0};
    size_t nsh{0};
    size_t max_shell_size{0};
    std::vector<occ::core::Atom> atoms;
    std::vector<occ::qm::OccShell> shells;
    std::vector<size_t> first_bf;
    mutable occ::qm::cint::IntegralEnvironment env;

    size_t buffer_size_1e() const { return max_shell_size * max_shell_size; }
    size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};

AOBasis read_atomic_orbital_basis(const std::vector<occ::core::Atom> &atoms,
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
    return AOBasis(atoms, shells);
}

template <Operator op, Kind kind = Kind::Cartesian>
occ::Mat compute_one_electron(const AOBasis &basis) {
    occ::Mat result = occ::Mat::Zero(basis.nbf, basis.nbf);
    std::vector<double> buffer(basis.buffer_size_1e());
    int bf1 = 0;
    for (int p = 0; p < basis.nsh; p++) {
        const auto &sh1 = basis.shells[p];
        int bf2 = bf1;
        for (int q = p; q < basis.nsh; q++) {
            const auto &sh2 = basis.shells[q];
            std::array<int, 2> idxs{p, q};
            std::array<int, 2> dims = basis.env.one_electron_helper<op, kind>(
                idxs, nullptr, buffer.data(), nullptr);
            Eigen::Map<const occ::Mat> tmp(buffer.data(), dims[0], dims[1]);
            result.block(bf1, bf2, dims[0], dims[1]) = tmp;
            if (p != q) {
                result.block(bf2, bf1, dims[1], dims[0]) = tmp.transpose();
            }
            bf2 += sh2.size();
        }
        bf1 += sh1.size();
    }
    return result;
}

using ShellPairList = std::vector<std::vector<size_t>>;

template <Kind kind = Kind::Cartesian>
ShellPairList compute_shellpairs(const AOBasis &basis,
                                 double threshold = 1e-12) {
    constexpr auto op = Operator::overlap;
    ShellPairList result(basis.nsh);
    std::vector<double> buffer(basis.buffer_size_1e());
    for (int p = 0; p < basis.nsh; p++) {
        auto &plist = result[p];
        const auto &sh1 = basis.shells[p];
        for (int q = 0; q <= p; q++) {
            if (basis.shells[p].origin == basis.shells[q].origin) {
                plist.push_back(q);
                continue;
            }
            const auto &sh2 = basis.shells[q];
            std::array<int, 2> idxs{p, q};
            std::array<int, 2> dims = basis.env.one_electron_helper<op, kind>(
                idxs, nullptr, buffer.data(), nullptr);
            Eigen::Map<const occ::Mat> tmp(buffer.data(), dims[0], dims[1]);
            if (tmp.norm() >= threshold) {
                plist.push_back(q);
            }
        }
    }
    return result;
}

template <Kind kind = Kind::Cartesian>
occ::Mat compute_schwarz_ints(const AOBasis &basis) {
    occ::Mat result = occ::Mat::Zero(basis.nbf, basis.nbf);
    constexpr auto op = Operator::coulomb;
    constexpr bool use_euclidean_norm{false};
    std::vector<double> buffer(basis.buffer_size_2e());
    for (int p = 0; p < basis.nsh; p++) {
        const auto &sh1 = basis.shells[p];
        for (int q = 0; q <= p; q++) {
            const auto &sh2 = basis.shells[q];
            std::array<int, 4> idxs{p, q, p, q};
            std::array<int, 4> dims = basis.env.two_electron_helper<op, kind>(
                idxs, nullptr, buffer.data(), nullptr);
            Eigen::Map<const occ::Mat> tmp(buffer.data(), dims[0], dims[1]);
            double sq_norm =
                use_euclidean_norm ? tmp.norm() : tmp.lpNorm<Eigen::Infinity>();
            result(p, q) = std::sqrt(sq_norm);
            result(q, p) = result(p, q);
        }
    }
    return result;
}

template <Kind kind = Kind::Cartesian>
occ::Mat compute_fock(const occ::Mat &D, const AOBasis &basis,
                      const ShellPairList &shellpairs) {
    occ::Mat result = occ::Mat::Zero(basis.nbf, basis.nbf);
    std::vector<double> buffer(basis.buffer_size_2e());
    std::array<int, 4> shell_idx;
    for (int p = 0; p < basis.nsh; p++) {
        const auto &sh1 = basis.shells[p];
        int bf1_first = basis.first_bf[p];
        const auto &plist = shellpairs.at(p);
        for (const int &q : plist) {
            int bf2_first = basis.first_bf[q];
            const auto &sh2 = basis.shells[q];

            for (int r = 0; r <= p; r++) {
                const auto &sh3 = basis.shells[r];
                int bf3_first = basis.first_bf[r];
                const auto s_max = (p == r) ? q : r;
                for (const int s : shellpairs.at(r)) {
                    if (s > s_max)
                        break;
                    int bf4_first = basis.first_bf[s];
                    const auto &sh4 = basis.shells[s];
                    shell_idx = {p, q, r, s};
                    std::array<int, 4> dims =
                        basis.env.two_electron_helper<Operator::coulomb, kind>(
                            shell_idx, nullptr, buffer.data(), nullptr);

                    auto pq_degree = (p == q) ? 1 : 2;
                    auto pr_qs_degree = (p == r) ? (q == s ? 1 : 2) : 2;
                    auto rs_degree = (r == s) ? 1 : 2;
                    auto scale = pq_degree * rs_degree * pr_qs_degree;

                    for (auto f4 = 0, f1234 = 0; f4 != dims[3]; ++f4) {
                        const auto bf4 = f4 + bf4_first;
                        for (auto f3 = 0; f3 != dims[2]; ++f3) {
                            const auto bf3 = f3 + bf3_first;
                            for (auto f2 = 0; f2 != dims[1]; ++f2) {
                                const auto bf2 = f2 + bf2_first;
                                for (auto f1 = 0; f1 != dims[0];
                                     ++f1, ++f1234) {
                                    const auto bf1 = f1 + bf1_first;
                                    const auto value = buffer[f1234] * scale;
                                    result(bf1, bf2) += D(bf3, bf4) * value;
                                    result(bf3, bf4) += D(bf1, bf2) * value;
                                    result(bf1, bf3) -=
                                        0.25 * D(bf2, bf4) * value;
                                    result(bf2, bf4) -=
                                        0.25 * D(bf1, bf3) * value;
                                    result(bf1, bf4) -=
                                        0.25 * D(bf2, bf3) * value;
                                    result(bf2, bf3) -=
                                        0.25 * D(bf1, bf4) * value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0.5 * (result + result.transpose());
}

template <typename T, Operator op, Kind kind = Kind::Cartesian>
void three_center_integrals(const AOBasis &bs, const AOBasis &dfbs,
                            const occ::Mat &D, const ShellPairList &splist,
                            const occ::Mat &Schwarz = occ::Mat()) {
    const auto nshells = bs.nsh;
    const auto nshells_df = dfbs.nsh;
    const bool do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;

    std::vector<double> buffer(bs.buffer_size_2e());
    for (int P = 0; P < nshells_df; P++) {
        auto bf1_first = dfbs.first_bf[P];
        const auto &sh1 = dfbs.shells[P];
        for (auto q = 0; q != nshells; q++) {
            auto bf2_first = bs.shells[q];
            const auto &sh1 = bs.shells[q];
            for (const int &r : splist.at(q)) {
                auto bf3_first = bs.shells[q];
                std::array<int, 3> idxs{P, q, r};
                std::array<int, 3> dims =
                    bs.env.three_center_two_electron_helper<op, kind>(
                        idxs, nullptr, buffer.data(), nullptr);
            }
        }
    }
}

TEST_CASE("Water nuclear attraction", "[cint]") {
    using occ::qm::OccShell;
    libint2::Shell::do_enforce_unit_normalization(true);
    if (!libint2::initialized())
        libint2::initialize();

    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::BasisSet basis("def2-tzvp", atoms);
    constexpr Kind kind = Kind::Cartesian;
    basis.set_pure(kind == Kind::Spherical);

    occ::hf::HartreeFock hf(atoms, basis);
    occ::ints::FockBuilder fock(basis.max_nprim(), basis.max_l());

    AOBasis basis2 = read_atomic_orbital_basis(atoms, "def2-tzvp.json",
                                               kind == Kind::Cartesian);
    auto o1 = hf.compute_overlap_matrix();
    auto o2 = compute_one_electron<Operator::overlap, kind>(basis2);
    Eigen::Index i, j;
    fmt::print("Overlap max err: {}\n", (o2 - o1).cwiseAbs().maxCoeff(&i, &j));
    fmt::print("@ ({} {})\n", i, j);
    std::cout << "libint\n" << o1.block(20, 20, 5, 5) << '\n';
    std::cout << "cint\n" << o2.block(20, 20, 5, 5) << '\n';

    auto n1 = hf.compute_nuclear_attraction_matrix();
    auto n2 = compute_one_electron<Operator::nuclear, kind>(basis2);
    fmt::print("Nuclear max err: {}\n", (n2 - n1).cwiseAbs().maxCoeff());

    auto k1 = hf.compute_kinetic_matrix();
    auto k2 = compute_one_electron<Operator::kinetic, kind>(basis2);
    fmt::print("Kinetic max err: {}\n", (k2 - k1).cwiseAbs().maxCoeff());

    occ::Mat D = occ::Mat::Random(basis2.nbf, basis2.nbf);
    occ::qm::MolecularOrbitals mo;
    mo.D = D;
    occ::Mat f1 = fock.compute_fock<occ::qm::SpinorbitalKind::Restricted>(
        basis, hf.shellpair_list(), hf.shellpair_data(), mo);
    fmt::print("Computed fock libint\n");
    ShellPairList splist = compute_shellpairs<kind>(basis2);
    fmt::print("Computed shellpairs cint\n");
    occ::Mat f2 = compute_fock<kind>(D, basis2, splist);
    fmt::print("Fock max err: {}\n", (f2 - f1).cwiseAbs().maxCoeff());
}
