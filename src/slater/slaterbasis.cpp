#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/util.h>
#include <occ/slater/slaterbasis.h>

namespace occ::slater {

using occ::IVec;
using occ::Mat;
using occ::Vec;
constexpr double inv_pi_4 = 1.0 / (4 * M_PI);

Shell::Shell() : m_occupation(1), m_n(1), m_z(1), m_c(1, 1) {
    m_occupation.setConstant(1);
    m_n.setConstant(1);
    m_z.setConstant(1.24);
    m_c.setConstant(1.0);
    m_n1 = Vec::Zero(1);
}

Shell::Shell(const IVec &occ, const IVec &n, const Vec &z, const Mat &c)
    : m_occupation(occ), m_n(n), m_z(z), m_c(c) {
    m_n1 = m_n.array().cast<double>() - 1.0;
}

Shell::Shell(const std::vector<int> &occ, const std::vector<int> &n,
             const std::vector<double> &z,
             const std::vector<std::vector<double>> &c)
    : m_occupation(occ.size()), m_n(n.size()), m_z(z.size()) {
    size_t ncols = c.size();
    size_t nrows = c[0].size();
    // assume all cols are the same size;
    m_c = Mat(nrows, ncols);
    size_t i = 0;
    for (const auto &col : c) {
        for (size_t j = 0; j < col.size(); j++) {
            if (j >= nrows)
                break;
            m_c(j, i) = col[j];
        }
        i++;
    }
    i = 0;
    for (const auto &o : occ) {
        m_occupation(i) = o;
        i++;
    }
    i = 0;
    for (const auto &num : n) {
        m_n(i) = num;
        i++;
    }
    i = 0;
    for (const auto &zz : z) {
        m_z(i) = zz;
        i++;
    }
    m_n1 = m_n.array().cast<double>() - 1.0;
}

size_t Shell::n_prim() const { return m_z.rows(); }

size_t Shell::n_orb() const { return m_occupation.rows(); }

double Shell::rho(double r) const {
    Vec e = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array() * r);
    double result = 0;
    for (size_t i = 0; i < n_orb(); i++) {
        double orb = e.dot(m_c.col(i));
        result += m_occupation(i) * orb * orb;
    }

    return result * inv_pi_4;
}

double Shell::grad_rho(double r) const {
    double result = 0.0;
    Vec g = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array() * r);
    Vec gprime(g.array() * (m_n1.array() / r - m_z.array()));

    for (size_t i = 0; i < n_orb(); i++) {
        auto ci = m_c.row(i);
        auto orb = g.dot(ci);
        auto dorb = gprime.dot(ci);
        result += 2 * m_occupation(i) * orb * dorb;
    }
    return result * inv_pi_4;
}

Vec Shell::rho(const Vec &r) const {
    Vec result = Vec::Zero(r.rows());
    rho(r, result);
    return result;
}

void Shell::rho(const Vec &r, Vec &result) const {
    Vec e(n_prim());
    for (size_t p = 0; p < r.rows(); p++) {
        double rp = r(p);
        e = Eigen::pow(rp, m_n1.array()) * Eigen::exp(-m_z.array() * rp);
        for (size_t i = 0; i < n_orb(); i++) {
            double orb = e.dot(m_c.col(i));
            result(p) += m_occupation(i) * orb * orb * inv_pi_4;
        }
    }
}

Vec Shell::grad_rho(const Vec &r) const {
    Vec result = Vec::Zero(r.rows());
    grad_rho(r, result);
    return result;
}

void Shell::grad_rho(const Vec &r, Vec &result) const {
    Vec g(n_prim()), gprime(n_prim());
    for (size_t i = 0; i < r.rows(); i++) {
        double rp = r(i);
        g = Eigen::pow(rp, m_n1.array()) * Eigen::exp(-m_z.array() * rp);
        gprime = g.array() * (m_n1.array() / rp - m_z.array());
        for (size_t j = 0; j < n_orb(); j++) {
            double orb = g.dot(m_c.col(j));
            double dorb = gprime.dot(m_c.col(j));
            result(i) += 2 * m_occupation(j) * orb * dorb * inv_pi_4;
        }
    }
}

void Shell::renormalize() {
    using occ::util::factorial;
    for (size_t i = 0; i < n_prim(); i++) {
        int n2 = 2 * m_n(i);
        double factor =
            sqrt(2 * m_z(i) / factorial(n2)) * (pow(2 * m_z(i), m_n(i)));
        m_c.row(i).array() /= factor;
    }
}

void Shell::unnormalize() {
    using occ::util::factorial;
    for (size_t i = 0; i < n_prim(); i++) {
        int n2 = 2 * m_n(i);
        double z2 = 2 * m_z(i);
        double factor = sqrt(z2 / factorial(n2)) * pow(z2, m_n(i));
        m_c.row(i).array() *= factor;
    }
}

Basis::Basis(const std::vector<Shell> &shells) : m_shells(shells) {
    unnormalize();
}

double Basis::rho(double r) const {
    double result = 0.0;
    for (const auto &s : m_shells) {
        result += s.rho(r);
    }
    return result;
}

double Basis::grad_rho(double r) const {
    double result = 0.0;
    for (const auto &s : m_shells) {
        result += s.grad_rho(r);
    }
    return result;
}

Vec Basis::rho(const Vec &r) const {
    Vec result = Vec::Zero(r.rows());
    for (const auto &s : m_shells) {
        s.rho(r, result);
    }
    return result;
}

Vec Basis::grad_rho(const Vec &r) const {
    Vec result = Vec::Zero(r.rows());
    for (const auto &s : m_shells) {
        s.grad_rho(r, result);
    }
    return result;
}

void Basis::renormalize() {
    for (auto &sh : m_shells) {
        sh.renormalize();
    }
}

void Basis::unnormalize() {
    for (auto &sh : m_shells) {
        sh.unnormalize();
    }
}

SlaterBasisSetMap
load_slaterbasis(const std::string &name) {
    namespace fs = std::filesystem;
    fs::path path;
    const char *basis_path_env = getenv("OCC_DATA_PATH");
    if (basis_path_env) {
        // TODO check
        path = basis_path_env;
        path /= "basis";
    } else {
        path = ".";
    }
    path /= name;
    path.replace_extension("json");
    SlaterBasisSetMap basis_set;

    if (!fs::exists(path)) {
        throw std::runtime_error("Could not locate slater basis file " +
                                 path.string());
    }
    std::ifstream in(path.c_str());

    if (!in.good()) {
        throw std::runtime_error("Could not read slater basis file " +
                                 path.string());
    }

    auto basis_json = nlohmann::json::parse(in);
    for (const auto &basis : basis_json.items()) {
        std::string k = basis.key();
        std::vector<Shell> shells;
        const auto &j = basis.value();
        for (const auto &shell : j) {
            const auto &occ_ref = shell["occ"];
            const auto &n_ref = shell["n"];
            const auto &z_ref = shell["z"];
            const auto &c_ref = shell["c"];
            size_t nocc = occ_ref.size();
            size_t nz = n_ref.size();
            IVec occupation(nocc);
            IVec n(nz);
            Vec z(z_ref.size());
            Mat c = Mat::Zero(nz, nocc);
            for (size_t row = 0; row < nocc; row++) {
                occupation(row) = occ_ref[row];
            }

            for (size_t row = 0; row < nz; row++) {
                n(row) = n_ref[row];
                z(row) = z_ref[row];
            }

            for (size_t oi = 0; oi < nocc; oi++) {
                for (size_t zi = 0; zi < nz; zi++) {
                    c(zi, oi) = c_ref[oi][zi];
                }
            }
            shells.push_back(Shell(occupation, n, z, c));
        }
        basis_set[k] = Basis(shells);
    }
    return basis_set;
}

std::vector<Basis>
slaterbasis_for_atoms(const std::vector<occ::core::Atom> &atoms,
                      const std::string &basis_name,
                      const std::vector<int> &oxidation_states) {
    auto slaterbasis_data = occ::slater::load_slaterbasis(basis_name);
    std::vector<Basis> result;
    size_t atom_idx = 0;
    for (const auto &atom : atoms) {
        auto el = occ::core::Element(atom.atomic_number);
        std::string sym = el.symbol();

        if (oxidation_states.size() > atom_idx) {
            char state = oxidation_states[atom_idx] > 0 ? '+' : '-';
            sym = fmt::format("{}{}", sym, state);
        } else if (sym == "H") {
            sym = "H_normal";
        }

        if (!slaterbasis_data.contains(sym)) {
            throw std::runtime_error(
                fmt::format("no such slater basis for {} found", sym));
        }
        result.push_back(slaterbasis_data[sym]);
        atom_idx++;
    }
    return result;
}

} // namespace occ::slater
