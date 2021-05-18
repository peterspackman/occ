#include <occ/slater/slaterbasis.h>
#include <occ/core/util.h>
#include <fmt/core.h>

namespace occ::slater {

using occ::IVec;
using occ::Vec;
using occ::Mat;
constexpr double inv_pi_4 = 1.0 / (4 * M_PI);

Shell::Shell() :
    m_occupation(1),
    m_n(1),
    m_z(1),
    m_c(1, 1)
{
    m_occupation.setConstant(1);
    m_n.setConstant(1);
    m_z.setConstant(1.24);
    m_c.setConstant(1.0);
    m_n1 = Vec::Zero(1);
}

Shell::Shell(
        const IVec& occ,
        const IVec& n,
        const Vec& z,
        const Mat& c) :
    m_occupation(occ), m_n(n), m_z(z), m_c(c)
{
    m_n1 = m_n.array().cast<double>() - 1.0;
}

Shell::Shell(const std::vector<int> &occ, const std::vector<int> &n,
             const std::vector<double> &z, const std::vector<std::vector<double>> &c) :
    m_occupation(occ.size()), m_n(n.size()), m_z(z.size())
{
    size_t ncols = c.size();
    size_t nrows = c[0].size();
    // assume all cols are the same size;
    m_c = Mat(nrows, ncols);
    size_t i = 0;
    for(const auto& col: c) {
        for(size_t j = 0; j < col.size(); j++) {
            if(j >= nrows) break;
            m_c(j, i) = col[j];
        }
        i++;
    }
    i = 0;
    for(const auto& o : occ) {
        m_occupation(i) = o;
        i++;
    }
    i = 0;
    for(const auto& num : n) {
        m_n(i) = num;
        i++;
    }
    i = 0;
    for(const auto& zz : z) {
        m_z(i) = zz;
        i++;
    }
    m_n1 = m_n.array().cast<double>() - 1.0;
}

size_t Shell::n_prim() const {
    return m_z.rows();
}

size_t Shell::n_orb() const {
    return m_occupation.rows();
}

double Shell::rho(double r) const {
    Vec e = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array()*r);
    double result = 0;
    for(size_t i = 0; i < n_orb(); i++) {
        double orb = e.dot(m_c.col(i));
        result += m_occupation(i) * orb * orb;
    }

    return result * inv_pi_4;
}

double Shell::grad_rho(double r) const {
    double result = 0.0;
    Vec g = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array()*r);
    Vec gprime(g.array() * (m_n1.array() / r - m_z.array()));

    for(size_t i = 0; i < n_orb(); i++) {
        auto ci = m_c.row(i);
        auto orb =  g.dot(ci);
        auto dorb = gprime.dot(ci);
        result += 2 * m_occupation(i) * orb * dorb;
    }
    return result * inv_pi_4;
}


Vec Shell::rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    rho(r, result);
    return result;
}

void Shell::rho(const Vec& r, Vec& result) const
{
    Vec e(n_prim());
    for(size_t p = 0; p < r.rows(); p++) {
        double rp = r(p);
        e = Eigen::pow(rp, m_n1.array()) * Eigen::exp(-m_z.array() * rp);
        for(size_t i = 0; i < n_orb(); i++) {
            double orb = e.dot(m_c.col(i));
            result(p) += m_occupation(i) * orb * orb * inv_pi_4;
        }
    }
}


Vec Shell::grad_rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    grad_rho(r, result);
    return result;
}

void Shell::grad_rho(const Vec& r, Vec& result) const
{
    Vec g(n_prim()), gprime(n_prim());
    for(size_t i = 0; i < r.rows(); i++)
    {
        double rp = r(i);
        g = Eigen::pow(rp, m_n1.array()) * Eigen::exp(-m_z.array() * rp);
        gprime = g.array() * (m_n1.array()/rp - m_z.array());
        for(size_t j = 0; j < n_orb(); j++)
        {
            double orb = g.dot(m_c.col(j));
            double dorb = gprime.dot(m_c.col(j));
            result(i) += 2 * m_occupation(j) * orb * dorb * inv_pi_4;
        }
    }
}

void Shell::renormalize()
{
    using occ::util::factorial;
    for(size_t i = 0; i < n_prim(); i++)
    {
        int n2 = 2 * m_n(i);
        double factor = sqrt(2 * m_z(i) / factorial(n2)) * (pow(2 * m_z(i), m_n(i)));
        m_c.row(i).array() /= factor;
    }
}

void Shell::unnormalize()
{
    using occ::util::factorial;
    for(size_t i = 0; i < n_prim(); i++)
    {
        int n2 = 2 * m_n(i);
        double z2 = 2 * m_z(i);
        double factor = sqrt(z2 / factorial(n2)) * pow(z2, m_n(i));
        m_c.row(i).array() *= factor;
    }
}

Basis::Basis(const std::vector<Shell>& shells) :
    m_shells(shells)
{
    unnormalize();
}

double Basis::rho(double r) const {
    double result = 0.0;
    for (const auto& s: m_shells) {
        result += s.rho(r);
    }
    return result;
}

double Basis::grad_rho(double r) const {
    double result = 0.0;
    for (const auto& s: m_shells) {
        result += s.grad_rho(r);
    }
    return result;
}

Vec Basis::rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    for (const auto& s: m_shells) {
        s.rho(r, result);
    }
    return result;
}

Vec Basis::grad_rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    for (const auto& s: m_shells) {
        s.grad_rho(r, result);
    }
    return result;
}

void Basis::renormalize()
{
    for(auto& sh: m_shells)
    {
        sh.renormalize();
    }
}

void Basis::unnormalize()
{
    for(auto& sh: m_shells)
    {
        sh.unnormalize();
    }
}

}
