#include "slaterbasis.h"
#include <fmt/core.h>

namespace tonto::slater {

using tonto::IVec;
using tonto::Vec;
using tonto::MatRM;

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
        const MatRM& c) :
    m_occupation(occ), m_n(n), m_z(z), m_c(c)
{
    m_n1 = m_n.array().cast<double>() - 1.0;
}

Shell::Shell(const std::vector<int> &occ, const std::vector<int> &n,
             const std::vector<double> &z, const std::vector<std::vector<double>> &c) :
    m_occupation(occ.size()), m_n(n.size()), m_z(z.size())
{
    size_t nrows = c.size();
    size_t ncols = c[0].size();
    // assume all cols are the same size;
    m_c = MatRM(nrows, ncols);
    size_t i = 0;
    for(const auto& row: c) {
        for(size_t j = 0; j < row.size(); j++) {
            if(j >= ncols) break;
            m_c(i, j) = row[j];
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
    Vec g = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array()*r);
    double result = 0;
    for(size_t i = 0; i < n_orb(); i++) {
        double orb = (m_c.row(i).array() * g.array()).array().sum();
        result += m_occupation(i) * orb * orb;
    }

    return result * (1/(4 * M_PI));
}

double Shell::grad_rho(double r) const {
    double result = 0.0;
    double rinv = 1/r;
    Vec g = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array()*r);
    Vec gprime(g.array() * ((m_n1.array() * rinv) - m_z.array()));

    for(size_t i = 0; i < n_orb(); i++) {
        auto ci = m_c.row(i);
        auto orb =  g.dot(ci);
        auto dorb = gprime.dot(ci);
        result += 2 * m_occupation(i) * orb * dorb;
    }
    return result * (1/(4 * M_PI));
}


Vec Shell::rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());

    for(auto j = 0; j < result.rows(); j++) {
        Vec val = Eigen::pow(r(j), m_n1.array()) * Eigen::exp(-m_z.array()*r(j));
        for(size_t i = 0; i < n_orb(); i++) {
            double orb = (m_c.row(i).array() * val.array()).array().sum();
            result(j) += m_occupation(i) * orb * orb;
        }
    }
    return result * (1/(4 * M_PI));
}


Vec Shell::grad_rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    Vec gprime(n_prim()), g(n_prim());

    double Rp = 0.0;
    for(auto p = 0; p < r.rows(); p++) {
        Rp = r(p);
        g = Eigen::pow(Rp, m_n1.array()) * Eigen::exp(-m_z.array() * Rp);
        gprime = g.array() * (m_n1.array()/Rp - m_z.array());

        for(size_t i = 0; i < n_orb(); i++) {
            auto ci = m_c.row(i);
            auto orb =  g.dot(ci);
            auto dorb = gprime.dot(ci);
            result(i) += 2 * m_occupation(i) * orb * dorb;
        }
    }
    return result * (1/(4 * M_PI));
}

Basis::Basis(const std::vector<Shell>& shells) :
    m_shells(shells)
{
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
        result += s.rho(r);
    }
    return result;
}

Vec Basis::grad_rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    for (const auto& s: m_shells) {
        result += s.grad_rho(r);
    }
    return result;
}

}
