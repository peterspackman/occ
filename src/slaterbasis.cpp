#include "slaterbasis.h"
#include <fmt/core.h>

namespace tonto::slater {

using tonto::IVec;
using tonto::Vec;
using tonto::Mat;

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
    return m_occupation.cols();
}

double Shell::rho(double r) const {
    Vec g = Eigen::pow(r, m_n1.array()) * Eigen::exp(-m_z.array()*r);
    double result = 0;
    for(size_t i = 0; i < n_orb(); i++) {
        double orb = (m_c.col(i).array() * g.array()).array().sum();
        result += m_occupation(i) * orb * orb;
    }

    return result * (1/(4 * M_PI));
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
    return result * (1/(4 * M_PI));
}


Vec Shell::rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());

    Mat e(m_n1.rows(), r.rows());
    for(auto i = 0; i < result.rows(); i++) {
        e.col(i).array() = Eigen::pow(r(i), m_n1.array()) * Eigen::exp(-m_z.array()*r(i));
    }

    for(size_t i = 0; i < n_orb(); i++) {
        Vec orb = (e.array().colwise() * m_c.col(i).array()).colwise().sum();
        result.array() += m_occupation(i) * orb.array() * orb.array();
    }
    return result * (1/(4 * M_PI));
}


Vec Shell::grad_rho(const Vec& r) const
{
    Vec result = Vec::Zero(r.rows());
    Mat e(m_n1.rows(), r.rows());
    Mat de(m_n1.rows(), r.rows());

    for(auto i = 0; i < result.rows(); i++) {
        e.col(i).array() = Eigen::pow(r(i), m_n1.array()) * Eigen::exp(-m_z.array()*r(i));
        if(r(i) == 0.0) de.col(i).array() = 0.0;
        else de.col(i).array() = e.col(i).array() * (m_n1.array() / r(i) - m_z.array());
    }

    for(size_t i = 0; i < n_orb(); i++) {
        auto ci = m_c.col(i).array();
        Vec orb = (e.array().colwise() * ci).colwise().sum();
        Vec dorb = (de.array().colwise() * ci).colwise().sum();
        result.array() += 2 * m_occupation(i) * orb.array() * dorb.array();
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
