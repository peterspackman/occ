#pragma once
#include "linear_algebra.h"
#include <vector>
#include <string>

namespace tonto::slater {

using tonto::IVec;
using tonto::Vec;
using tonto::MatRM;

class Shell {
public:
    Shell();
    Shell(
        const IVec&,
        const IVec&,
        const Vec&,
        const MatRM&
    );
    double rho(double r) const;
    double grad_rho(double r) const;
    Vec rho(const Vec&) const;
    Vec grad_rho(const Vec&) const;
    size_t n_prim() const;
    size_t n_orb() const;
private:
    IVec m_occupation;
    IVec m_n;
    Vec m_n1;
    Vec m_z;
    MatRM m_c;
};


class Basis {
public:
    Basis(const std::vector<Shell>&);
    double rho(double r) const;
    double grad_rho(double r) const;
    Vec rho(const Vec&) const;
    Vec grad_rho(const Vec&) const;
private:
    std::vector<Shell> m_shells;
};

}
