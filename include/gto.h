#pragma once
#include <libint2/shell.h>
#include <libint2/basis.h>
#include "linear_algebra.h"
#include <string>
#include <vector>
#include <array>
#include <libint2/cgshell_ordering.h>
#include <libint2/shgshell_ordering.h>
#include "timings.h"
#include <fmt/core.h>

namespace tonto::gto {

template<size_t max_derivative>
struct GTOValues {
    GTOValues(size_t nbf, size_t npts) : phi(npts, nbf) {}
    inline void set_zero() {
        phi.setZero();
    }
    tonto::Mat phi;
};

template<>
struct GTOValues<1>
{
    GTOValues(size_t nbf, size_t npts) : phi(npts, nbf), phi_x(npts, nbf), phi_y(npts, nbf), phi_z(npts, nbf) {}
    inline void set_zero() {
        phi.setZero();
        phi_x.setZero();
        phi_y.setZero();
        phi_z.setZero();
    }
    tonto::Mat phi;
    tonto::Mat phi_x;
    tonto::Mat phi_y;
    tonto::Mat phi_z;
};

struct Momenta {
    int l{0};
    int m{0};
    int n{0};

    std::string to_string() const {
        int am = l + m + n;
        static char lsymb[] = "SPDFGHIKMNOQRTUVWXYZ";
        if (am == 0) return std::string(1, lsymb[0]);

        std::string suffix = "";
        for(int i = 0; i < l; i++) suffix += "x";
        for(int i = 0; i < m; i++) suffix += "y";
        for(int i = 0; i < n; i++) suffix += "z";

        return std::string(1, lsymb[am]) + suffix;
    }
};

inline std::vector<Momenta> cartesian_ordering(int l) {
    if(l == 0) return {{0, 0, 0}};
    int i = 0, j = 0, k = 0;
    std::vector<Momenta> powers;
    FOR_CART(i,j,k,l)
        powers.push_back({i, j, k});
    END_FOR_CART
    return powers;
}

namespace impl {

template<size_t max_derivative>
void add_shell_contribution(size_t bf, const libint2::Shell &shell, const tonto::Mat& dists,
                    GTOValues<max_derivative>& result,
                    const tonto::MaskArray& mask)
{

    for(size_t pt = 0; pt < dists.cols(); pt++) {
        if(!mask(pt)) continue;
        double dx = dists(0, pt), dy = dists(1, pt), dz = dists(2, pt), r2 = dists(3, pt);
        double rdx = 0.0, rdy = 0.0, rdz = 0.0;
        if constexpr(max_derivative > 0) {
            if(dx != 0) rdx = 1.0 / dx;
            if(dy != 0) rdy = 1.0 / dy;
            if(dz != 0) rdy = 1.0 / dz;
        }
        for(size_t prim = 0; prim < shell.nprim(); prim++)
        {
            double alpha = shell.alpha[prim];
            double expfac = exp(-alpha * r2);
            for(const auto& contraction: shell.contr) {
                double cexp = contraction.coeff[prim] * expfac;
                int l, m, n;
                size_t offset = 0;
                FOR_CART(l, m, n, contraction.l)
                    double poly = pow(dx, l) * pow(dy, m) * pow(dz, n);
                    double f = poly * cexp;
                    result.phi(pt, bf + offset) += f;
                    if constexpr(max_derivative > 0) {
                        result.phi_x(pt, bf + offset) += (l * rdx - 2 * alpha * dx) * f;
                        result.phi_y(pt, bf + offset) += (m * rdy - 2 * alpha * dy) * f;
                        result.phi_z(pt, bf + offset) += (n * rdz - 2 * alpha * dz) * f;
                    }
                    offset++;
                END_FOR_CART
            }
        }
    }
}

}

template<size_t max_derivative>
GTOValues<max_derivative> evaluate_basis_on_grid(const libint2::BasisSet &basis,
                                         const std::vector<libint2::Atom> &atoms,
                                         const tonto::Mat &grid_pts)
{
    tonto::timing::start(tonto::timing::category::gto);
    size_t nbf = basis.nbf();
    size_t npts = grid_pts.cols();
    size_t natoms = atoms.size();
    GTOValues<max_derivative> gto_values(nbf, npts);
    gto_values.set_zero();
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);
    constexpr auto EXPCUTOFF{50};

    for(size_t i = 0; i < natoms; i++)
    {
        const auto& atom = atoms[i];
        tonto::Mat dists(4, npts);
        tonto::MaskArray mask(npts);
        tonto::Vec3 xyz(atom.x, atom.y, atom.z);

        dists.block(0, 0, 3, npts) = grid_pts.block(0, 0, 3, npts).colwise() - xyz;
        dists.row(3) = dists.block(0, 0, 3, npts).colwise().squaredNorm();
        for(const auto& shell_idx: atom2shell[i]) {
            const auto& shell = basis[shell_idx];
            size_t bf = shell2bf[shell_idx];
            for(size_t pt = 0; pt < npts; pt++) {
                mask(pt) = true;
                continue;
                for(size_t prim = 0; prim < shell.nprim(); prim++) {
                    if((shell.alpha[prim] * dists(3, pt) - shell.max_ln_coeff[prim]) < EXPCUTOFF) {
                        mask(pt) = true;
                        break;
                    }
                }
            }
            impl::add_shell_contribution<max_derivative>(bf, shell, dists, gto_values, mask);
        }
    }
    tonto::timing::stop(tonto::timing::category::gto);
    return gto_values;
}

}
