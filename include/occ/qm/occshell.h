#pragma once
#include <array>
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <vector>

namespace occ::qm {

using occ::core::Atom;

double gto_norm(int l, double alpha);

struct OccShell {

    enum Kind {
        Cartesian,
        Spherical,
    };

    constexpr static double pi2_34 =
        3.9685778240728024992720094621189610321284055835201069917099724937;

    OccShell(int, const std::vector<double> &expo,
             const std::vector<std::vector<double>> &contr,
             const std::array<double, 3> &pos);

    bool operator==(const OccShell &other) const;
    bool operator!=(const OccShell &other) const;

    size_t num_primitives() const;
    size_t num_contractions() const;

    double norm() const;

    double max_exponent() const;
    double min_exponent() const;
    void incorporate_shell_norm();
    double coeff_normalized(Eigen::Index contr_idx,
                            Eigen::Index coeff_idx) const;
    size_t size() const;

    static char l_to_symbol(uint_fast8_t l);

    static uint_fast8_t symbol_to_l(char symbol);
    char symbol() const;

    OccShell translated_copy(const Eigen::Vector3d &origin) const;
    size_t libcint_environment_size() const;

    int find_atom_index(const std::vector<Atom> &atoms) const;

    bool is_pure() const;

    bool operator<(const OccShell &other) const;
    Kind kind{Cartesian};
    uint_fast8_t l;
    Vec3 origin;
    Vec exponents;
    Mat contraction_coefficients;
    Vec max_ln_coefficient;
};

std::ostream &operator<<(std::ostream &stream, const OccShell &shell);

} // namespace occ::qm
