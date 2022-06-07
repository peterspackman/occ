#pragma once
#include <array>
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/point_charge.h>
#include <occ/core/util.h>
#include <occ/qm/basisset.h>
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
    OccShell(const occ::core::PointCharge &);
    OccShell();
    OccShell(const libint2::Shell &);

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
    double extent{0.0};
};

class AOBasis {
  public:
    using AtomList = std::vector<Atom>;

    using ShellList = std::vector<OccShell>;
    AOBasis(const AtomList &, const ShellList &);
    AOBasis() {}

    inline size_t nbf() const { return m_nbf; }
    inline size_t first_bf(size_t shell_index) const {
        return m_first_bf[shell_index];
    }

    void add_shell(const OccShell &);
    void merge(const AOBasis &);

    inline bool shells_share_origin(size_t p, size_t q) const {
        return m_shell_to_atom_idx[p] == m_shell_to_atom_idx[q];
    }

    inline auto size() const { return m_shells.size(); }
    inline auto nsh() const { return size(); }
    inline auto kind() const { return m_kind; }
    inline const OccShell &operator[](size_t n) const { return m_shells[n]; }
    inline const OccShell &at(size_t n) const { return m_shells.at(n); }

    inline const auto &shells() const { return m_shells; }
    inline const auto &atoms() const { return m_atoms; }
    inline const auto &first_bf() const { return m_first_bf; }
    inline const auto &bf_to_shell() const { return m_bf_to_shell; }
    inline const auto &shell_to_atom() const { return m_shell_to_atom_idx; }
    inline const auto &atom_to_shell() const { return m_atom_to_shell_idxs; }

    inline auto max_shell_size() const { return m_max_shell_size; }

  private:
    AtomList m_atoms;
    ShellList m_shells;
    std::vector<int> m_first_bf;
    std::vector<int> m_shell_to_atom_idx;
    std::vector<int> m_bf_to_shell;
    std::vector<std::vector<int>> m_atom_to_shell_idxs;
    size_t m_nbf{0};
    size_t m_max_shell_size{0};
    OccShell::Kind m_kind{OccShell::Kind::Cartesian};
};

std::vector<OccShell> from_libint2_basis(const BasisSet &basis);
std::ostream &operator<<(std::ostream &stream, const OccShell &shell);

} // namespace occ::qm
