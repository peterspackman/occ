#pragma once
#include <array>
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/point_charge.h>
#include <occ/core/util.h>
#include <vector>

namespace occ::qm {

using occ::core::Atom;

double gto_norm(int l, double alpha);

struct Shell {

    enum Kind {
        Cartesian,
        Spherical,
    };

    constexpr static double pi2_34 =
        3.9685778240728024992720094621189610321284055835201069917099724937;

    Shell(int, const std::vector<double> &expo,
          const std::vector<std::vector<double>> &contr,
          const std::array<double, 3> &pos);
    Shell(const occ::core::PointCharge &);
    Shell();

    bool operator==(const Shell &other) const;
    bool operator!=(const Shell &other) const;

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

    Shell translated_copy(const Eigen::Vector3d &origin) const;
    size_t libcint_environment_size() const;

    int find_atom_index(const std::vector<Atom> &atoms) const;

    bool is_pure() const;

    bool operator<(const Shell &other) const;
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

    using ShellList = std::vector<Shell>;
    AOBasis(const AtomList &, const ShellList &, const std::string &name = "");
    AOBasis() {}

    inline size_t nbf() const { return m_nbf; }

    void add_shell(const Shell &);
    void merge(const AOBasis &);

    inline bool shells_share_origin(size_t p, size_t q) const {
        return m_shell_to_atom_idx[p] == m_shell_to_atom_idx[q];
    }
    inline const auto &name() const { return m_basis_name; }
    inline auto size() const { return m_shells.size(); }
    inline auto nsh() const { return size(); }
    inline auto kind() const { return m_kind; }
    inline bool is_pure() const { return m_kind == Shell::Kind::Spherical; }
    inline bool is_cartesian() const {
        return m_kind == Shell::Kind::Cartesian;
    }
    inline void set_kind(Shell::Kind kind) {
        if (kind == m_kind)
            return;
        m_kind = kind;
        for (auto &sh : m_shells)
            sh.kind = kind;
        update_bf_maps();
    }
    inline void set_pure(bool pure) {
        set_kind(pure ? Shell::Kind::Spherical : Shell::Kind::Cartesian);
    }
    inline const Shell &operator[](size_t n) const { return m_shells[n]; }
    inline const Shell &at(size_t n) const { return m_shells.at(n); }

    inline const auto &shells() const { return m_shells; }
    inline const auto &atoms() const { return m_atoms; }
    inline const auto &first_bf() const { return m_first_bf; }
    inline const auto &bf_to_shell() const { return m_bf_to_shell; }
    inline const auto &bf_to_atom() const { return m_bf_to_atom; }
    inline const auto &shell_to_atom() const { return m_shell_to_atom_idx; }
    inline const auto &atom_to_shell() const { return m_atom_to_shell_idxs; }
    uint_fast8_t l_max() const;

    void rotate(const Mat3 &rotation);
    void translate(const Vec3 &rotation);

    inline auto max_shell_size() const { return m_max_shell_size; }
    static AOBasis load(const AtomList &atoms, const std::string &name);

    bool operator==(const AOBasis &rhs);

  private:
    void update_bf_maps();
    std::string m_basis_name;
    AtomList m_atoms;
    ShellList m_shells;
    std::vector<int> m_first_bf;
    std::vector<int> m_shell_to_atom_idx;
    std::vector<int> m_bf_to_shell;
    std::vector<int> m_bf_to_atom;
    std::vector<std::vector<int>> m_atom_to_shell_idxs;
    size_t m_nbf{0};
    size_t m_max_shell_size{0};
    Shell::Kind m_kind{Shell::Kind::Cartesian};
};

std::ostream &operator<<(std::ostream &stream, const Shell &shell);

} // namespace occ::qm
