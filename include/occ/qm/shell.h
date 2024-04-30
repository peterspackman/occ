#pragma once
#include <array>
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/point_charge.h>
#include <occ/core/util.h>
#include <fmt/core.h>
#include <vector>

namespace occ::qm {

using occ::core::Atom;


void override_basis_set_directory(const std::string &s);

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
    Mat coeffs_normalized_for_libecpint() const;
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
    Mat u_coefficients; // unnormalized coefficients
    Vec max_ln_coefficient;
    IVec ecp_r_exponents;
    double extent{0.0};
};

class AOBasis {
  public:
    using AtomList = std::vector<Atom>;

    using ShellList = std::vector<Shell>;
    AOBasis(const AtomList &, const ShellList &, const std::string &name = "",
            const ShellList &ecp = {});
    AOBasis() {}

    inline size_t nbf() const { return m_nbf; }

    void merge(const AOBasis &);

    inline bool shells_share_origin(size_t p, size_t q) const {
        return m_shell_to_atom_idx[p] == m_shell_to_atom_idx[q];
    }
    inline const auto &name() const { return m_basis_name; }
    inline auto size() const { return m_shells.size(); }
    inline auto ecp_size() const { return m_ecp_shells.size(); }
    inline auto nsh() const { return size(); }
    inline auto ecp_nsh() const { return m_ecp_shells.size(); }
    inline auto kind() const { return m_kind; }
    inline bool is_pure() const { return m_kind == Shell::Kind::Spherical; }
    inline bool is_cartesian() const {
        return m_kind == Shell::Kind::Cartesian;
    }
    inline bool have_ecps() const { return m_ecp_shells.size() > 0; }
    inline void set_kind(Shell::Kind kind) {
        if (kind == m_kind)
            return;
        m_kind = kind;
        for (auto &sh : m_shells) {
            sh.kind = kind;
            m_max_shell_size = std::max(m_max_shell_size, sh.size());
        }
        update_bf_maps();
    }
    inline void set_pure(bool pure) {
        set_kind(pure ? Shell::Kind::Spherical : Shell::Kind::Cartesian);
    }
    inline const Shell &operator[](size_t n) const { return m_shells[n]; }
    inline const Shell &at(size_t n) const { return m_shells.at(n); }
    inline const Shell &ecp_at(size_t n) const { return m_ecp_shells.at(n); }

    inline const auto &shells() const { return m_shells; }
    inline const auto &ecp_shells() const { return m_ecp_shells; }
    inline const auto &atoms() const { return m_atoms; }
    inline const auto &first_bf() const { return m_first_bf; }
    inline const auto &bf_to_shell() const { return m_bf_to_shell; }
    inline const auto &bf_to_atom() const { return m_bf_to_atom; }
    inline const auto &shell_to_atom() const { return m_shell_to_atom_idx; }
    inline const auto &ecp_shell_to_atom() const {
        return m_ecp_shell_to_atom_idx;
    }
    inline const auto &atom_to_shell() const { return m_atom_to_shell_idxs; }
    inline const auto &atom_to_ecp_shell() const {
        return m_atom_to_ecp_shell_idxs;
    }
    inline const auto &ecp_electrons() const { return m_ecp_electrons; }
    inline auto &ecp_electrons() { return m_ecp_electrons; }
    inline void set_ecp_electrons(const std::vector<int> &e) {
        m_ecp_electrons = e;
    }
    inline auto total_ecp_electrons() const {
        return std::accumulate(m_ecp_electrons.begin(), m_ecp_electrons.end(),
                               0);
    }
    uint_fast8_t l_max() const;
    uint_fast8_t ecp_l_max() const;

    void rotate(const Mat3 &rotation);
    void translate(const Vec3 &rotation);

    inline auto max_shell_size() const { return m_max_shell_size; }
    inline auto max_ecp_shell_size() const { return m_max_ecp_shell_size; }
    static AOBasis load(const AtomList &atoms, const std::string &name);

    bool operator==(const AOBasis &rhs) const;

    void calculate_shell_cutoffs();

  private:
    void update_bf_maps();
    std::string m_basis_name;
    AtomList m_atoms;
    ShellList m_shells, m_ecp_shells;
    std::vector<int> m_first_bf;
    std::vector<int> m_shell_to_atom_idx, m_ecp_shell_to_atom_idx;
    std::vector<int> m_bf_to_shell;
    std::vector<int> m_bf_to_atom;
    std::vector<std::vector<int>> m_atom_to_shell_idxs;
    std::vector<std::vector<int>> m_atom_to_ecp_shell_idxs;
    std::vector<int> m_ecp_electrons;
    size_t m_nbf{0};
    size_t m_max_shell_size{0}, m_max_ecp_shell_size{0};
    Shell::Kind m_kind{Shell::Kind::Cartesian};
};



} // namespace occ::qm


template<>
struct fmt::formatter<occ::qm::Shell> : nested_formatter<double> {
    auto format(const occ::qm::Shell&, format_context& ctx) const -> format_context::iterator;
};
