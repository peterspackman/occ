#pragma once
#include <libint2/cxxapi.h>
#include <libint2/shell.h>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

/* this has been modified from libint2::BasisSet
 * due to requirements in construction etc that were not possible
 * with the existing basisset
 */

namespace occ::qm {

using libint2::Shell;
using libint2::svector;
using occ::core::Atom;

template <typename ShellRange> size_t nbf(ShellRange &&shells) {
    size_t n = 0;
    for (const auto &shell : std::forward<ShellRange>(shells))
        n += shell.size();
    return n;
}

template <typename ShellRange> size_t max_nprim(ShellRange &&shells) {
    size_t n = 0;
    for (auto shell : std::forward<ShellRange>(shells))
        n = std::max(shell.nprim(), n);
    return n;
}

template <typename ShellRange> int max_l(ShellRange &&shells) {
    int l = 0;
    for (auto shell : std::forward<ShellRange>(shells))
        for (auto c : shell.contr)
            l = std::max(c.l, l);
    return l;
}

class BasisSet : public std::vector<libint2::Shell> {
  public:
    BasisSet() : m_name(""), m_nbf(-1), m_max_nprim(0), m_max_l(-1) {}
    BasisSet(const BasisSet &) = default;
    BasisSet(BasisSet &&other)
        : std::vector<libint2::Shell>(std::move(other)),
          m_name(std::move(other.m_name)), m_nbf(other.m_nbf),
          m_max_nprim(other.m_max_nprim), m_max_l(other.m_max_l),
          m_shell2bf(std::move(other.m_shell2bf)) {}
    ~BasisSet() = default;
    BasisSet &operator=(const BasisSet &) = default;

    BasisSet(std::string name, const std::vector<Atom> &atoms);

    BasisSet(const std::vector<Atom> &atoms,
             const std::vector<std::vector<Shell>> &element_bases,
             std::string name = "");

    void set_pure(bool solid) {
        for (auto &s : *this) {
            s.contr[0].pure = solid;
        }
        update();
    }

    bool is_pure() const {
        for (const auto &s : *this) {
            if (s.contr[0].pure)
                return true;
        }
        return false;
    }

    long nbf() const { return m_nbf; }

    size_t max_nprim() const { return m_max_nprim; }

    long max_l() const { return m_max_l; }

    const std::vector<size_t> &shell2bf() const { return m_shell2bf; }

    std::vector<long> shell2atom(const std::vector<Atom> &atoms) const {
        return shell2atom(*this, atoms);
    }

    std::vector<std::vector<long>>
    atom2shell(const std::vector<Atom> &atoms) const {
        return atom2shell(atoms, *this);
    }

    std::vector<long> bf2atom(const std::vector<Atom> &atoms) const;

    static std::vector<long> shell2atom(const std::vector<Shell> &shells,
                                        const std::vector<Atom> &atoms);

    static std::vector<std::vector<long>>
    atom2shell(const std::vector<Atom> &atoms,
               const std::vector<Shell> &shells);

    void update();

    void rotate(const occ::Mat3 &rotation);
    void translate(const occ::Vec3 &translation);

  private:
    std::string m_name;
    long m_nbf;
    size_t m_max_nprim;
    int m_max_l;
    std::vector<size_t> m_shell2bf;

    static std::string canonicalize_name(const std::string &name);

    bool gaussian_cartesian_d_convention(const std::string &canonical_name);

    std::vector<std::string> decompose_name_into_components(std::string name);

    static std::string data_path();
};

Mat rotate_molecular_orbitals(const BasisSet &, const Mat3 &, const Mat &);

inline std::vector<occ::core::Atom>
rotated_atoms(const std::vector<occ::core::Atom> &atoms,
              const occ::Mat3 &rotation) {
    auto result = atoms;
    rotate_atoms(result, rotation);
    return result;
}

inline std::vector<occ::core::Atom>
translated_atoms(const std::vector<occ::core::Atom> &atoms,
                 const occ::Vec3 &translation) {
    auto result = atoms;
    translate_atoms(result, translation);
    return result;
}

std::vector<size_t> pople_sp_shells(const BasisSet &);

} // namespace occ::qm
