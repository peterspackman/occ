#pragma once
#include <occ/3rdparty/cint_wrapper.h>
#include <occ/qm/occshell.h>

namespace occ::qm::cint {

using occ::core::Atom;
using Shell = occ::qm::OccShell;

enum class Operator { overlap, nuclear, kinetic, coulomb };

namespace impl {
struct AtomInfo {
    int data[6] = {0, 0, 0, 0, 0, 0};
    constexpr static int _charge_offset{0};
    constexpr static int _env_coord_offset{1};
    constexpr static int _nuclear_model_offset{2};
    constexpr static int _env_nuclear_charge_dist_offset{3};
    void set_charge(int charge) { data[_charge_offset] = charge; }
    int charge() const { return data[_charge_offset]; }

    void set_env_coord_offset(int offset) { data[_env_coord_offset] = offset; }
    int env_coord_offset() const { return data[_env_coord_offset]; }

    void set_nuclear_model(int model) { data[_nuclear_model_offset] = model; }
    int nuclear_model() const { return data[_nuclear_model_offset]; }

    void set_env_nuclear_charge_dist_offset(int offset) {
        data[_env_nuclear_charge_dist_offset] = offset;
    }
    int set_env_nuclear_charge_dist_offset() const {
        return data[_env_nuclear_charge_dist_offset];
    }
};

struct BasisInfo {
    int data[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static int _env_atom_offset{0};
    constexpr static int _angular_momentum_offset{1};
    constexpr static int _num_prim_offset{2};
    constexpr static int _num_contracted_offset{3};
    constexpr static int _kappa_offset{4};
    constexpr static int _env_primitive_offset{5};
    constexpr static int _env_contraction_offset{6};

    void set_env_atom_offset(int offset) { data[_env_atom_offset] = offset; }
    int env_atom_offset() const { return data[_env_atom_offset]; }

    void set_angular_momentum(int offset) {
        data[_angular_momentum_offset] = offset;
    }
    int angular_momentum() const { return data[_angular_momentum_offset]; }

    void set_num_primitives(int num_prim) { data[_num_prim_offset] = num_prim; }
    int num_primitives() const { return data[_num_prim_offset]; }

    void set_num_contracted(int num_contracted) {
        data[_num_contracted_offset] = num_contracted;
    }
    int num_contracted() const { return data[_num_contracted_offset]; }

    void set_kappa(int kappa) { data[_kappa_offset] = kappa; }
    int kappa() const { return data[_kappa_offset]; }

    void set_env_primitive_offset(int offset) {
        data[_env_primitive_offset] = offset;
    }
    int env_primitive_offset() const { return data[_env_primitive_offset]; }

    void set_env_contraction_offset(int offset) {
        data[_env_contraction_offset] = offset;
    }
    int env_contraction_offset() const { return data[_env_contraction_offset]; }
};

} // namespace impl

class IntegralEnvironment {
  public:
    static constexpr int environment_pointer_start_offset = PTR_ENV_START;

    IntegralEnvironment(const std::vector<Atom> &atoms,
                        const std::vector<Shell> &basis)
        : m_atom_info(atoms.size()), m_basis_info(basis.size()) {
        int atom_idx = 0;
        int env_data_size{environment_pointer_start_offset};

        for (const auto &atom : atoms) {
            auto &atom_info = m_atom_info[atom_idx];
            atom_info.set_charge(atom.atomic_number);
            atom_info.set_env_coord_offset(env_data_size);
            atom_info.set_env_nuclear_charge_dist_offset(env_data_size + 3);
            env_data_size += 4;
            atom_idx++;
        }

        int current_primitive_offset{0}, current_coefficient_offset{0};
        int bas_idx = 0;
        for (const auto &shell : basis) {
            auto &basis_info = m_basis_info[bas_idx];
            current_primitive_offset = env_data_size;
            int atom_idx = shell.find_atom_index(atoms);
            env_data_size += shell.libcint_environment_size();
            current_coefficient_offset =
                current_primitive_offset + shell.exponents.size();
            basis_info.set_angular_momentum(shell.l);
            basis_info.set_num_primitives(shell.num_primitives());
            basis_info.set_num_contracted(shell.num_contractions());
            basis_info.set_env_primitive_offset(current_primitive_offset);
            basis_info.set_env_contraction_offset(current_coefficient_offset);
            basis_info.set_env_atom_offset(atom_idx);
            bas_idx++;
        }
        m_env_data.reserve(env_data_size);
        for (size_t i = 0; i < environment_pointer_start_offset; i++)
            m_env_data.push_back(0.0);
        for (const auto &atom : atoms) {
            m_env_data.push_back(atom.x);
            m_env_data.push_back(atom.y);
            m_env_data.push_back(atom.z);
            m_env_data.push_back(-1.0);
        }
        for (const auto &shell : basis) {
            size_t nprim = shell.exponents.size();
            size_t ncoef = shell.contraction_coefficients.size();
            const auto &exp = shell.exponents.data();
            const auto &coeff = shell.contraction_coefficients.data();
            std::copy(exp, exp + nprim, std::back_inserter(m_env_data));
            std::copy(coeff, coeff + ncoef, std::back_inserter(m_env_data));
        }
    }

    //    size_t add_auxiliary_basis(const std::vector<Shell> &basis) {
    //        return m_auxiliary_offset;
    //    }

    size_t atom_info_size_bytes() const {
        return m_atom_info.size() * sizeof(impl::AtomInfo);
    }
    size_t basis_info_size_bytes() const {
        return m_basis_info.size() * sizeof(impl::BasisInfo);
    }
    size_t basis_env_size_bytes() const {
        return m_env_data.size() * sizeof(double);
    }

    const int *basis_data_ptr() const {
        return reinterpret_cast<const int *>(m_basis_info.data());
    }
    int *basis_data_ptr() {
        return reinterpret_cast<int *>(m_basis_info.data());
    }
    const int *atom_data_ptr() const {
        return reinterpret_cast<const int *>(m_atom_info.data());
    }
    int *atom_data_ptr() { return reinterpret_cast<int *>(m_atom_info.data()); }
    const double *env_data_ptr() const { return m_env_data.data(); }
    double *env_data_ptr() { return m_env_data.data(); }

    int num_atoms() const { return m_atom_info.size(); }
    int num_basis() const { return m_basis_info.size(); }

    template <Shell::Kind ST> inline int total_cgto() const {
        if constexpr (ST == Shell::Kind::Spherical)
            return libcint::CINTtot_cgto_spheric(basis_data_ptr(), num_basis());
        else
            return libcint::CINTtot_cgto_cart(basis_data_ptr(), num_basis());
    }

    template <Shell::Kind ST> inline int cgto(int shell_idx) const {
        if constexpr (ST == Shell::Kind::Spherical)
            return libcint::CINTcgto_spheric(shell_idx, basis_data_ptr());
        else
            return libcint::CINTcgto_cart(shell_idx, basis_data_ptr());
    }

    template <Operator OP, Shell::Kind ST>
    inline int one_electron_cache_size(std::array<int, 2> shells) {
        static_assert(OP != Operator::coulomb, "not a one-electron operator");
        std::array<int, 2> dims{cgto<ST>(shells[0]), cgto<ST>(shells[1])};
        if constexpr (ST == Shell::Kind::Spherical) {
            if constexpr (OP == Operator::overlap) {
                return libcint::int1e_ovlp_sph(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            } else if constexpr (OP == Operator::nuclear) {
                return libcint::int1e_nuc_sph(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            } else if constexpr (OP == Operator::kinetic) {
                return libcint::int1e_kin_sph(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            }
        } else {
            if constexpr (OP == Operator::overlap) {
                return libcint::int1e_ovlp_cart(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            } else if constexpr (OP == Operator::nuclear) {
                return libcint::int1e_nuc_cart(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            } else if constexpr (OP == Operator::kinetic) {
                return libcint::int1e_kin_cart(
                    nullptr, dims.data(), shells.data(), atom_data_ptr(),
                    num_atoms(), basis_data_ptr(), num_basis(), env_data_ptr(),
                    nullptr, nullptr);
            }
        }
    }

    template <Operator OP, Shell::Kind ST>
    inline std::array<int, 2>
    one_electron_helper(std::array<int, 2> shells, libcint::CINTOpt *opt,
                        double *buffer, double *cache) {
        static_assert(OP != Operator::coulomb, "not a one-electron operator");
        std::array<int, 2> dims{cgto<ST>(shells[0]), cgto<ST>(shells[1])};
        if constexpr (ST == Shell::Kind::Spherical) {
            if constexpr (OP == Operator::overlap)
                libcint::int1e_ovlp_sph(buffer, dims.data(), shells.data(),
                                        atom_data_ptr(), num_atoms(),
                                        basis_data_ptr(), num_basis(),
                                        env_data_ptr(), opt, cache);
            else if constexpr (OP == Operator::nuclear)
                libcint::int1e_nuc_sph(buffer, dims.data(), shells.data(),
                                       atom_data_ptr(), num_atoms(),
                                       basis_data_ptr(), num_basis(),
                                       env_data_ptr(), opt, cache);
            else if constexpr (OP == Operator::kinetic)
                libcint::int1e_kin_sph(buffer, dims.data(), shells.data(),
                                       atom_data_ptr(), num_atoms(),
                                       basis_data_ptr(), num_basis(),
                                       env_data_ptr(), opt, cache);
        } else {
            if constexpr (OP == Operator::overlap)
                libcint::int1e_ovlp_cart(buffer, dims.data(), shells.data(),
                                         atom_data_ptr(), num_atoms(),
                                         basis_data_ptr(), num_basis(),
                                         env_data_ptr(), opt, cache);
            else if constexpr (OP == Operator::nuclear)
                libcint::int1e_nuc_cart(buffer, dims.data(), shells.data(),
                                        atom_data_ptr(), num_atoms(),
                                        basis_data_ptr(), num_basis(),
                                        env_data_ptr(), opt, cache);
            else if constexpr (OP == Operator::kinetic)
                libcint::int1e_kin_cart(buffer, dims.data(), shells.data(),
                                        atom_data_ptr(), num_atoms(),
                                        basis_data_ptr(), num_basis(),
                                        env_data_ptr(), opt, cache);
        }
        return dims;
    }

    template <Operator OP, Shell::Kind ST>
    inline std::array<int, 4>
    two_electron_helper(std::array<int, 4> shells, libcint::CINTOpt *opt,
                        double *buffer, double *cache) {
        static_assert(OP == Operator::coulomb, "not a two-electron operator");
        std::array<int, 4> dims{
            cgto<ST>(shells[0]),
            cgto<ST>(shells[1]),
            cgto<ST>(shells[2]),
            cgto<ST>(shells[3]),
        };
        int nonzero = 0;

        if constexpr (ST == Shell::Kind::Spherical) {
            nonzero = libcint::int2e_sph(buffer, dims.data(), shells.data(),
                                         atom_data_ptr(), num_atoms(),
                                         basis_data_ptr(), num_basis(),
                                         env_data_ptr(), opt, cache);
        } else {
            nonzero = libcint::int2e_cart(buffer, dims.data(), shells.data(),
                                          atom_data_ptr(), num_atoms(),
                                          basis_data_ptr(), num_basis(),
                                          env_data_ptr(), opt, cache);
        }
        if (nonzero == 0) {
            fmt::print("Zero\n");
        }
        return dims;
    }

    template <Operator OP, Shell::Kind ST>
    inline std::array<int, 3>
    three_center_two_electron_helper(std::array<int, 3> shells,
                                     libcint::CINTOpt *opt, double *buffer,
                                     double *cache) {
        static_assert(OP == Operator::coulomb, "not a two-electron operator");
        std::array<int, 3> dims{
            cgto<ST>(shells[0]),
            cgto<ST>(shells[1]),
            cgto<ST>(shells[2]),
        };
        int nonzero = 0;

        if constexpr (ST == Shell::Kind::Spherical) {
            nonzero = libcint::int3c2e_sph(buffer, dims.data(), shells.data(),
                                           atom_data_ptr(), num_atoms(),
                                           basis_data_ptr(), num_basis(),
                                           env_data_ptr(), opt, cache);
        } else {
            nonzero = libcint::int3c2e_cart(buffer, dims.data(), shells.data(),
                                            atom_data_ptr(), num_atoms(),
                                            basis_data_ptr(), num_basis(),
                                            env_data_ptr(), opt, cache);
        }
        if (nonzero == 0) {
            fmt::print("Zero\n");
        }
        return dims;
    }

  private:
    size_t m_auxiliary_offset{0};
    std::vector<impl::AtomInfo> m_atom_info;
    std::vector<impl::BasisInfo> m_basis_info;
    std::vector<double> m_env_data;
};

template <Operator OP> class Optimizer {
  public:
    Optimizer(IntegralEnvironment env) {
        if constexpr (OP == Operator::coulomb)
            libcint::int2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                                     env.num_atoms(), env.basis_data_ptr(),
                                     env.num_basis(), env.env_data_ptr());
        else if constexpr (OP == Operator::nuclear)
            libcint::int1e_nuc_optimizer(&m_optimizer, env.atom_data_ptr(),
                                         env.num_atoms(), env.basis_data_ptr(),
                                         env.num_basis(), env.env_data_ptr());
    }

    ~Optimizer() { libcint::CINTdel_optimizer(&m_optimizer); }
    inline auto optimizer_ptr() { return m_optimizer; }

  private:
    libcint::CINTOpt *m_optimizer{nullptr};
};

} // namespace occ::qm::cint
