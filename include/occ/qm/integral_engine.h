#pragma once
#include <array>
#include <libecpint.hpp>
#include <occ/core/atom.h>
#include <occ/core/log.h>
#include <occ/core/multipole.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/qm/shellblock_norm.h>
#include <optional>
#include <vector>

namespace occ::qm {

class IntegralEngine {
  public:
    struct ECPCenter {
        int index{0};
        libecpint::ECP ecp;
    };

    template <size_t num_centers> struct IntegralResult {
        int thread{0};
        std::array<int, num_centers> shell{0};
        std::array<int, num_centers> bf{0};
        std::array<int, num_centers> dims{0};
        const double *buffer{nullptr};
    };

    using ShellList = std::vector<Shell>;
    using AtomList = std::vector<occ::core::Atom>;
    using ShellPairList = std::vector<std::vector<size_t>>;
    using IntEnv = cint::IntegralEnvironment;
    using ShellKind = Shell::Kind;
    using Op = cint::Operator;

    IntegralEngine(const AtomList &at, const ShellList &sh)
        : m_aobasis(at, sh), m_env(at, sh) {

        if (is_spherical()) {
            compute_shellpairs<ShellKind::Spherical>();
        } else {
            compute_shellpairs<ShellKind::Cartesian>();
        }
    }

    IntegralEngine(const AOBasis &basis)
        : m_aobasis(basis), m_env(basis.atoms(), basis.shells()) {

        if (is_spherical()) {
            compute_shellpairs<ShellKind::Spherical>();
        } else {
            compute_shellpairs<ShellKind::Cartesian>();
        }

        if (m_aobasis.have_ecps()) {
            set_effective_core_potentials(m_aobasis.ecp_shells(),
                                          m_aobasis.ecp_electrons());
        }
    }

    inline auto nbf() const noexcept { return m_aobasis.nbf(); }
    inline auto nbf_aux() const noexcept { return m_auxbasis.nbf(); }
    inline auto nsh() const noexcept { return m_aobasis.nsh(); }
    inline auto nsh_aux() const noexcept { return m_auxbasis.nsh(); }
    inline const AOBasis &aobasis() const { return m_aobasis; }
    inline const AOBasis &auxbasis() const { return m_auxbasis; }
    inline const auto &env() const { return m_env; }
    inline auto &env() { return m_env; }

    inline const auto &first_bf() const noexcept {
        return m_aobasis.first_bf();
    }
    inline const auto &first_bf_aux() const noexcept {
        return m_auxbasis.first_bf();
    }
    inline const auto &shellpairs() const noexcept { return m_shellpairs; }
    inline const auto &shells() const noexcept { return m_aobasis.shells(); }

    inline void set_auxiliary_basis(const ShellList &bs, bool dummy = false) {
        if (!dummy) {
            clear_auxiliary_basis();
            m_auxbasis = AOBasis(m_aobasis.atoms(), bs);
            ShellList combined = m_aobasis.shells();
            combined.insert(combined.end(), m_auxbasis.shells().begin(),
                            m_auxbasis.shells().end());
            m_env = IntEnv(m_aobasis.atoms(), combined);
        } else {
            AtomList dummy_atoms;
            dummy_atoms.reserve(bs.size());
            for (const auto &shell : bs) {
                dummy_atoms.push_back(
                    {0, shell.origin(0), shell.origin(1), shell.origin(2)});
            }
            set_dummy_basis(dummy_atoms, bs);
        }
    }

    inline void set_dummy_basis(const AtomList &dummy_atoms,
                                const ShellList &bs) {
        clear_auxiliary_basis();
        m_auxbasis = AOBasis(dummy_atoms, bs);
        AtomList combined_sites = m_aobasis.atoms();
        combined_sites.insert(combined_sites.end(), dummy_atoms.begin(),
                              dummy_atoms.end());
        ShellList combined = m_aobasis.shells();
        combined.insert(combined.end(), m_auxbasis.shells().begin(),
                        m_auxbasis.shells().end());
        m_env = IntEnv(combined_sites, combined);
    }

    inline void clear_auxiliary_basis() {
        m_auxbasis = AOBasis();
        m_env = IntEnv(m_aobasis.atoms(), m_aobasis.shells());
    }

    inline bool have_auxiliary_basis() const noexcept {
        return m_auxbasis.nsh() > 0;
    }

    inline bool have_effective_core_potentials() const noexcept {
        return m_have_ecp;
    }

    Mat one_electron_operator(Op op, bool use_shellpair_list = true) const;
    Mat effective_core_potential(bool use_shellpair_list = true) const;
    Mat fock_operator(SpinorbitalKind, const MolecularOrbitals &mo,
                      const Mat &Schwarz = Mat()) const;
    Mat fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                  bool is_shell_diagonal);
    Mat coulomb(SpinorbitalKind, const MolecularOrbitals &mo,
                const Mat &Schwarz = Mat()) const;
    std::pair<Mat, Mat> coulomb_and_exchange(SpinorbitalKind,
                                             const MolecularOrbitals &mo,
                                             const Mat &Schwarz = Mat()) const;

    Mat
    point_charge_potential(const std::vector<occ::core::PointCharge> &charges);
    Vec electric_potential(const MolecularOrbitals &mo, const Mat3N &points);

    template <ShellKind kind>
    inline void compute_shellpairs(double threshold = 1e-12) {
        occ::log::debug("computing shellpairs (threshold = {}, kind = {}",
                        threshold,
                        kind == ShellKind::Cartesian ? "cartesian" : "pure");
        constexpr auto op = Op::overlap;
        const auto nsh = m_aobasis.size();
        size_t num_significant_shellpairs{0}, num_total_shellpairs{0};
        m_shellpairs.resize(nsh);
        auto buffer = std::make_unique<double[]>(buffer_size_1e());
        for (int p = 0; p < nsh; p++) {
            auto &plist = m_shellpairs[p];
            const auto &sh1 = m_aobasis[p];
            for (int q = 0; q <= p; q++) {
                num_total_shellpairs++;
                if (m_aobasis.shells_share_origin(p, q)) {
                    num_significant_shellpairs++;
                    plist.push_back(q);
                    continue;
                }
                const auto &sh2 = m_aobasis[q];
                std::array<int, 2> idxs{p, q};
                std::array<int, 2> dims = m_env.two_center_helper<op, kind>(
                    idxs, nullptr, buffer.get(), nullptr);
                Eigen::Map<const occ::Mat> tmp(buffer.get(), dims[0], dims[1]);
                if (tmp.norm() >= threshold) {
                    num_significant_shellpairs++;
                    plist.push_back(q);
                }
            }
        }
        occ::log::debug("significant shellpairs = {} ({} total)",
                        num_significant_shellpairs, num_total_shellpairs);
    }

    Vec multipole(int order, const MolecularOrbitals &mo,
                  const Vec3 &origin = {0, 0, 0}) const;

    Mat schwarz() const;

    inline bool is_spherical() const {
        return m_aobasis.kind() == Shell::Kind::Spherical;
    }

    void set_effective_core_potentials(const ShellList &ecp_shells,
                                       const std::vector<int> &ecp_electrons);

    inline double range_separated_omega() const {
        return m_env.range_separated_omega();
    }

    inline void set_range_separated_omega(double omega) {
        m_env.set_range_separated_omega(omega);
    }

    inline void set_precision(double precision) { m_precision = precision; }

  private:
    double m_precision{1e-12};
    AOBasis m_aobasis, m_auxbasis;
    ShellPairList m_shellpairs;
    // TODO remove mutable
    mutable IntEnv m_env;

    bool m_have_ecp{false};
    std::vector<libecpint::GaussianShell> m_ecp_gaussian_shells;
    std::vector<libecpint::GaussianShell> m_ecp_aux_gaussian_shells;
    std::vector<libecpint::ECP>
        m_ecp; // might need to track center info for derivatives
    int m_ecp_ao_max_l{0};
    int m_ecp_max_l{0};

    inline size_t buffer_size_1e(const Op op = Op::overlap) const {
        auto bufsize = m_aobasis.max_shell_size() * m_aobasis.max_shell_size();
        switch (op) {
        case Op::dipole:
            bufsize *= occ::core::num_unique_multipole_components(1);
            break;
        case Op::quadrupole:
            bufsize *= occ::core::num_unique_multipole_components(2);
            break;
        case Op::octapole:
            bufsize *= occ::core::num_unique_multipole_components(3);
            break;
        case Op::hexadecapole:
            bufsize *= occ::core::num_unique_multipole_components(4);
            break;
        default:
            break;
        }
        return bufsize;
    }

    inline size_t buffer_size_3e() const {
        return m_auxbasis.max_shell_size() * buffer_size_1e();
    }

    inline size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};

} // namespace occ::qm
