#pragma once
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/io/orca_json.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::Vec;
using occ::io::FchkReader;
using occ::io::FchkWriter;
using occ::io::MoldenReader;
using occ::io::OrcaJSONReader;

struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
    double total{0};
    double ecp{0};

    inline auto operator-(const Energy &rhs) {
        return Energy{
            coulomb - rhs.coulomb,
            exchange - rhs.exchange,
            nuclear_repulsion - rhs.nuclear_repulsion,
            nuclear_attraction - rhs.nuclear_attraction,
            kinetic - rhs.kinetic,
            core - rhs.core,
            total - rhs.total,
            ecp - rhs.ecp,
        };
    }

    inline auto operator+(const Energy &rhs) {
        return Energy{
            coulomb + rhs.coulomb,
            exchange + rhs.exchange,
            nuclear_repulsion + rhs.nuclear_repulsion,
            nuclear_attraction + rhs.nuclear_attraction,
            kinetic + rhs.kinetic,
            core + rhs.core,
            total + rhs.total,
            ecp + rhs.ecp,
        };
    }
};

struct Wavefunction {
    Wavefunction() {}

    Wavefunction(const FchkReader &);
    Wavefunction(const MoldenReader &);
    Wavefunction(const OrcaJSONReader &);
    Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b);

    inline int multiplicity() const { return abs(num_beta - num_alpha) + 1; }
    inline int charge() const {
        size_t c = 0;
        for (const auto &atom : atoms)
            c += atom.atomic_number;
        c -= num_electrons;
        return c;
    }
    inline int n_alpha() const { return num_alpha; }
    inline int n_beta() const { return num_beta; }
    bool is_restricted() const {
        return spinorbital_kind == SpinorbitalKind::Restricted;
    }
    void update_occupied_orbitals();
    void set_molecular_orbitals(const FchkReader &fchk);
    void compute_density_matrix();
    void symmetric_orthonormalize_molecular_orbitals(const Mat &overlap);
    void apply_transformation(const occ::Mat3 &, const occ::Vec3 &);
    void apply_rotation(const occ::Mat3 &);
    void apply_translation(const occ::Vec3 &);

    occ::Mat3N positions() const;
    occ::IVec atomic_numbers() const;

    Vec mulliken_charges() const;

    Vec electric_potential(const Mat3N &points) const;
    Mat3N electric_field(const Mat3N &points) const;

    void save(FchkWriter &);

    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int num_alpha{0};
    int num_beta{0};
    int num_electrons{0};
    int num_frozen_electrons{0};
    AOBasis basis;
    size_t nbf{0};
    std::vector<occ::core::Atom> atoms;
    MolecularOrbitals mo;
    Mat T, V, H, J, K, Vecp;
    Energy energy;
    bool have_energies{false};
    bool have_xdm_parameters{false};
    Vec xdm_polarizabilities;
    Mat xdm_moments;
    Vec xdm_volumes;
    Vec xdm_free_volumes;
    double xdm_energy{0.0};
};

} // namespace occ::qm

template <> struct fmt::formatter<occ::qm::Energy> {
    char presentation{'f'};

    constexpr auto parse(format_parse_context &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 'f' || *it == 'e'))
            presentation = *it++;

        if (it != end && *it != '}')
            throw format_error("invalid format");

        return it;
    }

    template <typename FormatContext>
    auto format(const occ::qm::Energy &e, FormatContext &ctx) {
        auto fmt_string = fmt::format("{{:32s}} {{:20.12{}}}\n", presentation);
        auto result = format_to(ctx.out(), fmt_string, "E_coul", e.coulomb);
        result = format_to(ctx.out(), fmt_string, "E_ex", e.exchange);
        result = format_to(ctx.out(), fmt_string, "E_nn", e.nuclear_repulsion);
        result = format_to(ctx.out(), fmt_string, "E_en", e.nuclear_attraction);
        result = format_to(ctx.out(), fmt_string, "E_kin", e.kinetic);
        result = format_to(ctx.out(), fmt_string, "E_1e", e.core);
        if (e.ecp != 0.0)
            result = format_to(ctx.out(), fmt_string, "E_ecp", e.ecp);
        return result;
    }
};
