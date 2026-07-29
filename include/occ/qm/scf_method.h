#pragma once
#include <concepts>
#include <occ/core/energy_components.h>
#include <occ/core/point_charge.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;
using PointChargeList = std::vector<occ::core::PointCharge>;

/// What a method guarantees about its Fock build, so the SCF driver can decide
/// whether to accumulate `F += G(ΔD)` instead of rebuilding from H each cycle.
///
/// These are statements of fact about the method, not policy: declare what is
/// true and let `supports_incremental_fock_build` combine them. Splitting them
/// out keeps unrelated reasons from being conflated into one opaque bool —
/// implicit solvation, for instance, has a perfectly linear two-electron
/// build and still cannot be accumulated.
struct FockBuildProperties {
  /// G(D) is linear, so G(Dₙ) == G(Dₙ₋₁) + G(ΔD). False for anything built on
  /// a density-adapted quadrature — the XC functional, COSX exchange.
  bool linear_in_density{true};

  /// H is identical every cycle, so an accumulated F keeps a valid
  /// one-electron part. False when H gains a density-dependent term, as with
  /// implicit solvation's apparent surface charge.
  bool constant_core_hamiltonian{true};

  /// The two-electron build screens on the density, so handing it a small ΔD
  /// is actually cheaper. Purely about benefit: when false, accumulating is
  /// still correct but buys nothing.
  bool density_screened{true};
};

/// F can only be accumulated when the build is linear *and* the core
/// Hamiltonian is fixed; screening decides whether it is worth doing.
constexpr bool
supports_incremental_fock_build(const FockBuildProperties &properties) {
  return properties.linear_in_density &&
         properties.constant_core_hamiltonian && properties.density_screened;
}

// Core concept that defines minimum requirements for an SCF method
template <typename T>
concept SCFMethod =
    requires(T method, const MolecularOrbitals &mo, const Mat &schwarz,
             occ::core::EnergyComponents &energy) {
      { method.compute_fock(mo, schwarz) } -> std::same_as<Mat>;
      { method.compute_schwarz_ints() } -> std::same_as<Mat>;
      { method.nuclear_repulsion_energy() } -> std::same_as<double>;
      { method.update_scf_energy(energy) } -> std::same_as<void>;
      { method.nbf() } -> std::same_as<size_t>;
      { method.atoms() } -> std::same_as<const std::vector<occ::core::Atom> &>;
      { method.aobasis() } -> std::same_as<const AOBasis &>;
      { method.active_electrons() } -> std::same_as<int>;
      { method.total_electrons() } -> std::same_as<int>;
      { method.have_effective_core_potentials() } -> std::same_as<bool>;
      { method.usual_scf_energy() } -> std::same_as<bool>;
      { method.name() } -> std::same_as<std::string>;
      { method.fock_build_properties() } -> std::same_as<FockBuildProperties>;
    };

// Helper concept for methods that support density fitting
template <typename T>
concept DensityFittingMethod =
    SCFMethod<T> && requires(T method, const std::string &df_basis) {
      { method.set_density_fitting_basis(df_basis) } -> std::same_as<void>;
    };

// Helper concept for methods that support point charges
template <typename T>
concept PointChargeMethod =
    SCFMethod<T> && requires(T method, const PointChargeList &charges) {
      {
        method.nuclear_point_charge_interaction_energy(charges)
      } -> std::same_as<double>;
      {
        method.compute_point_charge_interaction_matrix(charges)
      } -> std::same_as<Mat>;
    };

// Common functionality that can be inherited/composed into SCF methods
class SCFMethodBase {
protected:
  int m_charge{0};
  int m_num_electrons{0};
  int m_num_frozen{0};
  std::vector<occ::core::Atom> m_atoms;
  std::vector<int> m_frozen_electrons;

  // Helper functions for derived classes
  void update_electron_count() {
    m_num_electrons = std::accumulate(m_atoms.begin(), m_atoms.end(), 0,
                                      [](int sum, const auto &atom) {
                                        return sum + atom.atomic_number;
                                      }) -
                      m_charge;
  }

public:
  SCFMethodBase(const std::vector<core::Atom> &);
  inline const auto &atoms() const { return m_atoms; }
  inline int system_charge() const { return m_charge; }
  inline int total_electrons() const { return m_num_electrons; }
  inline int active_electrons() const { return m_num_electrons - m_num_frozen; }
  inline const auto &frozen_electrons() const { return m_frozen_electrons; }

  Vec3 center_of_mass() const;
  void set_system_charge(int charge);
  double nuclear_repulsion_energy() const;
  Mat3N nuclear_repulsion_gradient() const;
  Vec nuclear_electric_potential_contribution(const Mat3N &) const;
  Mat3N nuclear_electric_field_contribution(const Mat3N &) const;
  void set_frozen_electrons(const std::vector<int> &);
};

// Template to check if a type meets SCF method requirements
template <typename T> constexpr bool is_scf_method_v = SCFMethod<T>;

// Template to check if a type supports density fitting
template <typename T>
constexpr bool supports_density_fitting_v = DensityFittingMethod<T>;

// Template to check if a type supports point charges
template <typename T>
constexpr bool supports_point_charges_v = PointChargeMethod<T>;

} // namespace occ::qm
