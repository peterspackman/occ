#pragma once
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/point_charge.h>
#include <string>
#include <string_view>
#include <vector>

namespace occ::qm {

using PointChargeList = std::vector<occ::core::PointCharge>;

/// Concept for an external-potential model: a self-contained engine that
/// produces a one-electron potential matrix `V_ext` and the nuclear–external
/// interaction energy when handed an SCF procedure (HF/DFT/etc.).
///
/// Models are independent of `SCF<Proc>` — callers can use them on the side
/// (e.g. for diagnostics) or feed them into `SCF::set_external_potential`.
template <typename T, typename Proc>
concept ExternalPotential = requires(const T &model, Proc &proc) {
  { model.compute_potential_matrix(proc) } -> std::convertible_to<Mat>;
  { model.nuclear_interaction_energy(proc) } -> std::convertible_to<double>;
  { model.label() } -> std::convertible_to<std::string_view>;
  { model.descriptor() } -> std::convertible_to<std::string>;
};

/// Plain external point charges (vacuum Coulomb). The Proc must expose
/// `compute_point_charge_interaction_matrix` and
/// `nuclear_point_charge_interaction_energy`.
struct PointChargePotential {
  PointChargeList charges;

  template <typename Proc>
  Mat compute_potential_matrix(Proc &proc) const {
    return proc.compute_point_charge_interaction_matrix(charges);
  }

  template <typename Proc>
  double nuclear_interaction_energy(const Proc &proc) const {
    return proc.nuclear_point_charge_interaction_energy(charges);
  }

  /// Energy-key suffix used when SCF reports this contribution
  /// (`nuclear.<label>` / `electronic.<label>`). Singular form preserves
  /// continuity with the pre-existing SCF point-charge plumbing.
  std::string_view label() const { return "point_charge"; }

  std::string descriptor() const {
    return fmt::format("PointCharges({})", charges.size());
  }
};

/// Wolf-summed periodic point charges. Requires Proc to expose
/// `compute_wolf_interaction_matrix` and `wolf_point_charge_interaction_energy`
/// (HartreeFock and DFT both do).
///
/// `alpha` is the Wolf damping parameter in 1/Å; `cutoff` is the real-space
/// cutoff in Å. `molecular_charges` are the per-atom partial charges of the
/// embedded molecule (length = N_atoms) used in the self-interaction term.
struct WolfPointChargePotential {
  PointChargeList charges;
  std::vector<double> molecular_charges;
  double alpha{0.2};
  double cutoff{16.0};

  template <typename Proc>
  Mat compute_potential_matrix(Proc &proc) const {
    return proc.compute_wolf_interaction_matrix(charges, molecular_charges,
                                                alpha, cutoff);
  }

  template <typename Proc>
  double nuclear_interaction_energy(const Proc &proc) const {
    return proc.wolf_point_charge_interaction_energy(charges, molecular_charges,
                                                     alpha, cutoff);
  }

  std::string_view label() const { return "wolf_potential"; }

  std::string descriptor() const {
    return fmt::format("Wolf(α={:.3f},rc={:.1f})", alpha, cutoff);
  }
};

} // namespace occ::qm
