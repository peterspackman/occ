#pragma once
#include <ankerl/unordered_dense.h>
#include <string>
#include <string_view>
#include <vector>

namespace occ::cg {

struct ContributionPair {
  double forward{0.0};
  double reverse{0.0};

  [[nodiscard]] double total() const;
  void exchange_with(ContributionPair &other);
};

/// Solvation attributed to one neighbour contact, as a set of named channels.
///
/// Channels are open-ended so a solvation model can carry whatever it
/// produces: SMD uses `electrostatic` and `cds`, the σ-potential model uses
/// `dielectric` and `residual` plus descriptors like `hbond_area`. Adding a
/// channel needs no change here, in the partitioner, or in the exchange
/// logic.
///
/// Energies are summed into `total_energy()`; descriptors (areas, per-contact
/// diagnostics) are carried and exchanged the same way but never summed into
/// an energy.
class SolvationContribution {
public:
  SolvationContribution() = default;

  void add_energy(std::string_view channel, double value,
                  bool is_forward = true);
  void add_descriptor(std::string_view channel, double value,
                      bool is_forward = true);

  /// Zero pair when the channel was never written.
  [[nodiscard]] const ContributionPair &energy(std::string_view channel) const;
  [[nodiscard]] const ContributionPair &
  descriptor(std::string_view channel) const;

  [[nodiscard]] std::vector<std::string> energy_channels() const;
  [[nodiscard]] std::vector<std::string> descriptor_channels() const;

  /// Sum over every energy channel, with the antisymmetric correction when
  /// enabled.
  [[nodiscard]] double total_energy() const;

  void exchange_with(SolvationContribution &other);
  [[nodiscard]] bool has_been_exchanged() const { return m_exchanged; }

  [[nodiscard]] bool antisymmetrize() const { return m_antisymmetrize; }
  void set_antisymmetrize(bool on) { m_antisymmetrize = on; }

  // Named accessors for the SMD channels, kept so existing callers and tests
  // read unchanged.
  void add_coulomb(double value, bool is_forward = true) {
    add_energy("coulomb", value, is_forward);
  }
  void add_cds(double value, bool is_forward = true) {
    add_energy("cds", value, is_forward);
  }
  void add_coulomb_area(double value, bool is_forward = true) {
    add_descriptor("coulomb_area", value, is_forward);
  }
  void add_cds_area(double value, bool is_forward = true) {
    add_descriptor("cds_area", value, is_forward);
  }
  [[nodiscard]] const ContributionPair &coulomb() const {
    return energy("coulomb");
  }
  [[nodiscard]] const ContributionPair &cds() const { return energy("cds"); }
  [[nodiscard]] const ContributionPair &coulomb_area() const {
    return descriptor("coulomb_area");
  }
  [[nodiscard]] const ContributionPair &cds_area() const {
    return descriptor("cds_area");
  }

private:
  using ChannelMap = ankerl::unordered_dense::map<std::string, ContributionPair>;
  ChannelMap m_energies;
  ChannelMap m_descriptors;
  bool m_exchanged{false};
  bool m_antisymmetrize{true};
};

} // namespace occ::cg
