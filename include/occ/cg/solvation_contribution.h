#pragma once
#include <utility>

namespace occ::cg {

struct ContributionPair {
  double forward{0.0};
  double reverse{0.0};

  [[nodiscard]] double total() const;
  void exchange_with(ContributionPair &other);
};

class SolvationContribution {
public:
  SolvationContribution() = default;

  void add_coulomb(double value, bool is_forward = true);
  void add_cds(double value, bool is_forward = true);
  void add_coulomb_area(double value, bool is_forward = true);
  void add_cds_area(double value, bool is_forward = true);

  [[nodiscard]] double total_energy() const;
  void exchange_with(SolvationContribution &other);
  [[nodiscard]] bool has_been_exchanged() const { return m_exchanged; }

  // Getters for testing and inspection
  [[nodiscard]] const ContributionPair &coulomb() const { return m_coulomb; }
  [[nodiscard]] const ContributionPair &cds() const { return m_cds; }

  [[nodiscard]] const ContributionPair &coulomb_area() const {
    return m_coulomb_area;
  }
  [[nodiscard]] const ContributionPair &cds_area() const { return m_cds_area; }

  [[nodiscard]] bool antisymmetrize() const { return m_antisymmetrize; }
  void set_antisymmetrize(bool on) { m_antisymmetrize = on; }

private:
  ContributionPair m_coulomb;
  ContributionPair m_cds;
  ContributionPair m_coulomb_area;
  ContributionPair m_cds_area;
  bool m_exchanged{false};
  bool m_antisymmetrize{true};
};

} // namespace occ::cg
