#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/log.h>
#include <occ/core/point_charge.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <vector>

namespace occ::qm {

using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using PointChargeList = std::vector<occ::core::PointCharge>;

template <typename Proc> class PointChargeCorrectedProcedure {
public:
  PointChargeCorrectedProcedure(Proc &proc, const PointChargeList &point_charges)
      : m_proc(proc), m_point_charges(point_charges) {
    // Precompute nuclear-point charge interaction energy
    m_nuclear_external_energy = m_proc.nuclear_point_charge_interaction_energy(m_point_charges);
    occ::log::info("Point charge correction initialized with {} charges", m_point_charges.size());
    occ::log::info("Nuclear-point charge interaction energy: {:.8f}", m_nuclear_external_energy);
  }

  // Forward all basic procedure methods
  bool supports_incremental_fock_build() const { return m_proc.supports_incremental_fock_build(); }
  bool have_effective_core_potentials() const { return m_proc.have_effective_core_potentials(); }
  bool usual_scf_energy() const { return m_proc.usual_scf_energy(); }
  
  const auto &atoms() const { return m_proc.atoms(); }
  const auto &aobasis() const { return m_proc.aobasis(); }
  auto nbf() const { return m_proc.nbf(); }
  
  int total_electrons() const { return m_proc.total_electrons(); }
  int active_electrons() const { return m_proc.active_electrons(); }
  
  double nuclear_repulsion_energy() const { return m_proc.nuclear_repulsion_energy(); }
  
  void set_precision(double precision) { m_proc.set_precision(precision); }
  
  // Integral computation methods
  auto compute_kinetic_matrix() const { return m_proc.compute_kinetic_matrix(); }
  auto compute_overlap_matrix() const { return m_proc.compute_overlap_matrix(); }
  auto compute_overlap_matrix_for_basis(const occ::qm::AOBasis &bs) const {
    return m_proc.compute_overlap_matrix_for_basis(bs);
  }
  auto compute_nuclear_attraction_matrix() const { return m_proc.compute_nuclear_attraction_matrix(); }
  auto compute_effective_core_potential_matrix() const { return m_proc.compute_effective_core_potential_matrix(); }
  auto compute_schwarz_ints() const { return m_proc.compute_schwarz_ints(); }
  
  auto compute_point_charge_interaction_matrix(const PointChargeList &pc) const {
    return m_proc.compute_point_charge_interaction_matrix(pc);
  }
  
  double nuclear_point_charge_interaction_energy(const PointChargeList &pc) const {
    return m_proc.nuclear_point_charge_interaction_energy(pc);
  }
  
  Mat compute_fock(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) const {
    return m_proc.compute_fock(mo, Schwarz);
  }
  
  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs, const occ::qm::AOBasis &bs, bool is_shell_diagonal) {
    return m_proc.compute_fock_mixed_basis(mo_bs, bs, is_shell_diagonal);
  }
  
  // Core Hamiltonian update - add point charge potential
  void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) {
    // Add point charge interaction matrix to Hamiltonian
    Mat V_ext = m_proc.compute_point_charge_interaction_matrix(m_point_charges);
    
    switch (mo.kind) {
    case SpinorbitalKind::Restricted: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H);
      H += V_ext;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    case SpinorbitalKind::Unrestricted: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H);
      occ::qm::block::a(H) += V_ext;
      occ::qm::block::b(H) += V_ext;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    case SpinorbitalKind::General: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H);
      occ::qm::block::aa(H) += V_ext;
      occ::qm::block::bb(H) += V_ext;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    }
  }
  
  // Energy update - add external potential contributions
  void update_scf_energy(occ::core::EnergyComponents &energy, bool incremental) const {
    m_proc.update_scf_energy(energy, incremental);
    
    if (incremental) {
      energy["electronic.point_charges"] += m_electronic_external_energy;
      energy["nuclear.point_charges"] += m_nuclear_external_energy;
    } else {
      energy["electronic.point_charges"] = m_electronic_external_energy;
      energy["nuclear.point_charges"] = m_nuclear_external_energy;
    }
    
    energy["total"] = energy["electronic"] + energy["nuclear.repulsion"] + 
                      energy["nuclear.point_charges"];
  }
  
  std::string name() const {
    return fmt::format("{}+PointCharges({})", m_proc.name(), m_point_charges.size());
  }

private:
  Proc &m_proc;
  PointChargeList m_point_charges;
  double m_nuclear_external_energy{0.0};
  mutable double m_electronic_external_energy{0.0};
};

template <typename Proc> class WolfSumCorrectedProcedure {
public:
  WolfSumCorrectedProcedure(Proc &proc, const PointChargeList &point_charges,
                           const std::vector<double> &molecular_charges,
                           double alpha, double cutoff)
      : m_proc(proc), m_point_charges(point_charges), m_molecular_charges(molecular_charges),
        m_alpha(alpha), m_cutoff(cutoff) {
    // Precompute nuclear-Wolf interaction energy
    m_nuclear_external_energy = m_proc.wolf_point_charge_interaction_energy(
        m_point_charges, m_molecular_charges, m_alpha, m_cutoff);
    occ::log::info("Wolf sum correction initialized with {} charges", m_point_charges.size());
    occ::log::info("Wolf parameters: alpha={:.4f}, cutoff={:.2f}", m_alpha, m_cutoff);
    occ::log::info("Nuclear-Wolf interaction energy: {:.8f}", m_nuclear_external_energy);
  }

  // Forward all basic procedure methods (same as PointChargeCorrectedProcedure)
  bool supports_incremental_fock_build() const { return m_proc.supports_incremental_fock_build(); }
  bool have_effective_core_potentials() const { return m_proc.have_effective_core_potentials(); }
  bool usual_scf_energy() const { return m_proc.usual_scf_energy(); }
  
  const auto &atoms() const { return m_proc.atoms(); }
  const auto &aobasis() const { return m_proc.aobasis(); }
  auto nbf() const { return m_proc.nbf(); }
  
  int total_electrons() const { return m_proc.total_electrons(); }
  int active_electrons() const { return m_proc.active_electrons(); }
  
  double nuclear_repulsion_energy() const { return m_proc.nuclear_repulsion_energy(); }
  
  void set_precision(double precision) { m_proc.set_precision(precision); }
  
  // Integral computation methods
  auto compute_kinetic_matrix() const { return m_proc.compute_kinetic_matrix(); }
  auto compute_overlap_matrix() const { return m_proc.compute_overlap_matrix(); }
  auto compute_overlap_matrix_for_basis(const occ::qm::AOBasis &bs) const {
    return m_proc.compute_overlap_matrix_for_basis(bs);
  }
  auto compute_nuclear_attraction_matrix() const { return m_proc.compute_nuclear_attraction_matrix(); }
  auto compute_effective_core_potential_matrix() const { return m_proc.compute_effective_core_potential_matrix(); }
  auto compute_schwarz_ints() const { return m_proc.compute_schwarz_ints(); }
  
  auto compute_point_charge_interaction_matrix(const PointChargeList &pc) const {
    return m_proc.compute_point_charge_interaction_matrix(pc);
  }
  
  double nuclear_point_charge_interaction_energy(const PointChargeList &pc) const {
    return m_proc.nuclear_point_charge_interaction_energy(pc);
  }
  
  Mat compute_fock(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) const {
    return m_proc.compute_fock(mo, Schwarz);
  }
  
  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs, const occ::qm::AOBasis &bs, bool is_shell_diagonal) {
    return m_proc.compute_fock_mixed_basis(mo_bs, bs, is_shell_diagonal);
  }
  
  // Core Hamiltonian update - add Wolf potential
  void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) {
    // Add Wolf interaction matrix to Hamiltonian
    Mat V_wolf = m_proc.compute_wolf_interaction_matrix(m_point_charges, m_molecular_charges, m_alpha, m_cutoff);
    
    switch (mo.kind) {
    case SpinorbitalKind::Restricted: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H);
      H += V_wolf;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    case SpinorbitalKind::Unrestricted: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H);
      occ::qm::block::a(H) += V_wolf;
      occ::qm::block::b(H) += V_wolf;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    case SpinorbitalKind::General: {
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H);
      occ::qm::block::aa(H) += V_wolf;
      occ::qm::block::bb(H) += V_wolf;
      m_electronic_external_energy = 2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H) - m_electronic_external_energy;
      break;
    }
    }
  }
  
  // Energy update - add Wolf potential contributions
  void update_scf_energy(occ::core::EnergyComponents &energy, bool incremental) const {
    m_proc.update_scf_energy(energy, incremental);
    
    if (incremental) {
      energy["electronic.wolf_potential"] += m_electronic_external_energy;
      energy["nuclear.wolf_potential"] += m_nuclear_external_energy;
    } else {
      energy["electronic.wolf_potential"] = m_electronic_external_energy;
      energy["nuclear.wolf_potential"] = m_nuclear_external_energy;
    }
    
    energy["total"] = energy["electronic"] + energy["nuclear.repulsion"] + 
                      energy["nuclear.wolf_potential"];
  }
  
  std::string name() const {
    return fmt::format("{}+Wolf(α={:.3f},rc={:.1f})", m_proc.name(), m_alpha, m_cutoff);
  }

private:
  Proc &m_proc;
  PointChargeList m_point_charges;
  std::vector<double> m_molecular_charges;
  double m_alpha, m_cutoff;
  double m_nuclear_external_energy{0.0};
  mutable double m_electronic_external_energy{0.0};
};

} // namespace occ::qm