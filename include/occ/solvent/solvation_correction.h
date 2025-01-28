#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/energy_components.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/parameters.h>

namespace occ::solvent {

using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using PointChargeList = std::vector<occ::core::PointCharge>;

class ContinuumSolvationModel {
public:
  ContinuumSolvationModel(const std::vector<occ::core::Atom> &,
                          const std::string &solvent = "water",
                          double charge = 0.0, bool draco = false);

  void set_solvent(const std::string &);
  const std::string &solvent() const { return m_solvent_name; }

  const Mat3N &nuclear_positions() const { return m_nuclear_positions; }
  const Mat3N &surface_positions_coulomb() const {
    return m_surface_positions_coulomb;
  }
  const Mat3N &surface_positions_cds() const { return m_surface_positions_cds; }
  const Vec &surface_areas_coulomb() const { return m_surface_areas_coulomb; }
  const Vec &surface_areas_cds() const { return m_surface_areas_cds; }
  const Vec &nuclear_charges() const { return m_nuclear_charges; }
  size_t num_surface_points() const { return m_surface_areas_coulomb.rows(); }
  void set_surface_potential(const Vec &);
  const Vec &apparent_surface_charge();

  double surface_polarization_energy();
  double surface_charge() const { return m_asc.array().sum(); }
  double smd_cds_energy() const;

  inline double charge() const { return m_charge; }
  inline void set_charge(double charge) {
    m_charge = charge;
    initialize_surfaces();
  }

  Vec surface_cds_energy_elements() const;
  Vec surface_polarization_energy_elements() const;

  template <typename Proc>
  Vec surface_nuclear_energy_elements(const Proc &proc) const {
    Vec qn = proc.nuclear_electric_potential_contribution(
        m_surface_positions_coulomb);
    qn.array() *= m_asc.array();
    return qn;
  }

  template <typename Proc>
  Vec surface_electronic_energy_elements(const MolecularOrbitals &mo,
                                         const Proc &p) const {
    Vec result(m_surface_areas_coulomb.rows());
    Mat X;
    std::vector<core::PointCharge> point_charges;
    point_charges.emplace_back(0, 0.0, 0.0, 0.0);
    for (int i = 0; i < m_surface_areas_coulomb.rows(); i++) {
      point_charges[0].set_charge(m_asc(i));
      point_charges[0].set_position(m_surface_positions_coulomb.col(i));

      X = p.compute_point_charge_interaction_matrix(point_charges);
      switch (mo.kind) {
      case SpinorbitalKind::Restricted: {
        result(i) =
            2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, X);
        break;
      }
      case SpinorbitalKind::Unrestricted: {
        result(i) =
            2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, X);
        break;
      }
      case SpinorbitalKind::General: {
        result(i) = 2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, X);
        break;
      }
      }
    }
    return result;
  }

  void write_surface_file(const std::string &filename);
  inline std::string name() const {
    return fmt::format("SMD(solvent='{}')", m_solvent_name);
  }

private:
  void initialize_surfaces();
  void update_radii();

  double m_charge{0.0};
  Vec m_coulomb_radii;
  Vec m_cds_radii;
  Vec m_atomic_charges;

  std::string m_solvent_name;
  Mat3N m_nuclear_positions;
  Vec m_nuclear_charges;
  Mat3N m_surface_positions_coulomb, m_surface_positions_cds;
  Vec m_surface_areas_coulomb, m_surface_areas_cds;
  Vec m_surface_potential;
  Vec m_asc;
  IVec m_surface_atoms_coulomb, m_surface_atoms_cds;
  bool m_asc_needs_update{true};
  SMDSolventParameters m_params;

  COSMO m_cosmo;
  bool m_scale_radii{false};
};

template <typename Proc> class SolvationCorrectedProcedure {
public:
  SolvationCorrectedProcedure(Proc &proc, const std::string &solvent = "water",
                              bool radii_scaling = false)
      : m_atoms(proc.atoms()), m_proc(proc),
        m_solvation_model(proc.atoms(), solvent, m_proc.system_charge(),
                          radii_scaling) {
    occ::Mat3N pos(3, m_atoms.size());
    occ::IVec nums(m_atoms.size());
    for (int i = 0; i < m_atoms.size(); i++) {
      pos(0, i) = m_atoms[i].x;
      pos(1, i) = m_atoms[i].y;
      pos(2, i) = m_atoms[i].z;
      nums(i) = m_atoms[i].atomic_number;
    }
    m_qn = m_proc.nuclear_electric_potential_contribution(
        m_solvation_model.surface_positions_coulomb());
    m_point_charges.reserve(m_qn.rows());

    for (int i = 0; i < m_solvation_model.num_surface_points(); i++) {
      const auto &pt = m_solvation_model.surface_positions_coulomb().col(i);
      m_point_charges.emplace_back(0.0, pt);
    }

    m_cds_solvation_energy = m_solvation_model.smd_cds_energy();
  }

  bool supports_incremental_fock_build() const {
    return m_proc.supports_incremental_fock_build();
  }
  inline bool have_effective_core_potentials() const {
    return m_proc.have_effective_core_potentials();
  }
  inline const auto &atoms() const { return m_proc.atoms(); }
  inline const auto &aobasis() const { return m_proc.aobasis(); }
  inline auto nbf() const { return m_proc.nbf(); }

  inline Vec3 center_of_mass() const { return m_proc.center_of_mass(); }

  void set_system_charge(int charge) {
    m_proc.set_system_charge(charge);
    m_solvation_model.set_charge(charge);
  }

  inline void set_precision(double precision) {
    m_proc.set_precision(precision);
  }

  inline int system_charge() const { return m_proc.system_charge(); }

  int total_electrons() const { return m_proc.total_electrons(); }
  int active_electrons() const { return m_proc.active_electrons(); }
  inline const auto &frozen_electrons() const {
    return m_proc.frozen_electrons();
  }

  bool usual_scf_energy() const { return m_proc.usual_scf_energy(); }

  double nuclear_repulsion_energy() const {
    return m_proc.nuclear_repulsion_energy();
  }

  inline double
  nuclear_point_charge_interaction_energy(const PointChargeList &pc) const {
    return m_proc.nuclear_point_charge_interaction_energy(pc);
  }

  void update_scf_energy(occ::core::EnergyComponents &energy,
                         bool incremental) const {

    m_proc.update_scf_energy(energy, incremental);
    if (incremental) {
      energy["solvation.electronic"] += m_electronic_solvation_energy;
      energy["solvation.surface"] += m_surface_solvation_energy;
      energy["solvation.nuclear"] += m_nuclear_solvation_energy;
    } else {
      energy["solvation.electronic"] = m_electronic_solvation_energy;
      energy["solvation.nuclear"] = m_nuclear_solvation_energy;
      energy["solvation.surface"] = m_surface_solvation_energy;
      energy["solvation.CDS"] = m_cds_solvation_energy;
    }
    energy["total"] = energy["electronic"] + energy["nuclear.repulsion"] +
                      energy["solvation.nuclear"] +
                      energy["solvation.surface"] + energy["solvation.CDS"];
  }

  auto compute_kinetic_matrix() const {
    return m_proc.compute_kinetic_matrix();
  }

  auto compute_overlap_matrix() const {
    return m_proc.compute_overlap_matrix();
  }

  auto compute_overlap_matrix_for_basis(const occ::qm::AOBasis &bs) const {
    return m_proc.compute_overlap_matrix_for_basis(bs);
  }

  auto compute_nuclear_attraction_matrix() const {
    return m_proc.compute_nuclear_attraction_matrix();
  }

  auto compute_effective_core_potential_matrix() const {
    return m_proc.compute_effective_core_potential_matrix();
  }

  auto compute_schwarz_ints() const { return m_proc.compute_schwarz_ints(); }

  inline auto
  compute_point_charge_interaction_matrix(const PointChargeList &pc) {
    return m_proc.compute_point_charge_interaction_matrix(pc);
  }

  void update_core_hamiltonian(const MolecularOrbitals &mo, occ::Mat &H) {
    occ::timing::start(occ::timing::category::solvent);
    occ::Vec v =
        (m_qn + m_proc.electronic_electric_potential_contribution(
                    mo, m_solvation_model.surface_positions_coulomb()));
    m_solvation_model.set_surface_potential(v);
    auto asc = m_solvation_model.apparent_surface_charge();
    for (int i = 0; i < m_point_charges.size(); i++) {
      m_point_charges[i].set_charge(asc(i));
    }
    double surface_energy = m_solvation_model.surface_polarization_energy();
    m_nuclear_solvation_energy = m_qn.dot(asc);
    m_surface_solvation_energy = surface_energy;
    m_electronic_solvation_energy = 0.0;
    occ::log::debug("PCM surface polarization energy: {:.12f}", surface_energy);
    occ::log::debug("PCM surface charge: {:.12f}",
                    m_solvation_model.surface_charge());
    m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);

    switch (mo.kind) {
    case SpinorbitalKind::Restricted: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H);
      H += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    case SpinorbitalKind::Unrestricted: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H);
      occ::qm::block::a(H) += m_X;
      occ::qm::block::b(H) += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    case SpinorbitalKind::General: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H);
      occ::qm::block::aa(H) += m_X;
      occ::qm::block::bb(H) += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    }
    occ::timing::stop(occ::timing::category::solvent);
  }

  Mat compute_fock(const MolecularOrbitals &mo,
                   const Mat &Schwarz = Mat()) const {
    return m_proc.compute_fock(mo, Schwarz);
  }

  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs,
                               const qm::AOBasis &bs, bool is_shell_diagonal) {
    return m_proc.compute_fock_mixed_basis(mo_bs, bs, is_shell_diagonal);
  }

  void set_solvent(const std::string &solvent) {
    m_solvation_model.set_solvent(solvent);
  }

  void write_surface_file(const std::string &filename) {
    m_solvation_model.write_surface_file(filename);
  }

  auto surface_positions_coulomb() const {
    return m_solvation_model.surface_positions_coulomb();
  }
  auto surface_positions_cds() const {
    return m_solvation_model.surface_positions_cds();
  }
  auto surface_areas_coulomb() const {
    return m_solvation_model.surface_areas_coulomb();
  }
  auto surface_areas_cds() const {
    return m_solvation_model.surface_areas_cds();
  }

  auto surface_cds_energy_elements() const {
    return m_solvation_model.surface_cds_energy_elements();
  }
  auto surface_polarization_energy_elements() const {
    return m_solvation_model.surface_polarization_energy_elements();
  }

  auto surface_nuclear_energy_elements() const {
    return m_solvation_model.surface_nuclear_energy_elements(m_proc);
  }
  auto surface_electronic_energy_elements(const MolecularOrbitals &mo) const {
    return m_solvation_model.surface_electronic_energy_elements(mo, m_proc);
  }

  template <unsigned int order = 1>
  inline auto
  compute_electronic_multipole_matrices(const Vec3 &o = {0.0, 0.0, 0.0}) const {
    return m_proc.template compute_electronic_multipole_matrices<order>(o);
  }

  template <unsigned int order = 1>
  inline auto compute_electronic_multipoles(const MolecularOrbitals &mo,
                                            const Vec3 &o = {0.0, 0.0,
                                                             0.0}) const {
    return m_proc.template compute_electronic_multipoles<order>(mo, o);
  }

  template <unsigned int order = 1>
  inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0,
                                                          0.0}) const {
    return m_proc.template compute_nuclear_multipoles<order>(o);
  }

  inline std::string name() const {
    return fmt::format("{}+{}", m_proc.name(), m_solvation_model.name());
  }

private:
  const std::string m_solvent_name{"water"};
  const std::vector<occ::core::Atom> &m_atoms;
  Proc &m_proc;
  ContinuumSolvationModel m_solvation_model;
  std::vector<core::PointCharge> m_point_charges;
  double m_electronic_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0},
      m_surface_solvation_energy, m_cds_solvation_energy;
  Mat m_X;
  Vec m_qn;
};

} // namespace occ::solvent
