#pragma once
#include <occ/core/conditioning_orthogonalizer.h>
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/cdiis.h>
#include <occ/qm/ediis.h>
#include <occ/qm/expectation.h>
#include <occ/qm/guess_density.h>
#include <occ/qm/mo.h>
#include <occ/qm/opmatrix.h>
#include <occ/qm/scf_convergence_settings.h>
#include <occ/qm/scf_method.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

constexpr auto OCC_MINIMAL_BASIS = "sto-3g";
using qm::expectation;
using qm::SpinorbitalKind;
using qm::Wavefunction;
using qm::SpinorbitalKind::General;
using qm::SpinorbitalKind::Restricted;
using qm::SpinorbitalKind::Unrestricted;
using util::is_odd;
using PointChargeList = std::vector<occ::core::PointCharge>;

struct SCFContext {
  Mat S, T, V, H, K, X, Xinv, F, Vpc, Vecp;
  double XtX_condition_number{0.0};
  int n_electrons{0};
  int n_frozen_electrons{0};
  int n_occ{0};
  int n_unpaired_electrons{0};
  size_t nbf{0};
  bool converged{false};
  occ::core::EnergyComponents energy;
  occ::qm::MolecularOrbitals mo;
  PointChargeList point_charges;
};

template <SCFMethod Procedure> struct SCF {
  SCF(Procedure &procedure, SpinorbitalKind sk = SpinorbitalKind::Restricted);
  int n_alpha() const;
  int n_beta() const;
  int charge() const;
  int multiplicity() const;
  void set_charge(int c);
  void set_multiplicity(int m);

  Wavefunction wavefunction() const;
  void set_charge_multiplicity(int chg, unsigned int mult);
  void update_occupied_orbital_count();
  const std::vector<occ::core::Atom> &atoms() const;

  const MolecularOrbitals &molecular_orbitals() const;

  Mat compute_soad(const Mat &overlap_minbs) const;
  void set_conditioning_orthogonalizer();
  void set_core_matrices();
  void set_initial_guess_from_wfn(const Wavefunction &wfn);

  void compute_initial_guess();
  void set_point_charges(const PointChargeList &charges);

  void update_scf_energy(bool incremental);
  inline const char *scf_kind() const;
  double compute_scf_energy();

  occ::qm::SCFConvergenceSettings convergence_settings;
  Procedure &m_procedure;
  SCFContext ctx;
  int maxiter{100};
  int iter = 0;
  double diis_error{1.0};
  double ediff_rel = 0.0;
  double total_time{0.0};
  occ::qm::CDIIS diis; // start DIIS on second iteration
  occ::qm::EDIIS ediis;
  bool reset_incremental_fock_formation{false};
  bool incremental_Fbuild_started{false};
  double next_reset_threshold{0.0};
  size_t last_reset_iteration{0};
  bool m_have_initial_guess{false};
};

} // namespace occ::qm

#include <occ/qm/scf_impl.h>
