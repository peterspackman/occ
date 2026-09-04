#pragma once
#include <occ/core/conditioning_orthogonalizer.h>
#include <occ/core/energy_components.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/convergence_accelerator.h>
#include <occ/qm/expectation.h>
#include <occ/qm/external_potential.h>
#include <occ/qm/initial_guess.h>
#include <occ/qm/mo.h>
#include <occ/qm/opmatrix.h>
#include <occ/qm/orthogonalizer.h>
#include <occ/qm/scf_convergence_settings.h>
#include <occ/qm/scf_method.h>
#include <occ/gto/shell.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

using qm::expectation;
using qm::SpinorbitalKind;
using qm::Wavefunction;
using qm::SpinorbitalKind::General;
using qm::SpinorbitalKind::Restricted;
using qm::SpinorbitalKind::Unrestricted;
using util::is_odd;
using PointChargeList = std::vector<occ::core::PointCharge>;

struct SCFContext {
  Mat S, T, V, H, K, F, V_ext, Vecp;
  int n_electrons{0};
  int n_frozen_electrons{0};
  int n_occ{0};
  int n_unpaired_electrons{0};
  size_t nbf{0};
  bool converged{false};
  CanonicalOrthogonalizer orthogonalizer;
  occ::core::EnergyComponents energy;
  occ::qm::MolecularOrbitals mo;

  /// Energy-key suffix for the active external-potential contribution; empty
  /// when none is set. When non-empty, SCF reports `nuclear.<label>` /
  /// `electronic.<label>` and folds them into `total`.
  std::string external_potential_label;
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

  void set_conditioning_orthogonalizer();
  void set_core_matrices();
  void set_initial_guess_from_wfn(const Wavefunction &wfn);

  /// Which guess to start from. `GuessKind::Auto`, the default, lets
  /// `select_guess` decide per system.
  void set_guess_kind(GuessKind kind) { m_guess_kind = kind; }
  GuessKind guess_kind() const { return m_guess_kind; }

  /// Build the starting orbitals: core Hamiltonian, plus whatever the
  /// selected guess contributes, diagonalised.
  void compute_initial_guess();

  /// Generic external-potential entry point. `V_ext_single` is a single
  /// `nbf x nbf` one-electron operator in the AO basis — SCF expands it
  /// into the correct spin block(s) of `ctx.V_ext`. Records
  /// `nuclear.<label>` and arranges `electronic.<label>` to be updated each
  /// iteration. Pass any matrix and energy you like; the caller's choice of
  /// `label` ends up in the final energy report.
  void set_external_potential(const Mat &V_ext_single, double nuclear_energy,
                              std::string_view label);

  /// Convenience overload: takes any `ExternalPotential` model and forwards
  /// the matrix/energy/label it produces. The model must satisfy
  /// `ExternalPotential<Model, Procedure>` from
  /// `<occ/qm/external_potential.h>`.
  template <typename Model>
    requires ExternalPotential<Model, Procedure>
  void set_external_potential(const Model &model) {
    set_external_potential(model.compute_potential_matrix(m_procedure),
                           model.nuclear_interaction_energy(m_procedure),
                           model.label());
  }

  void update_scf_energy(bool incremental);

  /// Raise the virtual orbitals by `shift` Hartree, in place.
  void apply_level_shift(Mat &F, double shift) const;
  inline const char *scf_kind() const;
  double compute_scf_energy();

  /// Whether the procedure's declared `FockBuildProperties` permit
  /// accumulating F += G(ΔD). Asked of the procedure directly rather than
  /// cached, so it can't go stale if a DF basis or COSX grid is installed
  /// after the SCF is constructed. Enforced where the accumulation starts, so
  /// `convergence_settings.incremental_fock_threshold` can only ever turn
  /// incremental builds *off* — never on for a method that can't take them.
  bool incremental_fock_supported() const {
    return supports_incremental_fock_build(m_procedure.fock_build_properties());
  }

  occ::qm::SCFConvergenceSettings convergence_settings;
  Procedure &m_procedure;
  SCFContext ctx;
  int maxiter{100};
  int iter = 0;
  double diis_error{1.0};
  double ediff_rel = 0.0;
  double total_time{0.0};
  occ::qm::ConvergenceAccelerator convergence_accelerator;
  bool reset_incremental_fock_formation{false};
  bool incremental_Fbuild_started{false};
  /// Fock builds that accumulated F += G(ΔD) rather than rebuilding from H.
  /// Zero whenever the procedure doesn't support incremental builds.
  int num_incremental_fock_builds{0};
  double next_reset_threshold{0.0};
  size_t last_reset_iteration{0};
  bool m_have_initial_guess{false};

private:
  /// Spread a guess's one-electron potential over this calculation's spin
  /// blocks and add it to `ctx.F`.
  void add_guess_potential(const Mat &potential);
  /// Build the Fock contribution of a guess density and add it to `ctx.F`.
  void add_guess_density(const Guess &guess);

  GuessKind m_guess_kind{GuessKind::Auto};
};

} // namespace occ::qm

#include <occ/qm/scf_impl.h>
