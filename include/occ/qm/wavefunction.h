#pragma once
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/io/fchkreader.h>
#include <occ/qm/io/fchkwriter.h>
#include <occ/qm/io/moldenreader.h>
#include <occ/qm/io/orca_json.h>
#include <occ/qm/mo.h>
#include <occ/gto/shell.h>
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
  inline std::string to_string() const {
    return fmt::format("{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n"
                       "{:10s} {:20.12f}\n",
                       "E_coul", coulomb, "E_ex", exchange, "E_nn",
                       nuclear_repulsion, "E_en", nuclear_attraction, "E_kin",
                       kinetic, "E_1e", core, "E_ecp", ecp);
  }
};

struct Wavefunction {
  Wavefunction() {}

  Wavefunction(const FchkReader &);
  Wavefunction(const MoldenReader &);
  Wavefunction(const OrcaJSONReader &);
  Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b);

  static Wavefunction load(const std::string &);
  static bool is_likely_wavefunction_filename(const std::string &);

  inline int multiplicity() const { return abs(n_beta() - n_alpha()) + 1; }
  inline int charge() const {
    size_t c = 0;
    for (const auto &atom : atoms)
      c += atom.atomic_number;
    c -= num_electrons;
    c -= basis.total_ecp_electrons();
    return c;
  }
  Vec nuclear_charges() const { return  nuclear_charges(atomic_numbers());}
  Vec nuclear_charges(const IVec& nums) const {
    Vec atomic_prop = nums.cast<double>();
    if (basis.have_ecps()) {
      // set nuclear charges to include ecp
      const auto &ecp_electrons = basis.ecp_electrons();
      for (int i = 0; i < atoms.size(); i++) {
        atomic_prop(i) -= ecp_electrons[i];
      }
    }
    return atomic_prop;
  }
  inline int n_alpha() const { return mo.n_alpha; }
  inline int n_beta() const { return mo.n_beta; }
  bool is_restricted() const { return mo.kind == SpinorbitalKind::Restricted; }
  void set_molecular_orbitals(const FchkReader &fchk);
  void compute_density_matrix();
  void symmetric_orthonormalize_molecular_orbitals(const Mat &overlap);
  void apply_transformation(const occ::Mat3 &, const occ::Vec3 &);
  void apply_rotation(const occ::Mat3 &);
  void apply_translation(const occ::Vec3 &);

  occ::Mat3N positions() const;
  occ::IVec atomic_numbers() const;

  Vec mulliken_charges() const;
  /// Löwdin charges, from the diagonal of S^½ P S^½.
  Vec lowdin_charges() const;
  /// Mayer bond orders B_AB = 2 Σ_{μ∈A,ν∈B} Σ_σ (P^σ S)_μν (P^σ S)_νμ,
  /// atoms × atoms with a zero diagonal. Restricted and unrestricted only.
  Mat mayer_bond_orders() const;
  Mat overlap_matrix() const;

  Vec electron_density(const Mat3N &points,
                       SpinComponent spin = SpinComponent::Total) const;
  Mat3N electron_density_gradient(const Mat3N &points) const;

  Vec electron_density_mo(const Mat3N &points, int mo_index) const;

  /**
   * Signed amplitude psi_i(r) of MO i (0-based) at each point, in bohr^-3/2.
   * The overall sign of an orbital is arbitrary. For unrestricted
   * wavefunctions spin must be Alpha or Beta.
   */
  Vec mo_amplitude(const Mat3N &points, int mo_index,
                   SpinComponent spin = SpinComponent::Total) const;
  Mat3N electron_density_mo_gradient(const Mat3N &points, int mo_index) const;

  /// ESP at `points` (bohr). For a GFN2-xTB wavefunction this is GFN2's own
  /// electrostatic model, see xtb_electric_potential.
  Vec electric_potential(const Mat3N &points) const;
  /// Undamped point-multipole ESP from the xTB atomic charges and, when
  /// present, the CAMM dipoles and quadrupoles. Requires have_xtb_data.
  Vec xtb_electric_potential(const Mat3N &points) const;
  Mat3N electric_field(const Mat3N &points) const;

  void save(FchkWriter &);
  bool save(const std::string &filename);

  std::string method{"SCF"};
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

  // GFN2-xTB result breakdown — preserved through .owf.json round-trip so
  // downstream tools can read back the SCC / repulsion / dispersion split,
  // the xtb-convention atomic charges, and the SCC convergence diagnostic
  // without re-running the calculator. `wfn.energy.total` always carries
  // the sum; these fields only carry the decomposition.
  bool have_xtb_data{false};
  double xtb_scc_energy{0.0};
  double xtb_repulsion_energy{0.0};
  double xtb_dispersion_energy{0.0};
  Vec xtb_atomic_charges;
  // CAMM atomic dipoles (3 × N) and traceless quadrupoles (6 × N,
  // [xx, xy, yy, xz, yz, zz]); empty without multipoles. See XtbResult.
  Mat3N xtb_atomic_dipoles;
  Mat xtb_atomic_quadrupoles;
  bool xtb_converged{false};
  int xtb_n_iterations{0};

  // Post-HF (MP2 / CCSD / CCSD(T)) breakdown, likewise preserved through
  // .owf.json. `wfn.energy.total` carries the correlated total; these keep
  // the reference SCF energy and the correlation split. Components that do
  // not apply to the method run stay zero.
  bool have_correlation_data{false};
  double scf_reference_energy{0.0};
  double correlation_energy{0.0}; // as in energy.total: incl. scaling and (T)
  double mp2_same_spin_energy{0.0};
  double mp2_opposite_spin_energy{0.0};
  double ccsd_correlation_energy{0.0}; // excl. (T)
  double triples_correction{0.0};

  /// Whether the SCF that produced these orbitals converged. Anything
  /// derived from an unconverged density is meaningless.
  bool converged{true};
};

} // namespace occ::qm
