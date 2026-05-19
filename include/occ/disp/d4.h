#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::disp {

// Damping parameters for the rational Becke-Johnson form of DFT-D4.
//   E2 = -Σ_{A<B} [ s6·C6/(r^6 + r0^6) + s8·C8/(r^8 + r0^8) ]
//   r0 = a1·√(3·sqrt_zr4r2_A·sqrt_zr4r2_B) + a2
// `s9` scales the Axilrod-Teller-Muto 3-body term (set to 0 to disable).
struct D4Damping {
  double s6{1.0};
  double s8{0.0};
  double s9{0.0};
  double a1{0.4};
  double a2{5.0};
  int alp{16}; // ATM exponent
};

// Charge / coordination-number scaling for the reference projection. These are
// the parameters that appear in the ζ function of the D4 model. xtb's GFN2-xTB
// uses (3.0, 2.0, 6.0); the standard DFT-D4 also uses these defaults.
struct D4Scaling {
  double ga{3.0};
  double gc{2.0};
  double wf{6.0};
};

// Convenience: GFN2-xTB damping parameters (s6=1, s8=2.7, s9=5, a1=0.52, a2=5).
inline constexpr D4Damping gfn2_damping{1.0, 2.7, 5.0, 0.52, 5.0, 16};

// Reference-charge dataset selector — see d4_data.h for details.
enum class RefqMode { GFN2, DFT };

// Native DFT-D4 evaluator. Construct with a list of atoms (positions in Bohr,
// elements via core::Atom::atomic_number), set damping/charges, compute energy
// or energy + gradient. Reference data is loaded once from
// share/dftd4/refdata.json and cached; per-functional damping params come from
// share/dftd4/functionals.json.
//
// Two main usage patterns:
//   1. GFN2-xTB SCC-coupled D4 (default RefqMode::GFN2):
//        D4Dispersion d(atoms);
//        d.set_damping(gfn2_damping);
//        d.set_charges(mulliken_atomic);
//        e = d.energy();
//   2. DFT-D4 with a functional (RefqMode::DFT, EEQ atomic charges):
//        D4Dispersion d(atoms, RefqMode::DFT);
//        d.set_functional("pbe");
//        d.set_charges_eeq(net_charge);
//        e = d.energy();
class D4Dispersion {
public:
  explicit D4Dispersion(std::vector<core::Atom> atoms,
                      RefqMode mode = RefqMode::GFN2);

  void set_damping(const D4Damping &d) { m_damping = d; }
  void set_scaling(const D4Scaling &s) { m_scaling = s; }
  void set_refq_mode(RefqMode m) { m_refq_mode = m; }
  RefqMode refq_mode() const { return m_refq_mode; }

  // Set damping parameters from the bundled functional database.
  // Throws if the functional name is unknown. Common aliases (pbe, b3lyp,
  // wb97x, blyp, b97-3c, …) are supported.
  void set_functional(const std::string &functional);

  // Per-atom partial charges (positive = electron-deficient). Required for the
  // charge-aware reference projection. The supplied charges are treated as
  // independent of geometry — for DFT-D4 with EEQ charges that respond to
  // displacement, prefer set_charges_eeq() so the gradient picks up the
  // ∂q/∂R chain rule.
  void set_charges(const Vec &q_atomic) {
    m_q = q_atomic;
    m_dq_dR.clear(); // explicit charges are taken as fixed
  }

  // Compute and store EEQ partial charges + their position derivatives for
  // the current geometry. Use this for DFT-D4 where SCF Mulliken charges
  // aren't available; the analytical gradient will then include the EEQ
  // chain rule for full forces. `net_charge` is the system total charge.
  void set_charges_eeq(double net_charge = 0.0);

  // Cutoffs in Bohr for the 2-body, 3-body, and CN sums.
  void set_cutoffs(double disp2 = 60.0, double disp3 = 40.0, double cn = 30.0) {
    m_cutoff_disp2 = disp2;
    m_cutoff_disp3 = disp3;
    m_cutoff_cn = cn;
  }

  // Update positions while keeping element identities unchanged.
  void update_positions(const std::vector<core::Atom> &atoms);

  // Total dispersion energy (Hartree).
  double energy() const;

  // (energy, gradient) where gradient is 3 × N_atoms in Hartree/Bohr.
  std::pair<double, Mat3N> energy_and_gradient() const;

  // Periodic 2-body dispersion energy. Driven by the lattice (columns =
  // a, b, c in Bohr); two real-space sums are taken internally with cutoffs
  // `m_cutoff_disp2` (BJ damped 1/R^6+8) and `m_cutoff_cn` (CN counting
  // function), so the periodic-CN feeds the C6 weights correctly.
  //
  // Convention for the 2-body lattice sum:
  //   T=0 enumerates each pair once (i<j) — matches the molecular case.
  //   T!=0 enumerates all (i,j) including i==j with weight 1/2.
  // 3-body ATM is central-cell only (full ATM lattice sum is more involved;
  // central-cell ATM is small for molecular crystals).
  double energy_periodic(const occ::Mat3 &lattice_bohr) const;

  // The D4 covalent coordination numbers (erf-counted, EN-weighted).
  // Molecular sum.
  occ::Vec covalent_coordination_numbers() const;

  // Lattice-summed D4 covalent coordination numbers.
  occ::Vec covalent_coordination_numbers_periodic(
      const occ::Mat3 &lattice_bohr) const;

private:
  std::vector<core::Atom> m_atoms;
  Vec m_q;                       // atomic partial charges; defaults to zero
  // ∂q/∂R from EEQ — populated by set_charges_eeq(), empty otherwise.
  // dq_dR[i](α, j) = ∂q_i/∂R_j^α (per-Bohr). When empty the gradient treats
  // m_q as fixed (Hellmann-Feynman / SCC convention).
  std::vector<Mat3N> m_dq_dR;
  D4Damping m_damping{};
  D4Scaling m_scaling{};
  RefqMode m_refq_mode{RefqMode::GFN2};
  double m_cutoff_disp2{60.0};
  double m_cutoff_disp3{40.0};
  double m_cutoff_cn{30.0};
};

} // namespace occ::disp
