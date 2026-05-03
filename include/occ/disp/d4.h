#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
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

// Native DFT-D4 evaluator. Construct with a list of atoms (positions in Bohr,
// elements via core::Atom::atomic_number), set damping/charges, compute energy
// or energy + gradient. Reference data is loaded once from share/dftd4/refdata.json
// and cached.
class Dispersion {
public:
  explicit Dispersion(std::vector<core::Atom> atoms);

  void set_damping(const D4Damping &d) { m_damping = d; }
  void set_scaling(const D4Scaling &s) { m_scaling = s; }
  // Per-atom partial charges (positive = electron-deficient). Required for the
  // charge-aware reference projection.
  void set_charges(const Vec &q_atomic) { m_q = q_atomic; }
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

  // The D4 covalent coordination numbers (erf-counted, EN-weighted).
  Vec covalent_coordination_numbers() const;

private:
  std::vector<core::Atom> m_atoms;
  Vec m_q;                  // atomic partial charges; defaults to zero
  D4Damping m_damping{};
  D4Scaling m_scaling{};
  double m_cutoff_disp2{60.0};
  double m_cutoff_disp3{40.0};
  double m_cutoff_cn{30.0};
};

} // namespace occ::disp
