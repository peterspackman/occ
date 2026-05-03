#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::disp {

// Damping parameters for the rational Becke-Johnson form of DFT-D3.
//   E2 = -Σ_{A<B} [ s6·C6/(r^6 + r0^6) + s8·C8/(r^8 + r0^8) ]
//   r0 = a1·√(3·sqrt_zr4r2_A·sqrt_zr4r2_B) + a2
// `s9` scales the Axilrod-Teller-Muto 3-body term (set to 0 to disable).
struct D3Damping {
  double s6{1.0};
  double s8{0.0};
  double s9{1.0};
  double a1{0.0};
  double a2{0.0};
  int alp{14}; // ATM exponent (D3 default differs from D4's 16)
};

// Native DFT-D3 (Becke-Johnson damping) evaluator. Construct with a list of
// atoms, set damping (or load via `set_functional`), compute energy or
// energy + gradient. Reference data is loaded from
// share/dftd3/refdata.json on first use; per-functional damping presets come
// from share/dftd3/functionals.json.
//
// Supports atomic numbers 1..94. Heavier elements throw at construction.
//
//   Dispersion d(atoms);
//   d.set_functional("pbe");
//   double e = d.energy();
class DispersionD3 {
public:
  explicit DispersionD3(std::vector<core::Atom> atoms);

  void set_damping(const D3Damping &d) { m_damping = d; }
  // Set damping parameters from the bundled functional database (BJ variant).
  // Throws if the functional name is unknown.
  void set_functional(const std::string &functional);

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
  // Currently uses central differences; analytical gradient is a follow-up.
  std::pair<double, Mat3N> energy_and_gradient() const;

  // The D3 covalent coordination numbers (erf-counted, no EN factor).
  Vec coordination_numbers() const;

private:
  std::vector<core::Atom> m_atoms;
  D3Damping m_damping{};
  double m_cutoff_disp2{60.0};
  double m_cutoff_disp3{40.0};
  double m_cutoff_cn{30.0};
};

} // namespace occ::disp
