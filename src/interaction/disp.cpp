#include <array>
#include <cmath>
#include <occ/interaction/disp.h>

namespace occ::disp {
using occ::core::Atom;

std::array<double, 110> Grimme06_a_6_disp_coeff{
    1.56,  1.18,  5.28,  5.28,  7.37,  5.51,  4.62,  3.48,  3.61,  3.31,  9.95,
    9.95,  13.68, 12.65, 11.66, 9.83,  9.38,  8.94,  13.69, 13.69, 13.69, 13.69,
    13.69, 13.69, 13.69, 13.69, 13.69, 13.69, 13.69, 13.69, 17.17, 17.22, 16.85,
    14.81, 14.71, 14.43, 20.69, 20.69, 20.69, 20.69, 20.69, 20.69, 20.69, 20.69,
    20.69, 20.69, 20.69, 20.69, 25.44, 25.91, 25.82, 23.46, 23.37, 22.81, 0.00,
    0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
    0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
    0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
    0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
    0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00};

/*
 placeholder value of 2.0 angstrom for those not int the Grimme06 paper
 Note that these are also in bohr/au  These are essentially vdw radii
*/

std::array<double, 110> Grimme06_r_6_disp_coeff{
    1.892, 1.912, 1.559, 2.661, 2.806, 2.744, 2.640, 2.536, 2.432, 2.349,
    2.162, 2.578, 3.097, 3.243, 3.222, 3.180, 3.097, 3.014, 2.806, 2.785,
    2.952, 2.952, 2.952, 2.952, 2.952, 2.952, 2.952, 2.952, 2.952, 2.952,
    3.118, 3.264, 3.326, 3.347, 3.305, 3.264, 3.076, 3.035, 3.097, 3.097,
    3.097, 3.097, 3.097, 3.097, 3.097, 3.097, 3.097, 3.097, 3.160, 3.409,
    3.555, 3.575, 3.575, 3.555, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779,
    3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779,
    3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779,
    3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779,
    3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779,
    3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779, 3.779};

double ce_model_dispersion_energy(std::vector<Atom> &atoms_a,
                                  std::vector<Atom> &atoms_b) {
  /*
  Return Grimmes's D2 dispersion energy between "self" and "atom"s.
  Based on C6 terms from Grimme (2006) J. Comp. Chem.  27(15) p. 1787
  E_disp = sum over atoms (C6 / r^6).
  */
  double result = 0.0;
  double d = 20.0;

  for (const auto &atom_i : atoms_a) {
    size_t ni = atom_i.atomic_number;
    double ai = Grimme06_a_6_disp_coeff[ni - 1];
    double ri = Grimme06_r_6_disp_coeff[ni - 1];

    for (const auto &atom_j : atoms_b) {
      size_t nj = atom_j.atomic_number;
      double aj = Grimme06_a_6_disp_coeff[nj - 1];
      double rj = Grimme06_r_6_disp_coeff[nj - 1];

      double rr = ri + rj;
      double dx = atom_i.x - atom_j.x;
      double dy = atom_i.y - atom_j.y;
      double dz = atom_i.z - atom_j.z;
      double rij = sqrt(dx * dx + dy * dy + dz * dz);
      double damping_factor = (1 / (1 + std::exp(-d * (rij / (rr)-1))));
      result += -(ai * aj / pow(rij, 6) * damping_factor);
    }
  }
  return result;
}

} // namespace occ::disp
