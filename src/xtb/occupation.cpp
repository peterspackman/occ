#include <cmath>
#include <occ/xtb/occupation.h>

namespace occ::xtb {

namespace {

// Aufbau filling: fully occupy the lowest ⌊n⌋ orbitals, put the remainder in
// the next one. Returns the number of (partially) occupied orbitals.
int aufbau_filling(double n_electrons, Vec &occ) {
  occ.setZero();
  const int n_orb = static_cast<int>(occ.size());
  int filled = static_cast<int>(std::floor(n_electrons + 1e-10));
  filled = std::min(filled, n_orb);
  occ.head(filled).setConstant(1.0);
  const double remainder = n_electrons - filled;
  if (filled < n_orb && remainder > 1e-10) {
    occ(filled) = remainder;
    return filled + 1;
  }
  return filled;
}

} // namespace

AlphaBetaOccupation alpha_beta_occupation(double n_elec, double n_unpaired) {
  // Clamp so we can never produce a negative β occupation.
  const double diff = std::min(std::abs(n_unpaired), n_elec);
  const double paired = n_elec - diff;
  return {0.5 * paired + diff, 0.5 * paired};
}

OrbitalFilling fermi_filling(double n_electrons, double kt,
                             const Vec &orbital_energies) {
  OrbitalFilling result;
  const int n_orb = static_cast<int>(orbital_energies.size());
  result.occupations = Vec::Zero(n_orb);
  if (n_orb == 0)
    return result;

  const int homo = aufbau_filling(n_electrons, result.occupations);
  // Nothing to smear: no electrons, or every orbital is already full.
  if (kt <= 0.0 || homo <= 0 || homo >= n_orb)
    return result;

  // Newton iteration on the Fermi level so that Σ_i f_i(μ) = n_electrons.
  double mu = 0.5 * (orbital_energies(homo - 1) + orbital_energies(homo));
  constexpr int max_cycles = 200;
  const double threshold = std::sqrt(std::numeric_limits<double>::epsilon());
  Vec occ = result.occupations;
  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    double total = 0.0, dtotal = 0.0;
    for (int i = 0; i < n_orb; ++i) {
      const double x = (orbital_energies(i) - mu) / kt;
      // exp overflows well before the occupation stops being 0/1 to double
      // precision; short-circuit rather than relying on inf arithmetic.
      if (x > 50.0) {
        occ(i) = 0.0;
        continue;
      }
      if (x < -50.0) {
        occ(i) = 1.0;
        total += 1.0;
        continue;
      }
      const double ex = std::exp(x);
      const double f = 1.0 / (ex + 1.0);
      occ(i) = f;
      total += f;
      dtotal += ex / (kt * (ex + 1.0) * (ex + 1.0));
    }
    const double residual = n_electrons - total;
    if (std::abs(residual) <= threshold)
      break;
    if (dtotal <= std::sqrt(std::numeric_limits<double>::min()))
      break; // flat — no level near μ to move charge into
    mu += residual / dtotal;
  }

  result.occupations = occ;
  result.fermi_level = mu;
  // −T·S = kT Σ [f ln f + (1−f) ln(1−f)]; the x ln x limits are 0 at f = 0, 1.
  double entropy_energy = 0.0;
  for (int i = 0; i < n_orb; ++i) {
    const double f = occ(i);
    if (f > 1e-14)
      entropy_energy += f * std::log(f);
    if (f < 1.0 - 1e-14)
      entropy_energy += (1.0 - f) * std::log(1.0 - f);
  }
  result.entropy_energy = kt * entropy_energy;
  return result;
}

} // namespace occ::xtb
