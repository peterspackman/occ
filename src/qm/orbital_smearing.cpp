#include <occ/core/log.h>
#include <occ/core/optimize.h>
#include <occ/qm/mo.h>
#include <occ/qm/orbital_smearing.h>
#include <unsupported/Eigen/SpecialFunctions>

/* compute inverse error functions with maximum error of 2.35793 ulp */
double erfinv(double a) {
  double t = fmaf(a, 0.0f - a, 1.0f);
  t = std::log(t);
  double p;
  if (std::abs(t) > 6.125f) {        // maximum ulp error = 2.35793
    p = 3.03697567e-10f;             //  0x1.4deb44p-32
    p = fmaf(p, t, 2.93243101e-8f);  //  0x1.f7c9aep-26
    p = fmaf(p, t, 1.22150334e-6f);  //  0x1.47e512p-20
    p = fmaf(p, t, 2.84108955e-5f);  //  0x1.dca7dep-16
    p = fmaf(p, t, 3.93552968e-4f);  //  0x1.9cab92p-12
    p = fmaf(p, t, 3.02698812e-3f);  //  0x1.8cc0dep-9
    p = fmaf(p, t, 4.83185798e-3f);  //  0x1.3ca920p-8
    p = fmaf(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = fmaf(p, t, 8.40016484e-1f);  //  0x1.ae16a4p-1
  } else {                           // maximum ulp error = 2.35002
    p = 5.43877832e-9f;              //  0x1.75c000p-28
    p = fmaf(p, t, 1.43285448e-7f);  //  0x1.33b402p-23
    p = fmaf(p, t, 1.22774793e-6f);  //  0x1.499232p-20
    p = fmaf(p, t, 1.12963626e-7f);  //  0x1.e52cd2p-24
    p = fmaf(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = fmaf(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = fmaf(p, t, 2.31468678e-3f);  //  0x1.2f6400p-9
    p = fmaf(p, t, 1.15392581e-2f);  //  0x1.7a1e50p-7
    p = fmaf(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = fmaf(p, t, 8.86226892e-1f);  //  0x1.c5bf88p-1
  }
  double r = a * p;
  return r;
}

namespace occ::qm {

Vec OrbitalSmearing::calculate_fermi_occupations(
    const MolecularOrbitals &mo) const {
  Vec result = Vec::Zero(mo.energies.rows());
  Vec de = (mo.energies.array() - mu).array() / sigma;
  for (int i = 0; i < result.rows(); i++) {
    if (de(i) < 40) {
      result(i) = 1.0 / (std::exp(de(i)) + 1.0);
    }
  }
  return result;
}

Vec OrbitalSmearing::calculate_gaussian_occupations(
    const MolecularOrbitals &mo) const {
  return 0.5 * ((mo.energies.array() - mu).array() / sigma).erfc();
}

Vec OrbitalSmearing::calculate_linear_occupations(
    const MolecularOrbitals &mo) const {
  const double sigma2 = sigma * sigma;
  Vec result = Vec::Zero(mo.energies.rows());
  Vec de = mo.energies.array() - mu;
  for (int i = 0; i < result.rows(); i++) {
    const double d = de(i);
    if (d <= -sigma) {
      result(i) = 1.0;
    } else if (d >= sigma) {
      result(i) = 0.0;
    } else if (d < 1e-10) {
      result(i) = 1.0 - (d + sigma) * (d + sigma) / 2 / sigma2;
    } else if (d > 0) {
      result(i) = (d - sigma) * (d - sigma) / 2 / sigma2;
    }
  }
  return result;
}

void OrbitalSmearing::smear_orbitals(MolecularOrbitals &mo) {
  if (kind == Kind::None)
    return;
  using occ::core::opt::LineSearch;

  size_t n_occ = mo.n_alpha;
  fermi_level = mo.energies(n_occ - 1);
  occ::log::debug("Fermi level: {}, occ_sum: {}", fermi_level,
                  mo.occupation.sum());

  auto cost_function = [&](double mu_value) {
    mu = mu_value;
    switch (kind) {
    case Kind::Fermi:
      mo.occupation = calculate_fermi_occupations(mo);
      break;
    case Kind::Gaussian:
      mo.occupation = calculate_gaussian_occupations(mo);
      break;
    case Kind::Linear:
      mo.occupation = calculate_linear_occupations(mo);
      break;
    default:
      break;
    }
    int num_electrons = mo.n_alpha + mo.n_beta;
    double sum_occ = (mo.occupation.array() * 2).sum();
    double diff = (num_electrons - sum_occ);
    return diff * diff;
  };

  LineSearch opt(cost_function);
  opt.set_left(fermi_level);
  opt.set_right(-fermi_level);
  opt.set_guess(fermi_level);
  double fmin = opt.f_xmin();

  // occ::log::info("Occ\n{}\n", mo.occupation);
  // occ::log::info("energies:\n{}\n", mo.energies);
  mo.update_occupied_orbitals_fractional();

  entropy = calculate_entropy(mo);
  occ::log::debug("HOMO: {}, LUMO: {}, fmin = {}", mo.energies(n_occ - 1),
                  mo.energies(n_occ), fmin);
  occ::log::debug("mu: {}, entropy: {}, ec_entropy: {}", mu, entropy,
                  -sigma * entropy);
}

double OrbitalSmearing::calculate_entropy(const MolecularOrbitals &mo) const {
  double result = 0.0;
  if (kind == Kind::Fermi) {
    for (int i = 0; i < mo.occupation.rows(); i++) {
      double occ = mo.occupation(i);
      if (occ >= 1.0 || occ <= 0.0)
        continue;
      result -= occ * std::log(occ) + (1 - occ) * std::log(1 - occ);
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
  return result * 2;
}

} // namespace occ::qm
