#include <algorithm>
#include <cmath>
#include <numbers>
#include <occ/solvent/sigma_kernel.h>
#include <stdexcept>

namespace occ::solvent::sigma {

std::string model_name(Model model) {
  switch (model) {
  case Model::CosmoSac2002:
    return "COSMO-SAC (2002)";
  case Model::CosmoSac2010:
    return "COSMO-SAC (2010)";
  }
  return "unknown";
}

double Parameters::c_es(double temperature) const {
  return a_es + b_es / (temperature * temperature);
}

Parameters Parameters::cosmo_sac_2002() {
  Parameters p;
  p.model = Model::CosmoSac2002;
  p.a_eff = 7.5;
  // Mullins averaging.
  p.r_av = 0.8176300195;
  p.f_decay = 1.0;
  // alpha' / 2, with no temperature dependence.
  p.a_es = 16466.72 / 2.0;
  p.b_es = 0.0;
  p.sigma_hb = 0.0084;
  p.c_hb = 85580.0;
  return p;
}

Parameters Parameters::cosmo_sac_2010() {
  Parameters p;
  p.model = Model::CosmoSac2010;
  p.a_eff = 7.25;
  // Hsieh averaging: r_av^2 = a_eff / pi.
  p.r_av = std::sqrt(7.25 / std::numbers::pi_v<double>);
  p.f_decay = 3.57;
  p.a_es = 6525.69;
  p.b_es = 1.4859e8;
  p.c_oh_oh = 4013.78;
  p.c_ot_ot = 932.31;
  p.c_oh_ot = 3016.43;
  p.sigma_0 = 0.007;
  return p;
}

Parameters Parameters::for_model(Model model) {
  switch (model) {
  case Model::CosmoSac2002:
    return cosmo_sac_2002();
  case Model::CosmoSac2010:
    return cosmo_sac_2010();
  }
  throw std::runtime_error("sigma::Parameters: unknown model");
}

namespace {

/// H-bond strength for a class pair; zero if either side cannot bond.
double hbond_strength(const Parameters &params, int class_a, int class_b) {
  const int oh = static_cast<int>(HBondClass::OH);
  const int ot = static_cast<int>(HBondClass::OT);
  if (class_a == oh && class_b == oh)
    return params.c_oh_oh;
  if (class_a == ot && class_b == ot)
    return params.c_ot_ot;
  if ((class_a == oh && class_b == ot) || (class_a == ot && class_b == oh))
    return params.c_oh_ot;
  return 0.0;
}

} // namespace

Kernel build_kernel(const Grid &grid, const Parameters &params,
                    double temperature) {
  Kernel kernel;
  kernel.grid = grid;
  kernel.num_classes = params.num_classes();

  const Eigen::Index dim = kernel.dim();
  kernel.misfit = Mat::Zero(dim, dim);
  kernel.hbond = Mat::Zero(dim, dim);

  const Vec centers = grid.centers();
  const double c_es = params.c_es(temperature);
  const bool resolved = params.resolves_hbond_classes();

  for (Eigen::Index a = 0; a < dim; a++) {
    const int class_a = static_cast<int>(a / grid.n);
    const double sigma_a = centers(a % grid.n);
    for (Eigen::Index b = 0; b < dim; b++) {
      const int class_b = static_cast<int>(b / grid.n);
      const double sigma_b = centers(b % grid.n);

      const double sum = sigma_a + sigma_b;
      kernel.misfit(a, b) = c_es * sum * sum;

      if (resolved) {
        if (sigma_a * sigma_b < 0.0) {
          const double c = hbond_strength(params, class_a, class_b);
          const double diff = sigma_a - sigma_b;
          kernel.hbond(a, b) = -c * diff * diff;
        }
      } else {
        const double acceptor = std::max(sigma_a, sigma_b);
        const double donor = std::min(sigma_a, sigma_b);
        kernel.hbond(a, b) = params.c_hb *
                             std::max(0.0, acceptor - params.sigma_hb) *
                             std::min(0.0, donor + params.sigma_hb);
      }
    }
  }
  return kernel;
}

} // namespace occ::solvent::sigma
