#pragma once
#include <occ/solvent/sigma_profile.h>
#include <string>

namespace occ::solvent::sigma {

/// Which published parameterisation to use. All share the same fixed-point
/// solver; they differ in the exchange-energy kernel, the segment averaging,
/// and whether the profile is H-bond resolved.
enum class Model {
  CosmoSac2002, ///< Lin & Sandler, Ind. Eng. Chem. Res. 41 (2002) 899
  CosmoSac2010, ///< Hsieh, Sandler & Lin, Fluid Phase Equilib. 297 (2010) 90
};

std::string model_name(Model model);

/// Parameters for one model. Energies are kcal/mol per contact, so the
/// interaction coefficients are quoted in kcal Å⁴ mol⁻¹ e⁻².
struct Parameters {
  Model model{Model::CosmoSac2010};

  double a_eff{7.25};   ///< effective contact area, Å²
  double r_av{1.51929}; ///< segment averaging radius, Å
  double f_decay{3.57}; ///< decay factor in the averaging exponent

  /// Electrostatic misfit coefficient `c_ES(T) = a_es + b_es/T²`.
  double a_es{6525.69};
  double b_es{1.4859e8};

  /// 2002 H-bond term: threshold `sigma_hb` and strength `c_hb`.
  double sigma_hb{0.0084};
  double c_hb{85580.0};

  /// 2010 H-bond term: per-class-pair strengths, and the width `sigma_0` of
  /// the profile split.
  double c_oh_oh{4013.78};
  double c_ot_ot{932.31};
  double c_oh_ot{3016.43};
  double sigma_0{0.007};

  /// Gas constant used in the Boltzmann weighting, kcal mol⁻¹ K⁻¹.
  ///
  /// The published COSMO-SAC parameters were fitted with the truncated value
  /// 0.001987 rather than the CODATA one (`sigma::gas_constant_kcal`, which
  /// is larger by 1.1e-4 relative). Reproducing published numbers means
  /// using the value the parameters were fitted with, so that is the default.
  double gas_constant{0.001987};

  /// Staverman–Guggenheim combinatorial term.
  double q0{79.53};  ///< standard area, Å²
  double r0{66.69};  ///< standard volume, Å³
  double z_coordination{10.0};

  bool resolves_hbond_classes() const { return model == Model::CosmoSac2010; }
  int num_classes() const {
    return resolves_hbond_classes() ? num_hbond_classes : 1;
  }
  HBondSplit hbond_split() const {
    return HBondSplit{resolves_hbond_classes(), sigma_0};
  }
  double c_es(double temperature) const;

  static Parameters cosmo_sac_2002();
  static Parameters cosmo_sac_2010();
  static Parameters for_model(Model model);
};

/// Pairwise segment exchange energies on the grid, kcal/mol per contact.
///
/// The misfit and H-bond terms are held separately rather than summed: the
/// moment contractions in `sigma_potential.h` report them individually,
/// which is only possible if they are never merged here.
///
/// Rows and columns run over the composite index `class * grid.n + bin`, so
/// `dim() == grid.n * num_classes`.
struct Kernel {
  Grid grid;
  int num_classes{1};
  Mat misfit;
  Mat hbond;
  /// Carried from the parameters so the solver weights with the same R the
  /// kernel coefficients were fitted against.
  double gas_constant{0.001987};

  Eigen::Index dim() const {
    return static_cast<Eigen::Index>(grid.n) * num_classes;
  }
  Mat total() const { return misfit + hbond; }
};

/// Build the exchange-energy matrices.
///
/// COSMO-SAC 2002 uses a threshold H-bond term,
/// `c_hb · max(0, σ_acc − σ_hb) · min(0, σ_don + σ_hb)`, which is zero
/// unless the pair straddles ±σ_hb.
///
/// COSMO-SAC 2010 instead uses `−c_hb^{ts} (σ_m − σ_n)²`, gated on the two
/// densities having opposite signs and on both classes being H-bonding. That
/// gate is a step, so this kernel is discontinuous across σ = 0.
Kernel build_kernel(const Grid &grid, const Parameters &params,
                    double temperature);

} // namespace occ::solvent::sigma
