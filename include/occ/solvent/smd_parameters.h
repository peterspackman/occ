#pragma once

namespace occ::solvent {

struct SMDSolventParameters {
  SMDSolventParameters() {}
  SMDSolventParameters(double n, double n25, double a, double b, double g,
                       double d, double ar, double eh, bool w = false)
      : refractive_index_293K(n), refractive_index_298K(n25), acidity(a),
        basicity(b), gamma(g), dielectric(d), aromaticity(ar),
        electronegative_halogenicity(eh), is_water(w) {}

  double refractive_index_293K{
      0.0}; // refractive index optical frequencies @ 293K
  double refractive_index_298K{
      0.0};             // refractive index optical frequencies @ 298K
  double acidity{0.0};  // Abraham's hydrogen bond acidity
  double basicity{0.0}; // Abraham's hydrogen bond basicity
  double gamma{0.0};    // macroscopic surface tension parameter @298K
  double dielectric{
      0.0}; // dielectric constant @ 298K i.e. relative permittivity
  double aromaticity{
      0.0}; // fraction of non-H solvent atoms that are aromatic carbon
  double electronegative_halogenicity{0.0}; // fraction of non-H solvent atoms
                                            // that are F, Cl or Br
  bool is_water{false};                     // flag if this is water };
};

} // namespace occ::solvent
