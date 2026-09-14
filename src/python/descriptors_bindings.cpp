#include "descriptors_bindings.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <occ/crystal/crystal.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/promolecule_shape.h>
#include <occ/descriptors/rinse.h>
#include <occ/descriptors/steinhardt.h>

using namespace occ::descriptors;
using namespace occ::crystal;
using namespace occ;

nb::module_ register_descriptors_bindings(nb::module_ &m) {
  using namespace nb::literals;

  // PointwiseDistanceDistributionConfig
  nb::class_<PointwiseDistanceDistributionConfig>(m, "PDDConfig")
      .def(nb::init<>())
      .def_rw("lexsort", &PointwiseDistanceDistributionConfig::lexsort,
              "Lexicographically sort rows")
      .def_rw("collapse", &PointwiseDistanceDistributionConfig::collapse,
              "Merge similar rows within tolerance")
      .def_rw("collapse_tol", &PointwiseDistanceDistributionConfig::collapse_tol,
              "Tolerance for merging rows (Chebyshev distance)")
      .def_rw("return_groups", &PointwiseDistanceDistributionConfig::return_groups,
              "Return grouping information");

  // PointwiseDistanceDistribution (PDD)
  nb::class_<PointwiseDistanceDistribution>(m, "PDD")
      .def(nb::init<const Crystal &, int>(),
           "crystal"_a, "k"_a,
           "Construct PDD from crystal structure with k nearest neighbors")
      .def(nb::init<const Crystal &, int, const PointwiseDistanceDistributionConfig &>(),
           "crystal"_a, "k"_a, "config"_a,
           "Construct PDD from crystal structure with k nearest neighbors and configuration")
      .def_prop_ro("weights", &PointwiseDistanceDistribution::weights,
                   "Get the weights for each environment")
      .def_prop_ro("distances", &PointwiseDistanceDistribution::distances,
                   "Get the distance matrix (environments as columns)")
      .def("average_minimum_distance", &PointwiseDistanceDistribution::average_minimum_distance,
           "Calculate Average Minimum Distance from this PDD")
      .def("matrix", &PointwiseDistanceDistribution::matrix,
           "Get the full PDD matrix (weights + distances)")
      .def("size", &PointwiseDistanceDistribution::size,
           "Number of unique chemical environments")
      .def("k", &PointwiseDistanceDistribution::k,
           "Number of neighbors considered")
      .def_prop_ro("groups", &PointwiseDistanceDistribution::groups,
                   "Get grouping information if available");

  // Steinhardt descriptors
  nb::class_<Steinhardt>(m, "Steinhardt")
      .def(nb::init<size_t>(), "lmax"_a,
           "Initialize Steinhardt descriptor with maximum l value")
      .def("compute_q", &Steinhardt::compute_q, "positions"_a,
           "Compute Steinhardt Q parameters for given positions")
      .def("compute_w", &Steinhardt::compute_w, "positions"_a,
           "Compute Steinhardt W parameters for given positions")
      .def("compute_qlm", &Steinhardt::compute_qlm, "positions"_a,
           "Compute complex Steinhardt Q_lm parameters for given positions")
      .def("compute_averaged_q", &Steinhardt::compute_averaged_q,
           "positions"_a, "radius"_a = 6.0,
           "Compute locally averaged Steinhardt Q parameters")
      .def("compute_averaged_w", &Steinhardt::compute_averaged_w,
           "positions"_a, "radius"_a = 6.0,
           "Compute locally averaged Steinhardt W parameters")
      .def("precompute_wigner3j_coefficients", &Steinhardt::precompute_wigner3j_coefficients,
           "Precompute Wigner 3j coefficients for better performance")
      .def("size", &Steinhardt::size,
           "Number of l values (lmax + 1)")
      .def("nlm", &Steinhardt::nlm,
           "Total number of (l,m) combinations");

  // PromoleculeDensityShape types
  nb::class_<PromoleculeDensityShape::InterpolatorParameters>(m, "PromoleculeInterpolatorParameters")
      .def(nb::init<>())
      .def_rw("num_points", &PromoleculeDensityShape::InterpolatorParameters::num_points,
              "Number of interpolation points")
      .def_rw("domain_lower", &PromoleculeDensityShape::InterpolatorParameters::domain_lower,
              "Lower bound of interpolation domain")
      .def_rw("domain_upper", &PromoleculeDensityShape::InterpolatorParameters::domain_upper,
              "Upper bound of interpolation domain");

  nb::class_<PromoleculeDensityShape::AtomInterpolator>(m, "PromoleculeAtomInterpolator")
      .def(nb::init<>())
      .def_rw("positions", &PromoleculeDensityShape::AtomInterpolator::positions,
              "Atomic positions")
      .def_rw("threshold", &PromoleculeDensityShape::AtomInterpolator::threshold,
              "Distance threshold for interpolation");

  // RINSE
  nb::enum_<RinseRadialBasis>(m, "RinseRadialBasis")
      .value("SmoothShellsNL", RinseRadialBasis::SmoothShellsNL,
             "Volume-uniform shells, edges at scale * n^(1/3)")
      .value("SmoothShellsCW", RinseRadialBasis::SmoothShellsCW,
             "Linearly spaced shells normalised to a partition of unity");

  nb::class_<RinseParams>(m, "RinseParams")
      .def(nb::init<>())
      .def_rw("n_max", &RinseParams::n_max,
              "Number of radial shells; orders are n = 0 .. n_max-1")
      .def_rw("l_min", &RinseParams::l_min,
              "First angular level; 4 drops the monopole and quadrupole terms")
      .def_rw("l_max", &RinseParams::l_max,
              "One past the last angular level, which steps by two")
      .def_rw("radial_scale", &RinseParams::radial_scale,
              "Per-shell scale in inverse Angstroms")
      .def_rw("radial_basis", &RinseParams::radial_basis, "Radial basis family")
      .def_rw("fixed_uiso", &RinseParams::fixed_uiso,
              nb::for_setter(nb::arg("value").none()),
              "Replace the structure's ADPs with this isotropic U, in A^2, or "
              "None to use the ADPs as read")
      .def_rw("monopole_normalisation", &RinseParams::monopole_normalisation,
              "Divide each shell's power by its monopole, removing the "
              "resolution-dependent intensity envelope")
      .def_rw("log1p", &RinseParams::log1p,
              "Compress the power spectrum with log1p")
      .def_rw("l2", &RinseParams::l2, "Scale the descriptor to unit L2 norm")
      .def_prop_ro("q_max", &RinseParams::q_max,
                   "|G| cutoff in inverse Angstroms")
      .def_prop_ro("sin_theta_over_lambda_max",
                   &RinseParams::sin_theta_over_lambda_max,
                   "Resolution cutoff in inverse Angstroms")
      .def_prop_ro("l_values", &RinseParams::l_values, "The angular levels")
      .def_prop_ro("num_l_levels", &RinseParams::num_l_levels,
                   "Number of angular levels")
      .def_prop_ro("size", &RinseParams::size,
                   "Length of the flattened descriptor");

  nb::class_<ReflectionList>(m, "ReflectionList")
      .def_ro("hkl", &ReflectionList::hkl, "Miller indices, (3, M)")
      .def_ro("q_vectors", &ReflectionList::q_vectors,
              "Cartesian reciprocal-space vectors in inverse Angstroms, (3, M)")
      .def_ro("q_magnitudes", &ReflectionList::q_magnitudes,
              "|G| in inverse Angstroms")
      .def_ro("intensities", &ReflectionList::intensities,
              "2 |F(hkl)|^2 -- only one of each Friedel pair is stored, so the "
              "mate's contribution is already counted")
      .def("__len__", &ReflectionList::size);

  nb::class_<Rinse>(m, "Rinse")
      .def(nb::init<const RinseParams &>(), "params"_a = RinseParams{},
           "Reciprocal-space INvariant Spectral Embedding. Reuse one instance "
           "across a batch: constructing it precomputes the shell layout and "
           "the harmonic recurrences.")
      .def_prop_ro("parameters", &Rinse::parameters)
      .def("power_spectrum",
           nb::overload_cast<const ReflectionList &>(&Rinse::power_spectrum,
                                                     nb::const_),
           "reflections"_a,
           "Power spectrum for a reflection list, (n_max, num_l_levels)")
      .def("power_spectrum",
           nb::overload_cast<const Crystal &>(&Rinse::operator(), nb::const_),
           "crystal"_a, "Power spectrum for a crystal, (n_max, num_l_levels)")
      .def("compute", &Rinse::compute, "crystal"_a,
           "The descriptor as a flat vector, row-major in (n, l)")
      .def_static("flatten", &Rinse::flatten, "power_spectrum"_a,
                  "Flatten a power spectrum matrix into a descriptor vector");

  m.def("rinse_reflections", &rinse_reflections, "crystal"_a,
        "params"_a = RinseParams{},
        "Enumerate the resolution sphere and compute reflection intensities");
  m.def("rinse_radial_basis", &rinse_radial_basis, "q"_a, "params"_a,
        "Evaluate the radial basis at reciprocal-space magnitudes, (M, n_max)");
  m.def("rinse_hash", &rinse_hash, "descriptor"_a,
        "num_words"_a = occ::descriptors::default_hash_words,
        "A locality-sensitive hash of a descriptor, as proquint words");
  m.def("rinse_hash_to_bits", &rinse_hash_to_bits, "hash"_a,
        "Decode a proquint hash back to its bits, most significant first");

  return m;
}