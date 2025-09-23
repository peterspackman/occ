#include "descriptors_bindings.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/steinhardt.h>
#include <occ/descriptors/promolecule_shape.h>
#include <occ/crystal/crystal.h>

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

  return m;
}