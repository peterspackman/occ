#include "descriptors_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/steinhardt.h>
#include <occ/descriptors/promolecule_shape.h>
#include <occ/crystal/crystal.h>

using namespace emscripten;
using namespace occ::descriptors;
using namespace occ::crystal;
using namespace occ;

void register_descriptors_bindings() {
  // PointwiseDistanceDistributionConfig
  class_<PointwiseDistanceDistributionConfig>("PDDConfig")
      .constructor<>()
      .property("lexsort", &PointwiseDistanceDistributionConfig::lexsort)
      .property("collapse", &PointwiseDistanceDistributionConfig::collapse)
      .property("collapseTol", &PointwiseDistanceDistributionConfig::collapse_tol)
      .property("returnGroups", &PointwiseDistanceDistributionConfig::return_groups);

  // PointwiseDistanceDistribution (PDD)
  class_<PointwiseDistanceDistribution>("PDD")
      .constructor<const Crystal &, int>()
      .constructor<const Crystal &, int, const PointwiseDistanceDistributionConfig &>()
      .function("weights", optional_override([](const PointwiseDistanceDistribution &pdd) {
                  const Vec &weights = pdd.weights();
                  val result = val::global("Float64Array").new_(weights.size());
                  for (int i = 0; i < weights.size(); ++i) {
                    result.set(i, weights(i));
                  }
                  return result;
                }))
      .function("distances", optional_override([](const PointwiseDistanceDistribution &pdd) {
                  const Mat &distances = pdd.distances();
                  val result = val::global("Float64Array").new_(distances.size());
                  for (int i = 0; i < distances.rows(); ++i) {
                    for (int j = 0; j < distances.cols(); ++j) {
                      result.set(i * distances.cols() + j, distances(i, j));
                    }
                  }
                  return result;
                }))
      .function("averageMinimumDistance", optional_override([](const PointwiseDistanceDistribution &pdd) {
                  const Vec amd = pdd.average_minimum_distance();
                  val result = val::global("Float64Array").new_(amd.size());
                  for (int i = 0; i < amd.size(); ++i) {
                    result.set(i, amd(i));
                  }
                  return result;
                }))
      .function("matrix", optional_override([](const PointwiseDistanceDistribution &pdd) {
                  const Mat matrix = pdd.matrix();
                  val result = val::global("Float64Array").new_(matrix.size());
                  for (int i = 0; i < matrix.rows(); ++i) {
                    for (int j = 0; j < matrix.cols(); ++j) {
                      result.set(i * matrix.cols() + j, matrix(i, j));
                    }
                  }
                  return result;
                }))
      .function("size", &PointwiseDistanceDistribution::size)
      .function("k", &PointwiseDistanceDistribution::k)
      .function("groups", optional_override([](const PointwiseDistanceDistribution &pdd) {
                  const Eigen::MatrixXi &groups = pdd.groups();
                  val result = val::global("Int32Array").new_(groups.size());
                  for (int i = 0; i < groups.rows(); ++i) {
                    for (int j = 0; j < groups.cols(); ++j) {
                      result.set(i * groups.cols() + j, groups(i, j));
                    }
                  }
                  return result;
                }));

  // Steinhardt descriptors
  class_<Steinhardt>("Steinhardt")
      .constructor<size_t>()
      .function("computeQ", optional_override([](Steinhardt &steinhardt, const val &positionsArray) {
                  // Convert JavaScript array to Eigen matrix
                  int numAtoms = positionsArray["length"].as<int>() / 3;
                  Mat3N positions(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    positions(0, i) = positionsArray[i * 3 + 0].as<double>();
                    positions(1, i) = positionsArray[i * 3 + 1].as<double>();
                    positions(2, i) = positionsArray[i * 3 + 2].as<double>();
                  }

                  Vec q = steinhardt.compute_q(positions);
                  val result = val::global("Float64Array").new_(q.size());
                  for (int i = 0; i < q.size(); ++i) {
                    result.set(i, q(i));
                  }
                  return result;
                }))
      .function("computeW", optional_override([](Steinhardt &steinhardt, const val &positionsArray) {
                  // Convert JavaScript array to Eigen matrix
                  int numAtoms = positionsArray["length"].as<int>() / 3;
                  Mat3N positions(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    positions(0, i) = positionsArray[i * 3 + 0].as<double>();
                    positions(1, i) = positionsArray[i * 3 + 1].as<double>();
                    positions(2, i) = positionsArray[i * 3 + 2].as<double>();
                  }

                  Vec w = steinhardt.compute_w(positions);
                  val result = val::global("Float64Array").new_(w.size());
                  for (int i = 0; i < w.size(); ++i) {
                    result.set(i, w(i));
                  }
                  return result;
                }))
      .function("computeAveragedQ", optional_override([](Steinhardt &steinhardt, const val &positionsArray, double radius) {
                  // Convert JavaScript array to Eigen matrix
                  int numAtoms = positionsArray["length"].as<int>() / 3;
                  Mat3N positions(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    positions(0, i) = positionsArray[i * 3 + 0].as<double>();
                    positions(1, i) = positionsArray[i * 3 + 1].as<double>();
                    positions(2, i) = positionsArray[i * 3 + 2].as<double>();
                  }

                  Vec q = steinhardt.compute_averaged_q(positions, radius);
                  val result = val::global("Float64Array").new_(q.size());
                  for (int i = 0; i < q.size(); ++i) {
                    result.set(i, q(i));
                  }
                  return result;
                }))
      .function("computeAveragedW", optional_override([](Steinhardt &steinhardt, const val &positionsArray, double radius) {
                  // Convert JavaScript array to Eigen matrix
                  int numAtoms = positionsArray["length"].as<int>() / 3;
                  Mat3N positions(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    positions(0, i) = positionsArray[i * 3 + 0].as<double>();
                    positions(1, i) = positionsArray[i * 3 + 1].as<double>();
                    positions(2, i) = positionsArray[i * 3 + 2].as<double>();
                  }

                  Vec w = steinhardt.compute_averaged_w(positions, radius);
                  val result = val::global("Float64Array").new_(w.size());
                  for (int i = 0; i < w.size(); ++i) {
                    result.set(i, w(i));
                  }
                  return result;
                }))
      .function("precomputeWigner3jCoefficients", &Steinhardt::precompute_wigner3j_coefficients)
      .function("size", &Steinhardt::size)
      .function("nlm", &Steinhardt::nlm);

  // PromoleculeDensityShape types
  class_<PromoleculeDensityShape::InterpolatorParameters>("PromoleculeInterpolatorParameters")
      .constructor<>()
      .property("numPoints", &PromoleculeDensityShape::InterpolatorParameters::num_points)
      .property("domainLower", &PromoleculeDensityShape::InterpolatorParameters::domain_lower)
      .property("domainUpper", &PromoleculeDensityShape::InterpolatorParameters::domain_upper);

  class_<PromoleculeDensityShape::AtomInterpolator>("PromoleculeAtomInterpolator")
      .constructor<>()
      .function("getPositions", optional_override([](const PromoleculeDensityShape::AtomInterpolator &interp) {
                  const FMat3N &positions = interp.positions;
                  val result = val::global("Float32Array").new_(positions.size());
                  for (int i = 0; i < positions.rows(); ++i) {
                    for (int j = 0; j < positions.cols(); ++j) {
                      result.set(i * positions.cols() + j, positions(i, j));
                    }
                  }
                  return result;
                }))
      .function("setPositions", optional_override([](PromoleculeDensityShape::AtomInterpolator &interp, const val &positionsArray) {
                  int numAtoms = positionsArray["length"].as<int>() / 3;
                  interp.positions = FMat3N(3, numAtoms);
                  for (int i = 0; i < numAtoms; ++i) {
                    interp.positions(0, i) = positionsArray[i * 3 + 0].as<float>();
                    interp.positions(1, i) = positionsArray[i * 3 + 1].as<float>();
                    interp.positions(2, i) = positionsArray[i * 3 + 2].as<float>();
                  }
                }))
      .property("threshold", &PromoleculeDensityShape::AtomInterpolator::threshold);
}