#include "rinse_reference_data.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <fmt/core.h>
#include <map>
#include <numbers>
#include <occ/core/element.h>
#include <occ/core/parallel.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/xray_form_factors.h>
#include <occ/descriptors/rinse.h>
#include <occ/io/cifparser.h>
#include <occ/io/load_geometry.h>
#include <string>
#include <tuple>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using occ::crystal::Crystal;
using occ::crystal::SymmetryOperation;
using occ::descriptors::Rinse;
using occ::descriptors::rinse_hash;
using occ::descriptors::rinse_hash_to_bits;
using occ::descriptors::rinse_radial_basis;
using occ::descriptors::rinse_reflections;
using occ::descriptors::RinseParams;
using occ::descriptors::RinseRadialBasis;

namespace {

Crystal load(const std::string &name) {
  return occ::io::load_crystal(
      fmt::format("{}/rinse/{}", OCC_TEST_DATA_DIR, name));
}

/// The largest difference tolerated against the reference. What is left after
/// matching the algorithm exactly is summation order, and cctbx summing over
/// the whole sphere where occ sums over half of it and doubles.
constexpr double reference_tolerance = 1e-13;

} // namespace

TEST_CASE("RINSE descriptor matches the reference implementation", "[rinse]") {
  const Rinse rinse{};
  for (const auto &expected : rinse_reference::structures) {
    const occ::Vec descriptor = rinse.compute(load(expected.name));
    REQUIRE(descriptor.size() == 128);

    double worst = 0.0;
    int worst_index = 0;
    for (int i = 0; i < 128; i++) {
      const double difference =
          std::abs(descriptor(i) - expected.descriptor[i]);
      if (difference > worst) {
        worst = difference;
        worst_index = i;
      }
    }
    INFO(fmt::format("{}: element {} is {:.17g}, reference {:.17g}",
                     expected.name, worst_index, descriptor(worst_index),
                     expected.descriptor[worst_index]));
    CHECK(worst < reference_tolerance);
  }
}

TEST_CASE("RINSE reflection intensities match the reference", "[rinse]") {
  const RinseParams params{};
  for (const auto &expected : rinse_reference::structures) {
    const auto reflections = rinse_reflections(load(expected.name), params);
    INFO(expected.name);

    // The reference works with the whole sphere; occ keeps one of each Friedel
    // pair. It also keeps systematically absent reflections, which cctbx drops,
    // so the two lists are not the same length -- but an absent reflection
    // contributes nothing to the sum.
    CHECK(2 * reflections.size() >= expected.num_reflections);
    CHECK_THAT(reflections.intensities.sum(),
               WithinRel(expected.intensity_sum, 1e-12));

    std::map<std::tuple<int, int, int>, double> intensity;
    for (Eigen::Index i = 0; i < reflections.size(); i++)
      intensity.emplace(std::tuple{reflections.hkl(0, i), reflections.hkl(1, i),
                                   reflections.hkl(2, i)},
                        0.5 * reflections.intensities(i));

    for (const auto &sample : expected.sample_reflections) {
      auto found = intensity.find(std::tuple{sample.h, sample.k, sample.l});
      if (found == intensity.end()) // the Friedel mate is the one we kept
        found = intensity.find(std::tuple{-sample.h, -sample.k, -sample.l});
      INFO(fmt::format("({} {} {})", sample.h, sample.k, sample.l));
      REQUIRE(found != intensity.end());
      CHECK_THAT(found->second, WithinRel(sample.intensity, 1e-12) ||
                                    WithinAbs(sample.intensity, 1e-9));
    }
  }
}

TEST_CASE("RINSE hashes match the reference", "[rinse]") {
  const Rinse rinse{};
  for (const auto &expected : rinse_reference::structures) {
    const occ::Vec descriptor = rinse.compute(load(expected.name));
    INFO(expected.name);
    CHECK(rinse_hash(descriptor) == std::string(expected.hash_one_word));
    CHECK(rinse_hash(descriptor, 5) == std::string(expected.hash_five_words));
  }
}

TEST_CASE("RINSE hash round-trips through its bits", "[rinse]") {
  const Rinse rinse{};
  const occ::Vec descriptor = rinse.compute(load("ylid.cif"));
  const std::string hash = rinse_hash(descriptor, 3);
  const auto bits = rinse_hash_to_bits(hash);
  REQUIRE(bits.size() == 48);

  // The bits are the signs of the PCA projection, most significant first.
  CHECK(rinse_hash_to_bits(rinse_hash(descriptor, 1)).size() == 16);
  CHECK(std::equal(bits.begin(), bits.begin() + 16,
                   rinse_hash_to_bits(rinse_hash(descriptor, 1)).begin()));
}

TEST_CASE("RINSE results do not depend on the thread count", "[rinse]") {
  const Rinse rinse{};
  const Crystal crystal = load("QEHWEG01_P21.cif");

  const int threads_before = occ::parallel::get_num_threads();
  occ::parallel::set_num_threads(1);
  const occ::Vec serial = rinse.compute(crystal);
  occ::parallel::set_num_threads(8);
  const occ::Vec threaded = rinse.compute(crystal);
  occ::parallel::set_num_threads(threads_before);

  // Not "close to": the reduction is split by reflection count, never by how
  // many threads happen to be available, so this is bit for bit.
  CHECK(serial == threaded);
}

TEST_CASE("RINSE does not depend on the choice of unit cell basis", "[rinse]") {
  // TETRAZ01 described again on the C-centred cell a' = a + b, b' = -a + b:
  // twice the volume, and operations gemmi has no table entry for, so the CIF
  // reader has to take them as given. The reciprocal lattice and its
  // intensities are the same physical objects either way.
  const Crystal original = load("TETRAZ01.res");
  occ::Mat3 basis;
  basis << 1, -1, 0, 1, 1, 0, 0, 0, 1;
  const occ::Mat3 inverse = basis.inverse();

  const occ::Mat3 direct = original.unit_cell().direct() * basis;
  const occ::Vec3 lengths = direct.colwise().norm();
  const auto angle = [&](int i, int j) {
    return std::acos(direct.col(i).dot(direct.col(j)) /
                     (lengths(i) * lengths(j))) *
           180.0 / std::numbers::pi;
  };

  std::string cif = fmt::format(
      "data_transformed\n_cell_length_a {:.17g}\n_cell_length_b {:.17g}\n"
      "_cell_length_c {:.17g}\n_cell_angle_alpha {:.17g}\n"
      "_cell_angle_beta {:.17g}\n_cell_angle_gamma {:.17g}\n"
      "_symmetry_space_group_name_H-M '{}'\nloop_\n"
      "_symmetry_equiv_pos_as_xyz\n",
      lengths(0), lengths(1), lengths(2), angle(1, 2), angle(0, 2), angle(0, 1),
      original.space_group().symbol());
  const std::vector<occ::Vec3> centring{occ::Vec3(0.0, 0.0, 0.0),
                                        occ::Vec3(0.5, 0.5, 0.0)};
  for (const auto &op : original.space_group().symmetry_operations()) {
    occ::Mat4 seitz = occ::Mat4::Identity();
    seitz.topLeftCorner(3, 3) =
        (inverse * op.rotation() * basis).array().round().matrix();
    seitz.topRightCorner(3, 1) = inverse * op.translation();
    for (const occ::Vec3 &translation : centring)
      cif +=
          SymmetryOperation(seitz).translated(translation, true).to_string() +
          "\n";
  }

  cif += "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
         "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
         "_atom_site_occupancy\n";
  const auto &asym = original.asymmetric_unit();
  for (Eigen::Index i = 0; i < asym.positions.cols(); i++) {
    const occ::Vec3 x = inverse * asym.positions.col(i);
    cif +=
        fmt::format("{} {} {:.17g} {:.17g} {:.17g} {:.17g}\n", asym.labels[i],
                    occ::core::Element(asym.atomic_numbers(i)).symbol(), x(0),
                    x(1), x(2), asym.occupations(i));
  }

  occ::io::CifParser parser;
  const auto transformed = parser.parse_crystal_from_string(cif);
  INFO(parser.failure_description());
  REQUIRE(transformed.has_value());
  CHECK(transformed->unit_cell_atoms().size() ==
        2 * original.unit_cell_atoms().size());

  // The CIF carries no displacement parameters, so give both the same.
  RinseParams params{};
  params.fixed_uiso = 0.02;
  const Rinse rinse(params);
  const occ::Vec difference =
      rinse.compute(original) - rinse.compute(*transformed);
  CHECK(difference.cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("RINSE parameters derive the resolution cutoff", "[rinse]") {
  RinseParams params{};
  CHECK_THAT(params.q_max(), WithinAbs(0.7, 1e-15));
  CHECK_THAT(params.sin_theta_over_lambda_max(), WithinAbs(0.35, 1e-15));
  CHECK(params.num_l_levels() == 16);
  CHECK(params.size() == 128);
  CHECK(params.l_values().front() == 4);
  CHECK(params.l_values().back() == 34);

  // Shells are anchored by index, so adding one leaves the others alone.
  params.n_max = 16;
  CHECK_THAT(params.q_max(), WithinAbs(0.35 * std::cbrt(16.0), 1e-15));

  params.radial_basis = RinseRadialBasis::SmoothShellsCW;
  params.n_max = 8;
  CHECK_THAT(params.q_max(), WithinAbs(0.35 * 7, 1e-15));
}

TEST_CASE("RINSE rejects unusable parameters", "[rinse]") {
  CHECK_THROWS(RinseParams{.n_max = 0}.validate());
  CHECK_THROWS(RinseParams{.l_min = 3}.validate());
  CHECK_THROWS(RinseParams{.l_max = 35}.validate());
  CHECK_THROWS((RinseParams{.l_min = 8, .l_max = 8}).validate());
  CHECK_THROWS((RinseParams{.radial_scale = 0.0}).validate());
}

TEST_CASE("RINSE radial basis is anchored by shell index", "[rinse]") {
  const occ::Vec q = occ::Vec::LinSpaced(64, 0.0, 0.7);

  RinseParams eight{};
  RinseParams sixteen{};
  sixteen.n_max = 16;
  const occ::Mat small = rinse_radial_basis(q, eight);
  const occ::Mat large = rinse_radial_basis(q, sixteen);
  REQUIRE(small.cols() == 8);
  REQUIRE(large.cols() == 16);
  // The first eight shells are the same functions in both.
  CHECK((small - large.leftCols(8)).cwiseAbs().maxCoeff() == 0.0);

  RinseParams partition{};
  partition.radial_basis = RinseRadialBasis::SmoothShellsCW;
  const occ::Mat rows = rinse_radial_basis(q, partition);
  CHECK_THAT((rows.rowwise().sum().array() - 1.0).abs().maxCoeff(),
             WithinAbs(0.0, 1e-14));
}

TEST_CASE("Waasmaier-Kirfel form factors", "[rinse]") {
  using occ::crystal::xray_form_factor;

  // f(0) is the electron count for a neutral atom, near enough: the fit is not
  // constrained there, and drifts further with Z.
  for (const int z : {1, 6, 8, 16, 26, 92}) {
    const auto form_factor = xray_form_factor(z);
    REQUIRE(form_factor.has_value());
    INFO(fmt::format("Z = {}", z));
    CHECK_THAT(form_factor->at_stol_sq(0.0), WithinRel(double(z), 1e-3));
  }

  // Values from cctbx's wk1995 table, which occ has to reproduce exactly.
  CHECK_THAT(xray_form_factor("C")->at_stol_sq(0.25 * 0.7 * 0.7),
             WithinRel(2.173365251096842, 1e-15));
  CHECK_THAT(xray_form_factor("Na1+")->at_stol_sq(0.04),
             WithinRel(8.373787197777855, 1e-15));
  CHECK_THAT(xray_form_factor("O")->at_stol_sq(0.1),
             WithinRel(3.8810929100501563, 1e-15));

  CHECK(xray_form_factor("Unobtainium") == std::nullopt);
  CHECK(xray_form_factor(0) == std::nullopt);
  CHECK(xray_form_factor(120) == std::nullopt);
}

TEST_CASE("RINSE timing", "[.rinse-benchmark]") {
  const Rinse rinse{};
  const int threads_before = occ::parallel::get_num_threads();
  const auto report = [&](const std::string &label, const Crystal &crystal) {
    const auto reflections = rinse_reflections(crystal, rinse.parameters());
    const int repeats = 50;
    fmt::print("{:>22}  {:6d} refl  {:5d} atoms  ", label, reflections.size(),
               crystal.unit_cell_atoms().size());
    for (const int threads : {1, 8}) {
      occ::parallel::set_num_threads(threads);
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeats; i++)
        (void)rinse_reflections(crystal, rinse.parameters());
      auto t1 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeats; i++)
        (void)rinse.power_spectrum(reflections);
      auto t2 = std::chrono::steady_clock::now();
      const auto ms = [&](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count() /
               repeats;
      };
      fmt::print("| {}t: refl {:7.3f} ps {:7.3f} total {:7.3f} ms ", threads,
                 ms(t0, t1), ms(t1, t2), ms(t0, t2));
    }
    fmt::print("\n");
  };

  for (const auto &expected : rinse_reference::structures)
    report(expected.name, load(expected.name));

  const Crystal ylid = load("ylid.cif");
  report("ylid 2x2x2", Crystal::create_primitive_supercell(ylid, {2, 2, 2}));
  report("ylid 3x3x3", Crystal::create_primitive_supercell(ylid, {3, 3, 3}));

  occ::parallel::set_num_threads(threads_before);
}
