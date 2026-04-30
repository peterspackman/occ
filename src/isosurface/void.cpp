#include <occ/core/log.h>
#include <occ/isosurface/void.h>
#include <occ/slater/slaterbasis.h>

namespace occ::isosurface {

VoidSurfaceFunctor::VoidSurfaceFunctor(const occ::crystal::Crystal &crystal,
                                       float sep,
                                       const InterpolatorParams &params)
    : m_crystal(crystal), m_interpolator_params(params),
      m_target_separation(sep) {

  // Build a slab around the unit cell with at least m_buffer angstroms of
  // padding on each side; this slab carries enough periodic images that
  // density evaluation at any frac in [0, 1] is effectively periodic.
  double radius = m_buffer;
  const auto &uc_atoms = m_crystal.unit_cell_atoms();
  crystal::HKL upper = crystal::HKL::minimum();
  crystal::HKL lower = crystal::HKL::maximum();
  occ::Vec3 frac_radius = radius * 2 / m_crystal.unit_cell().lengths().array();

  for (size_t i = 0; i < uc_atoms.frac_pos.cols(); i++) {
    const auto &pos = uc_atoms.frac_pos.col(i);
    upper.h =
        std::max(upper.h, static_cast<int>(ceil(pos(0) + frac_radius(0))));
    upper.k =
        std::max(upper.k, static_cast<int>(ceil(pos(1) + frac_radius(1))));
    upper.l =
        std::max(upper.l, static_cast<int>(ceil(pos(2) + frac_radius(2))));

    lower.h =
        std::min(lower.h, static_cast<int>(floor(pos(0) - frac_radius(0))));
    lower.k =
        std::min(lower.k, static_cast<int>(floor(pos(1) - frac_radius(1))));
    lower.l =
        std::min(lower.l, static_cast<int>(floor(pos(2) - frac_radius(2))));
  }
  auto slab = m_crystal.slab(lower, upper);

  const auto &elements = slab.atomic_numbers;
  Mat3N coordinates = slab.cart_pos.array() * occ::units::ANGSTROM_TO_BOHR;

  m_molecule = occ::core::Molecule(elements, slab.cart_pos);

  auto basis = occ::slater::load_slaterbasis("thakkar");
  ankerl::unordered_dense::map<int, std::vector<int>> tmp_map;
  for (size_t i = 0; i < elements.rows(); i++) {
    int el = elements(i);
    tmp_map[el].push_back(i);
    auto search = m_interpolators.find(el);
    if (search == m_interpolators.end()) {
      auto b = basis[occ::core::Element(el).symbol()];
      auto func = [&b](float x) { return b.rho(std::sqrt(x)); };
      m_interpolators[el] = LinearInterpolatorFloat(
          func, m_interpolator_params.domain_lower,
          m_interpolator_params.domain_upper, m_interpolator_params.num_points);
    }
  }

  for (const auto &kv : tmp_map) {
    m_atom_interpolators.push_back(
        {m_interpolators.at(kv.first),
         coordinates(Eigen::all, kv.second).cast<float>()});
    m_atom_interpolators.back().threshold =
        m_atom_interpolators.back().interpolator.find_threshold(1e-8);
  }

  m_cell_list.build(m_atom_interpolators);
  occ::log::info("Void cell list: {} atoms in {} x {} x {} bins (size {:.2f} bohr)",
                 m_cell_list.num_atoms(), m_cell_list.dims()(0),
                 m_cell_list.dims()(1), m_cell_list.dims()(2),
                 m_cell_list.bin_size());

  update_region();
}

void VoidSurfaceFunctor::update_region() {
  // MC operates in fractional coordinates over [0, 1]^3.
  m_origin.setZero();
  m_side_length.setOnes();

  // Cache lattice transforms in bohr units. m_direct_bohr maps fractional ->
  // cartesian(bohr); the inverse-transpose maps fractional-space gradients to
  // cartesian-space gradients (used by MC's curvature path).
  m_direct_bohr =
      (m_crystal.unit_cell().direct() * occ::units::ANGSTROM_TO_BOHR)
          .cast<float>();
  m_normal_transform = m_direct_bohr.inverse().transpose();

  // Per-axis sample counts: target spacing along each lattice axis ~ sep
  // (in bohr). +1 because endpoint-inclusive sampling needs N+1 samples to
  // span N voxels exactly across [0, 1].
  Vec3 lengths_bohr =
      m_crystal.unit_cell().lengths() * occ::units::ANGSTROM_TO_BOHR;
  for (int i = 0; i < 3; i++) {
    int n = static_cast<int>(std::ceil(lengths_bohr(i) / m_target_separation));
    m_sample_counts(i) = std::max(2, n + 1);
  }

  occ::log::info("Void MC grid: {} x {} x {} samples (frac coords)",
                 m_sample_counts(0), m_sample_counts(1), m_sample_counts(2));
  occ::log::info("Lattice lengths (bohr): [{:.3f} {:.3f} {:.3f}]",
                 lengths_bohr(0), lengths_bohr(1), lengths_bohr(2));
  occ::log::info("Target separation: {:.3f} bohr", m_target_separation);
}

} // namespace occ::isosurface
