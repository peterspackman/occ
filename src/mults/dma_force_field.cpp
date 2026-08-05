#include <algorithm>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/mults/dimer_interaction.h>
#include <occ/mults/dma_force_field.h>
#include <occ/mults/force_field_params.h>
#include <set>

namespace occ::mults {

namespace {

// The label a parameter set uses for a NEIGHCRYS type code. W99 is
// parameterised per code; FIT groups codes (every polar hydrogen is H_F2), so
// the label -- not the code -- is the identity a consumer sees.
std::string typed_label(const ShortRangeModel &model, int type_code) {
  if (model.atom_typing == "neighcrys-fit") {
    const char *label = ForceFieldParams::fit_type_label(type_code);
    return label ? label : "";
  }
  const char *label = ForceFieldParams::short_range_type_label(type_code);
  return label ? label : "";
}

} // namespace

occ::io::Basis
build_dma_force_field_basis(const occ::dma::DMASites &sites,
                            const std::vector<occ::dma::Mult> &multipoles,
                            const DMAForceFieldOptions &options) {
  const auto &model = short_range_model_from_string(options.force_field);

  occ::io::Basis basis;
  occ::io::MoleculeType mt;
  mt.name = options.molecule_name;

  const int n = static_cast<int>(multipoles.size());
  std::vector<int> atomic_numbers;
  std::vector<occ::Vec3> body_positions;
  atomic_numbers.reserve(n);
  body_positions.reserve(n);

  for (int i = 0; i < n; ++i) {
    occ::io::MoleculeSite ms;
    ms.label = sites.name[i];

    const int atom_index = sites.atom_indices(i);
    int z = 0;
    if (atom_index >= 0 && atom_index < static_cast<int>(sites.atoms.size()))
      z = sites.atoms[atom_index].atomic_number;
    ms.element = occ::core::Element(z).symbol();

    const occ::Vec3 pos_ang =
        sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
    ms.position = {pos_ang.x(), pos_ang.y(), pos_ang.z()};

    std::vector<double> flat(multipoles[i].num_components());
    for (size_t j = 0; j < flat.size(); ++j)
      flat[j] = multipoles[i].q(static_cast<int>(j));
    ms.multipoles = occ::io::SiteMultipoles::from_flat(flat);

    mt.sites.push_back(std::move(ms));
    atomic_numbers.push_back(z);
    body_positions.push_back(pos_ang);
  }

  // CSP programs expect the body frame centred on the molecular centre of mass.
  if (options.center_on_com) {
    occ::Vec3 com = occ::Vec3::Zero();
    double total_mass = 0.0;
    for (int i = 0; i < n; ++i) {
      const double mass = occ::core::Element(atomic_numbers[i]).mass();
      com += mass * body_positions[i];
      total_mass += mass;
    }
    if (total_mass > 0)
      com /= total_mass;
    for (auto &s : mt.sites) {
      s.position[0] -= com.x();
      s.position[1] -= com.y();
      s.position[2] -= com.z();
    }
    for (auto &p : body_positions)
      p -= com;
  }

  // Type codes are shared by every NEIGHCRYS-typed set; only the label and the
  // parameter table differ between them.
  std::vector<int> type_codes(n, 0);
  if (model.typed())
    type_codes =
        ForceFieldParams::classify_atom_types(atomic_numbers, body_positions);

  for (int i = 0; i < n; ++i) {
    std::string label = model.typed() ? typed_label(model, type_codes[i]) : "";
    // An atom the classifier could not place would otherwise be labelled
    // "UNKN" -- and two unclassified atoms of *different* elements would then
    // collide on that single type name downstream. Fall back to the element.
    if (label.empty() || label == "UNKN") {
      if (model.typed()) {
        occ::log::warn("{}: no {} atom type for site {} ({}); falling back to "
                       "the element symbol, so this site gets no short-range "
                       "parameters",
                       options.molecule_name, model.name, i, mt.sites[i].label);
      }
      label = mt.sites[i].element;
      type_codes[i] = 0;
    }
    mt.sites[i].type = label;
  }

  basis.molecule_types.push_back(std::move(mt));
  basis.potentials.force_field = model.name;
  basis.potentials.atom_typing = model.atom_typing;

  if (model.name == "none")
    return basis;

  // Pair potentials, restricted to the types actually present in this
  // molecule, once per unordered pair, in ascending order.
  const auto emit = [&](const std::string &l1, const std::string &l2, int z1,
                        int z2, const BuckinghamParams &params) {
    occ::io::BuckinghamPair bp;
    bp.types = {l1, l2};
    bp.elements = {occ::core::Element(z1).symbol(),
                   occ::core::Element(z2).symbol()};
    bp.A = params.A * occ::units::KJ_PER_MOL_TO_EV;
    bp.rho = 1.0 / params.B;
    bp.C6 = params.C * occ::units::KJ_PER_MOL_TO_EV;
    basis.potentials.buckingham.push_back(std::move(bp));
  };

  if (!model.typed()) {
    // Element-based: the type name is the element symbol.
    const auto params = ForceFieldParams::williams_de_params();
    const std::set<int> present(atomic_numbers.begin(), atomic_numbers.end());
    for (auto it1 = present.begin(); it1 != present.end(); ++it1) {
      for (auto it2 = it1; it2 != present.end(); ++it2) {
        const auto found = params.find({*it1, *it2});
        if (found == params.end())
          continue;
        emit(occ::core::Element(*it1).symbol(),
             occ::core::Element(*it2).symbol(), *it1, *it2, found->second);
      }
    }
    return basis;
  }

  const auto params = (model.atom_typing == "neighcrys-fit")
                          ? ForceFieldParams::fit_typed_params()
                          : ForceFieldParams::williams_typed_params();

  // Several codes can share one label (FIT lumps all polar hydrogens into
  // H_F2) and they carry identical parameters, so key on the label pair to
  // avoid emitting the same pair repeatedly.
  std::set<int> present_codes;
  for (int code : type_codes)
    if (code > 0)
      present_codes.insert(code);

  std::set<std::pair<std::string, std::string>> emitted;
  for (auto it1 = present_codes.begin(); it1 != present_codes.end(); ++it1) {
    for (auto it2 = it1; it2 != present_codes.end(); ++it2) {
      const auto found = params.find({*it1, *it2});
      if (found == params.end())
        continue;
      const auto l1 = typed_label(model, *it1);
      const auto l2 = typed_label(model, *it2);
      if (l1.empty() || l2.empty())
        continue;
      const auto key = std::minmax(l1, l2);
      if (!emitted.insert({key.first, key.second}).second)
        continue;
      emit(l1, l2, ForceFieldParams::short_range_type_atomic_number(*it1),
           ForceFieldParams::short_range_type_atomic_number(*it2),
           found->second);
    }
  }

  return basis;
}

} // namespace occ::mults
