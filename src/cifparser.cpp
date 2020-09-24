#include "cifparser.h"
#include "element.h"
#include "util.h"
#include <gemmi/numb.hpp>
#include <iostream>

namespace craso::io {

CifParser::CifParser() {}

void CifParser::extract_atom_sites(const gemmi::cif::Loop &loop) {
  int label_idx = loop.find_tag("_atom_site_label");
  int symbol_idx = loop.find_tag("_atom_site_type_symbol");
  int x_idx = loop.find_tag("_atom_site_fract_x");
  int y_idx = loop.find_tag("_atom_site_fract_y");
  int z_idx = loop.find_tag("_atom_site_fract_z");
  for (size_t i = 0; i < loop.length(); i++) {
    AtomData atom;
    if (label_idx >= 0)
      atom.site_label = loop.val(i, label_idx);
    if (symbol_idx >= 0)
      atom.element = loop.val(i, symbol_idx);
    if (x_idx >= 0)
      atom.position[0] = gemmi::cif::as_number(loop.val(i, x_idx));
    if (y_idx >= 0)
      atom.position[1] = gemmi::cif::as_number(loop.val(i, y_idx));
    if (z_idx >= 0)
      atom.position[2] = gemmi::cif::as_number(loop.val(i, z_idx));
    m_atoms.push_back(atom);
  }
}

void CifParser::extract_cell_parameter(const gemmi::cif::Pair &pair) {
  const auto &tag = pair.front();
  if (tag == "_cell_length_a")
    m_cell.a = gemmi::cif::as_number(pair.back());
  else if (tag == "_cell_length_b")
    m_cell.b = gemmi::cif::as_number(pair.back());
  else if (tag == "_cell_length_c")
    m_cell.c = gemmi::cif::as_number(pair.back());
  else if (tag == "_cell_angle_alpha")
    m_cell.alpha = craso::util::deg2rad(gemmi::cif::as_number(pair.back()));
  else if (tag == "_cell_angle_beta")
    m_cell.beta = craso::util::deg2rad(gemmi::cif::as_number(pair.back()));
  else if (tag == "_cell_angle_gamma")
    m_cell.gamma = craso::util::deg2rad(gemmi::cif::as_number(pair.back()));
}

void CifParser::extract_symmetry_operations(const gemmi::cif::Loop &loop) {
  int idx = loop.find_tag("_symmetry_equiv_pos_as_xyz");
  if (idx < 0)
    return;

  for (size_t i = 0; i < loop.length(); i++) {
    std::string symop = loop.val(i, idx);
    m_sym.symops.push_back(symop);
  }
}

std::optional<craso::crystal::Crystal>
CifParser::parse_crystal(const std::string &filename) {
  try {
    auto doc = gemmi::cif::read_file(filename);
    auto block = doc.blocks.front();
    for (const auto &item : block.items) {
      if (item.type == gemmi::cif::ItemType::Pair) {
        if (item.has_prefix("_cell"))
          extract_cell_parameter(item.pair);
      }
      if (item.type == gemmi::cif::ItemType::Loop) {
        if (item.has_prefix("_atom_site"))
          extract_atom_sites(item.loop);
        else if (item.has_prefix("_symmetry_equiv_pos"))
          extract_symmetry_operations(item.loop);
      }
    }
    if (!cell_valid()) {
      m_failure_desc = "Missing unit cell data";
      return std::nullopt;
    }
    if (!symmetry_valid()) {
      m_failure_desc = "Missing symmetry data";
      return std::nullopt;
    }
    craso::crystal::AsymmetricUnit asym;
    if (num_atoms() > 0) {
      asym.atomic_numbers.conservativeResize(num_atoms());
      asym.positions.conservativeResize(3, num_atoms());
      int i = 0;

      for (const auto &atom : m_atoms) {
        asym.positions(0, i) = atom.position[0];
        asym.positions(1, i) = atom.position[1];
        asym.positions(2, i) = atom.position[2];
        asym.atomic_numbers(i) = craso::chem::Element(atom.element).n();
        asym.labels.push_back(atom.site_label);
        i++;
      }
    }
    craso::crystal::UnitCell uc(m_cell.a, m_cell.b, m_cell.c, m_cell.alpha,
                                m_cell.beta, m_cell.gamma);
    std::vector<craso::crystal::SymmetryOperation> symops;
    if (m_sym.num_symops() > 0) {
      for (const auto &s : m_sym.symops) {
        symops.push_back(craso::crystal::SymmetryOperation(s));
      }
    }
    if (symops.size() > 0) {
      auto sg = craso::crystal::SpaceGroup(symops);
      return craso::crystal::Crystal(asym, sg, uc);
    }
    craso::crystal::SpaceGroup sg("P 1");
    return craso::crystal::Crystal(asym, sg, uc);
  } catch (const std::exception &e) {
    m_failure_desc = e.what();
    return std::nullopt;
  }
}

} // namespace craso::io
