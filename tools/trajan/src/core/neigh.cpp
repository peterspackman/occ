#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/atom.h>
#include <trajan/core/frame.h>
#include <trajan/core/log.h>
#include <trajan/core/neigh.h>
#include <trajan/core/util.h>

namespace trajan::core {

using occ::IVec;

NeighbourListPacket::NeighbourListPacket(const std::vector<Atom> &atoms) {
  size_t num_atoms = atoms.size();
  cart_pos.resize(3, num_atoms);
  entity_types.resize(num_atoms, Entity::Type::Atom);
  for (size_t i = 0; i < num_atoms; i++) {
    const Atom &atom = atoms[i];
    cart_pos(0, i) = atom.x;
    cart_pos(1, i) = atom.y;
    cart_pos(2, i) = atom.z;
  }
}

NeighbourListPacket::NeighbourListPacket(const std::vector<Molecule> &molecules,
                                         Molecule::Origin o) {
  size_t num_molecules = molecules.size();
  cart_pos.resize(3, num_molecules);
  entity_types.resize(num_molecules, Entity::Type::Molecule);
  for (size_t i = 0; i < num_molecules; i++) {
    const Molecule &mol = molecules[i];
    Vec3 O = {0, 0, 0};
    switch (o) {
    case Molecule::Cartesian:
      throw std::runtime_error("Can't use Cartesian origin for NeighbourList");
    case Molecule::Centroid:
      O = mol.centroid();
      break;
    case Molecule::CenterOfMass:
      O = mol.center_of_mass();
      break;
    }
    cart_pos(0, i) = O.x();
    cart_pos(1, i) = O.y();
    cart_pos(2, i) = O.z();
  }
}

void NeighbourListPacket::initialise_from_entities(
    const std::vector<EntityVariant> &entities, Molecule::Origin o) {
  size_t num_entities = entities.size();
  cart_pos.resize(3, num_entities);
  entity_types.clear();
  entity_types.reserve(num_entities);
  for (size_t i = 0; i < num_entities; i++) {
    const EntityVariant &obj = entities[i];
    std::visit(
        [&](const auto &entity) {
          using T = std::decay_t<decltype(entity)>;
          if constexpr (std::is_same_v<T, Atom>) {
            cart_pos(0, i) = entity.x;
            cart_pos(1, i) = entity.y;
            cart_pos(2, i) = entity.z;
            entity_types.push_back(Entity::Type::Atom);
          } else if constexpr (std::is_same_v<T, Molecule>) {
            Vec3 O = {0, 0, 0};
            switch (o) {
            case Molecule::Cartesian:
              throw std::runtime_error(
                  "Can't use Cartesian origin for NeighbourList");
            case Molecule::Centroid:
              O = entity.centroid();
              break;
            case Molecule::CenterOfMass:
              O = entity.center_of_mass();
              break;
            }
            cart_pos(0, i) = O.x();
            cart_pos(1, i) = O.y();
            cart_pos(2, i) = O.z();
            entity_types.push_back(Entity::Type::Molecule);
          }
        },
        obj);
  }
  trajan::log::debug("Number of cartesian positions from entities = {}",
                     cart_pos.cols());
}
NeighbourListPacket::NeighbourListPacket(
    const std::vector<EntityVariant> &entities, Molecule::Origin o) {
  this->initialise_from_entities(entities, o);
}

NeighbourListPacket::NeighbourListPacket(
    const std::vector<std::vector<EntityVariant>> &entities_vectors,
    Molecule::Origin o) {
  auto [combined_entities, presence, canonical_map] =
      trajan::util::combine_map_check(entities_vectors,
                                      core::EntityVariantHash(),
                                      core::EntityVariantEqual());
  this->initialise_from_entities(combined_entities, o);
  presence_tracker = presence;
  check_presence = true;
  index_to_canonical = canonical_map;
}

void CellListPacket::initialise_from_unit_cell(
    const std::optional<UnitCell> &uc) {
  if (unit_cell.has_value()) {
    const UnitCell &uc = unit_cell.value();
    std::tie(wrapped_frac_pos, wrapped_cart_pos) =
        trajan::util::wrap_coordinates(cart_pos, uc);
    side_lengths[0] = uc.a_vector().norm();
    side_lengths[1] = uc.b_vector().norm();
    side_lengths[2] = uc.c_vector().norm();
    return;
  }
  Vec3 min_vals = cart_pos.rowwise().minCoeff();
  Vec3 max_vals = cart_pos.rowwise().maxCoeff();
  side_lengths = max_vals - min_vals;
  UnitCell dummy_uc = occ::crystal::orthorhombic_cell(
      side_lengths[0], side_lengths[1], side_lengths[2]);
  wrapped_cart_pos = cart_pos.colwise() - min_vals;
  wrapped_frac_pos = dummy_uc.to_fractional(wrapped_cart_pos);
}

CellListPacket::CellListPacket(const std::vector<Atom> &atoms,
                               const std::optional<UnitCell> &uc)
    : NeighbourListPacket(atoms), unit_cell(uc) {
  this->initialise_from_unit_cell(uc);
}

CellListPacket::CellListPacket(const std::vector<Molecule> &molecules,
                               const std::optional<UnitCell> &uc,
                               Molecule::Origin o)
    : NeighbourListPacket(molecules, o), unit_cell(uc) {
  this->initialise_from_unit_cell(uc);
}

CellListPacket::CellListPacket(const std::vector<EntityVariant> &entities,
                               const std::optional<UnitCell> &uc,
                               Molecule::Origin o)
    : NeighbourListPacket(entities, o), unit_cell(uc) {
  this->initialise_from_unit_cell(uc);
};

CellListPacket::CellListPacket(
    const std::vector<std::vector<EntityVariant>> &entities_vectors,
    const std::optional<UnitCell> &uc, Molecule::Origin o)
    : NeighbourListPacket(entities_vectors, o), unit_cell(uc) {
  this->initialise_from_unit_cell(uc);
};

NeighbourListBase::NeighbourListBase(double rcut)
    : m_cutoff(rcut), m_cutoffsq(rcut * rcut) {}

CellListParameters CellList::generate_cell_params(const Vec3 &side_lengths,
                                                  size_t ghost_cells) const {

  return CellListParameters(
      static_cast<int>(std::floor(side_lengths[0] / (m_cutoff / CELLDIVISOR))),
      static_cast<int>(std::floor(side_lengths[1] / (m_cutoff / CELLDIVISOR))),
      static_cast<int>(std::floor(side_lengths[2] / (m_cutoff / CELLDIVISOR))),
      ghost_cells);
}

void CellList::initialise_cells(const Vec3 &side_lengths, size_t ghost_cells) {
  m_params = this->generate_cell_params(side_lengths, ghost_cells);

  m_cells.resize(m_params.total);
  for (size_t i = 0; i < m_params.total; i++) {
    const size_t a = i / (m_params.total_b * m_params.total_c);
    const size_t b =
        (i % (m_params.total_b * m_params.total_c)) / m_params.total_c;
    const size_t c = i % m_params.total_c;
    m_cells[i].m_index = CellIndex{a, b, c};
  }
  const int adj = static_cast<int>(m_params.num_ghosts);
  for (int a = m_params.num_ghosts; a < m_params.a_end; ++a) {
    for (int b = m_params.num_ghosts; b < m_params.b_end; ++b) {
      for (int c = m_params.num_ghosts; c < m_params.c_end; ++c) {
        std::vector<size_t> neighs;
        neighs.reserve(m_params.num_neighs);
        for (int na = -adj; na <= adj; ++na) {
          for (int nb = -adj; nb <= adj; ++nb) {
            for (int nc = -adj; nc <= adj; ++nc) {
              if (na < 0 || (na == 0 && nb < 0) ||
                  (na == 0 && nb == 0 && nc <= 0)) {
                continue;
              }
              neighs.push_back(linear_index(static_cast<size_t>(a + na),
                                            static_cast<size_t>(b + nb),
                                            static_cast<size_t>(c + nc)));
            }
          }
        }
        m_cell_neighs.insert(std::make_pair(linear_index(a, b, c), neighs));
      }
    }
  }
  m_cell_indices.reserve(m_params.total_real);
  for (const auto &pair : m_cell_neighs) {
    m_cell_indices.push_back(pair.first);
  }
}

void CellList::clear_cells() {
  for (Cell &cell : m_cells) {
    cell.clear();
  }
}
void CellList::update_impl(const CellListPacket &clp) {
  this->clear_cells();
  m_clp = clp;
  this->initialise_cells(m_clp.side_lengths,
                         m_clp.unit_cell.has_value() ? GHOSTCELLS : 0);
  Mat3N &frac_pos = m_clp.wrapped_frac_pos;
  IVec inds_a = (frac_pos.row(0) * m_params.a).cast<int>();
  IVec inds_b = (frac_pos.row(1) * m_params.b).cast<int>();
  IVec inds_c = (frac_pos.row(2) * m_params.c).cast<int>();
  Mat3N &cart_pos = m_clp.wrapped_cart_pos;
  for (int ent_i = 0; ent_i < m_clp.size(); ent_i++) {
    Vec3 ent_pos = cart_pos.col(ent_i);
    int ind_a = inds_a[ent_i];
    int ind_b = inds_b[ent_i];
    int ind_c = inds_c[ent_i];
    this->cell_at(ind_a + m_params.num_ghosts, ind_b + m_params.num_ghosts,
                  ind_c + m_params.num_ghosts)
        .add_entity(ent_i, ent_pos, m_clp.entity_types[ent_i]);
    if (!m_clp.unit_cell.has_value()) {
      continue;
    }
    if (ind_a >= m_params.num_ghosts && ind_a < m_params.a_upper &&
        ind_b >= m_params.num_ghosts && ind_b < m_params.b_upper &&
        ind_c >= m_params.num_ghosts && ind_c < m_params.c_upper) {
      continue;
    }
    int a = (ind_a < m_params.num_ghosts)
                ? 1
                : (ind_a >= m_params.a_upper ? -1 : 0);
    int b = (ind_b < m_params.num_ghosts)
                ? 1
                : (ind_b >= m_params.b_upper ? -1 : 0);
    int c = (ind_c < m_params.num_ghosts)
                ? 1
                : (ind_c >= m_params.c_upper ? -1 : 0);

    const auto &direct_matrix = m_clp.unit_cell.value().direct();
    for (int ia = 0; ia <= std::abs(a); ia++) {
      for (int ib = 0; ib <= std::abs(b); ib++) {
        for (int ic = 0; ic <= std::abs(c); ic++) {
          if (ia == 0 && ib == 0 && ic == 0) {
            continue;
          }
          int a_shift = ia * a, b_shift = ib * b, c_shift = ic * c;
          Vec3 shift = direct_matrix * Vec3(a_shift, b_shift, c_shift);
          Vec3 ent_shift = ent_pos + shift;
          this->cell_at(ind_a + a_shift * m_params.a + m_params.num_ghosts,
                        ind_b + b_shift * m_params.b + m_params.num_ghosts,
                        ind_c + c_shift * m_params.c + m_params.num_ghosts)
              .add_entity(ent_i, ent_shift, m_clp.entity_types[ent_i]);
        }
      }
    }
  }
}

void CellList::cell_loop(const NeighbourCallback &callback) const {
  for (size_t cell_i = 0; cell_i < m_params.total_real; cell_i++) {
    size_t center_cell_idx = m_cell_indices[cell_i];
    auto center_entities = m_cells[center_cell_idx].get_entities();
    for (size_t i = 0; i < center_entities.size(); ++i) {
      const Entity &ent1 = center_entities[i];
      for (size_t j = i + 1; j < center_entities.size(); ++j) {
        const Entity &ent2 = center_entities[j];
        if (m_clp.check_presence) {
          bool is_cross_section = m_clp.presence_tracker[ent1.index] !=
                                  m_clp.presence_tracker[ent2.index];
          if (!is_cross_section) {
            continue;
          }
          if (m_clp.are_same_entity(ent1.index, ent2.index)) {
            continue;
          }
        }
        double rsq = ent1.square_distance(ent2);
        if (rsq <= m_cutoffsq) {
          callback(ent1, ent2, rsq);
        }
      }
    }
    std::vector<size_t> neighs = m_cell_neighs.at(center_cell_idx);
    for (const size_t &neigh_idx : neighs) {
      auto neigh_entities = m_cells[neigh_idx].get_entities();
      for (const Entity &ent1 : center_entities) {
        for (const Entity &ent2 : neigh_entities) {
          if (m_clp.check_presence) {
            bool is_cross_section = m_clp.presence_tracker[ent1.index] !=
                                    m_clp.presence_tracker[ent2.index];
            if (!is_cross_section) {
              continue;
            }
            if (m_clp.are_same_entity(ent1.index, ent2.index)) {
              continue;
            }
          }
          double rsq = ent1.square_distance(ent2);
          if (rsq <= m_cutoffsq) {
            callback(ent1, ent2, rsq);
          }
        }
      }
    }
  }
}

void VerletList::update_impl(const VerletListPacket &vlp) { m_vlp = vlp; }

void VerletList::verlet_loop(const NeighbourCallback &callback) const {
  for (size_t i = 0; i < m_vlp.size(); ++i) {
    for (size_t j = i + 1; j < m_vlp.size(); ++j) {
      if (m_vlp.check_presence) {
        bool is_cross_section =
            m_vlp.presence_tracker[i] != m_vlp.presence_tracker[j];
        if (!is_cross_section) {
          continue;
        }
        if (m_vlp.are_same_entity(i, j)) {
          continue;
        }
      }

      Vec3 ent_pos_i = m_vlp.cart_pos.col(i);
      Vec3 ent_pos_j = m_vlp.cart_pos.col(j);
      Vec3 dr = ent_pos_j - ent_pos_i;
      if (m_vlp.unit_cell.has_value()) {
        const auto &uc = m_vlp.unit_cell.value();
        Vec3 frac_dr = uc.to_fractional(dr);
        frac_dr = frac_dr.array() - frac_dr.array().round();
        dr = uc.to_cartesian(frac_dr);
      }
      double rsq = dr.squaredNorm();
      if (rsq <= m_cutoffsq) {
        Entity ent_i(i, ent_pos_i, m_vlp.entity_types[i]);
        Entity ent_j(j, ent_pos_j, m_vlp.entity_types[j]);
        callback(ent_i, ent_j, rsq);
      }
    }
  }
};

void NeighbourList::update(const std::vector<Atom> &atoms,
                           const std::optional<UnitCell> &uc) {
  this->update_impl(atoms, uc);
}

void NeighbourList::update(const std::vector<Molecule> &molecules,
                           const std::optional<UnitCell> &uc,
                           Molecule::Origin o) {
  this->update_impl(molecules, uc, o);
}

void NeighbourList::update(const std::vector<EntityVariant> &entities,
                           const std::optional<UnitCell> &uc,
                           Molecule::Origin o) {
  this->update_impl(entities, uc, o);
}

void NeighbourList::update(
    const std::vector<std::vector<EntityVariant>> &entities_vectors,
    const std::optional<UnitCell> &uc, Molecule::Origin o) {
  this->update_impl(entities_vectors, uc, o);
}

} // namespace trajan::core
