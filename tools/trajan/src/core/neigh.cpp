#include <stdexcept>
#include <trajan/core/frame.h>
#include <trajan/core/log.h>
#include <trajan/core/neigh.h>
#include <trajan/core/util.h>

namespace trajan::core {

NeighbourListBase::NeighbourListBase(const UnitCell &unit_cell, double rcut)
    : m_unit_cell(unit_cell), m_cutoff(rcut), m_cutoffsq(rcut * rcut) {}

CellList::CellList(const UnitCell &unit_cell, double cutoff)
    : NeighbourListBase(unit_cell, cutoff) {
  if (m_dummy_unit_cell) {
    trajan::log::debug("Using a dummy unit cell. This should only be the case "
                       "if no crystallographic information was input.");
    this->initialise_cells(0);
    return;
  }
  this->initialise_cells();
}

CellListParameters CellList::generate_cell_params(size_t ghost_cells) const {
  return CellListParameters(
      static_cast<int>(
          std::floor(m_unit_cell.a_vector().norm() / (m_cutoff / CELLDIVISOR))),
      static_cast<int>(
          std::floor(m_unit_cell.b_vector().norm() / (m_cutoff / CELLDIVISOR))),
      static_cast<int>(
          std::floor(m_unit_cell.c_vector().norm() / (m_cutoff / CELLDIVISOR))),
      ghost_cells);
}

void CellList::initialise_cells(size_t ghost_cells) {
  m_params = generate_cell_params(ghost_cells);

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
void CellList::update(const NeighbourListPacket &nlp) {
  m_current_nlp = nlp;
  if (m_dummy_unit_cell) {
    this->initialise_cells(0);
  }
  this->clear_cells();
  Mat3N frac_pos = nlp.wrapped_frac_pos;
  IVec inds_a = (frac_pos.row(0) * m_params.a).cast<int>();
  IVec inds_b = (frac_pos.row(1) * m_params.b).cast<int>();
  IVec inds_c = (frac_pos.row(2) * m_params.c).cast<int>();
  Mat3N cart_pos = nlp.wrapped_cart_pos;
  for (int ent_i = 0; ent_i < nlp.size(); ent_i++) {
    Vec3 ent_pos = cart_pos.col(ent_i);
    int ind_a = inds_a[ent_i];
    int ind_b = inds_b[ent_i];
    int ind_c = inds_c[ent_i];
    this->cell_at(ind_a + m_params.num_ghosts, ind_b + m_params.num_ghosts,
                  ind_c + m_params.num_ghosts)
        .add_entity(ent_i, ent_pos, nlp.obj_types[ent_i]);
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

    for (int ia = 0; ia <= std::abs(a); ia++) {
      for (int ib = 0; ib <= std::abs(b); ib++) {
        for (int ic = 0; ic <= std::abs(c); ic++) {
          if (ia == 0 && ib == 0 && ic == 0) {
            continue;
          }
          int a_shift = ia * a, b_shift = ib * b, c_shift = ic * c;
          Vec3 shift = m_unit_cell.direct() * Vec3(a_shift, b_shift, c_shift);
          Vec3 ent_shift = ent_pos + shift;
          this->cell_at(ind_a + a_shift * m_params.a + m_params.num_ghosts,
                        ind_b + b_shift * m_params.b + m_params.num_ghosts,
                        ind_c + c_shift * m_params.c + m_params.num_ghosts)
              .add_entity(ent_i, ent_shift, nlp.obj_types[ent_i]);
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
        if (m_current_nlp.check_presence) {
          bool is_cross_section = m_current_nlp.presence_tracker[ent1.idx] !=
                                  m_current_nlp.presence_tracker[ent2.idx];
          if (!is_cross_section) {
            continue;
          }
          if (m_current_nlp.are_same_entity(ent1.idx, ent2.idx)) {
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
          if (m_current_nlp.check_presence) {
            bool is_cross_section = m_current_nlp.presence_tracker[ent1.idx] !=
                                    m_current_nlp.presence_tracker[ent2.idx];
            if (!is_cross_section) {
              continue;
            }
            if (m_current_nlp.are_same_entity(ent1.idx, ent2.idx)) {
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

VerletList::VerletList(const UnitCell &unit_cell, double rcut)
    : NeighbourListBase(unit_cell, rcut) {}

void VerletList::update(const NeighbourListPacket &nlp) { m_current_nlp = nlp; }

void VerletList::verlet_loop(const NeighbourCallback &callback) const {
  for (size_t i = 0; i < m_current_nlp.size(); ++i) {
    for (size_t j = i + 1; j < m_current_nlp.size(); ++j) {
      if (m_current_nlp.check_presence) {
        bool is_cross_section = m_current_nlp.presence_tracker[i] !=
                                m_current_nlp.presence_tracker[j];
        if (!is_cross_section) {
          continue;
        }
        if (m_current_nlp.are_same_entity(i, j)) {
          continue;
        }
      }

      Vec3 ent_pos_i = m_current_nlp.wrapped_cart_pos.col(i);
      Vec3 ent_pos_j = m_current_nlp.wrapped_cart_pos.col(j);
      Vec3 dr = ent_pos_j - ent_pos_i;
      if (!m_dummy_unit_cell) {
        Vec3 frac_dr = m_unit_cell.to_fractional(dr);
        frac_dr = frac_dr.array() - frac_dr.array().round();
        dr = m_unit_cell.to_cartesian(frac_dr);
      }
      double rsq = dr.squaredNorm();
      if (rsq <= m_cutoffsq) {
        Entity ent_i(i, ent_pos_i, m_current_nlp.obj_types[i]);
        Entity ent_j(j, ent_pos_j, m_current_nlp.obj_types[j]);
        callback(ent_i, ent_j, rsq);
      }
    }
  }
};

void NeighbourList::update(const std::vector<Atom> &atoms) {
  size_t num_atoms = atoms.size();
  Mat3N cart_pos(3, num_atoms);
  std::vector<Entity::Type> obj_types(num_atoms, Entity::Type::Atom);
  for (size_t i = 0; i < num_atoms; i++) {
    const Atom &atom = atoms[i];
    cart_pos(0, i) = atom.x;
    cart_pos(1, i) = atom.y;
    cart_pos(2, i) = atom.z;
  }
  UnitCell uc = m_impl->unit_cell();
  bool dummy_unit_cell = m_impl->dummy_unit_cell();
  auto result = trajan::util::wrap_coordinates(cart_pos, uc, dummy_unit_cell);
  m_current_nlp = NeighbourListPacket(obj_types, result.second, result.first);

  m_impl->update(m_current_nlp);
}

void NeighbourList::base_update(const std::vector<EntityType> &og_objects,
                                Molecule::Origin o) {
  if (!m_init) {
    throw std::runtime_error("Need to initialise NeighbourList.");
  }
  // if (m_impl->dummy_unit_cell()) {
  //   throw std::runtime_error("Using uninitialised UnitCell.");
  // }

  size_t num_objects = og_objects.size();
  Mat3N cart_pos(3, num_objects);
  std::vector<Entity::Type> obj_types;
  obj_types.clear();
  obj_types.reserve(num_objects);
  for (size_t i = 0; i < num_objects; i++) {
    const EntityType &obj = og_objects[i];
    std::visit(
        [&](const auto &entity) {
          using T = std::decay_t<decltype(entity)>;
          if constexpr (std::is_same_v<T, Atom>) {
            cart_pos(0, i) = entity.x;
            cart_pos(1, i) = entity.y;
            cart_pos(2, i) = entity.z;
            obj_types.push_back(Entity::Type::Atom);
          } else if constexpr (std::is_same_v<T, Molecule>) {
            Vec3 O = {0, 0, 0};
            switch (o) {
            case Molecule::Cartesian:
              throw std::runtime_error(
                  "Can't use Cartesian origin for NeighbourList");
            case Molecule::Centroid:
              O = entity.centroid();
              break;
            case Molecule::CentreOfMass:
              O = entity.centre_of_mass();
              break;
            }
            cart_pos(0, i) = O.x();
            cart_pos(1, i) = O.y();
            cart_pos(2, i) = O.z();
            obj_types.push_back(Entity::Type::Molecule);
          }
        },
        obj);
  }
  trajan::log::debug("Number of cartesian positions from entities = {}",
                     cart_pos.cols());
  UnitCell uc = m_impl->unit_cell();
  bool dummy_unit_cell = m_impl->dummy_unit_cell();
  auto result = trajan::util::wrap_coordinates(cart_pos, uc, dummy_unit_cell);
  m_current_nlp = NeighbourListPacket(obj_types, result.second, result.first);
}

void NeighbourList::update(const std::vector<EntityType> &og_objects,
                           Molecule::Origin o) {
  this->base_update(og_objects, o);
  m_impl->update(m_current_nlp);
}

void NeighbourList::update(
    const std::vector<std::vector<EntityType>> &og_objects_vec,
    Molecule::Origin o) {
  // auto result = trajan::util::combine_deduplicate_map(
  //     og_objects_vec, core::VariantHash(), core::VariantEqual());
  // std::vector<EntityType> deduplicated_entities = result.first;
  // std::vector<std::bitset<8>> presence_tracker = result.second;
  // this->base_update(deduplicated_entities, o);
  // auto result = trajan::util::combine_map(og_objects_vec,
  // core::VariantHash(),
  //                                         core::VariantEqual());
  // std::vector<EntityType> combined_entities = result.first;
  // std::vector<std::bitset<8>> presence_tracker = result.second;
  // this->base_update(combined_entities, o);
  // m_current_nlp.presence_tracker = presence_tracker;
  // m_current_nlp.check_presence = true;

  auto [combined_entities, presence, canonical_map] =
      trajan::util::combine_map_check(og_objects_vec, core::VariantHash(),
                                      core::VariantEqual());
  this->base_update(combined_entities, o);
  m_current_nlp.presence_tracker = presence;
  m_current_nlp.check_presence = true;
  m_current_nlp.index_to_canonical = canonical_map;
  m_impl->update(m_current_nlp);
}

} // namespace trajan::core
