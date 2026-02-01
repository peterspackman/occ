#pragma once
#include <ankerl/unordered_dense.h>
#include <fmt/format.h>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/unitcell.h>
#include <stdexcept>
#include <trajan/core/atom.h>
#include <trajan/core/log.h>
#include <trajan/core/molecule.h>
#include <variant>
#include <vector>

namespace trajan::core {

using occ::Mat3N;
using occ::crystal::UnitCell;
using Atom = trajan::core::EnhancedAtom;
using Molecule = trajan::core::EnhancedMolecule;

struct Entity {
  int index;
  double x, y, z;
  enum class Type { Atom, Molecule } type;

  Entity(int index, const Vec3 &pos, Type t)
      : index(index), x(pos.x()), y(pos.y()), z(pos.z()), type(t) {}
  Entity(const Atom &atom)
      : index(atom.index), x(atom.x), y(atom.y), z(atom.z), type(Type::Atom) {}
  Entity(const Molecule &molecule, Molecule::Origin o = Molecule::CenterOfMass)
      : index(molecule.index), x(0.0), y(0.0), z(0.0), type(Type::Molecule) {
    Vec3 position = molecule.position(o);
    x = position[0];
    y = position[1];
    z = position[2];
  }

  inline Vec3 position() const { return {x, y, z}; }
  inline double square_distance(const Entity &other) const {
    double dx = other.x - x, dy = other.y - y, dz = other.z - z;
    return dx * dx + dy * dy + dz * dz;
  }

  bool operator==(const Atom &atom) const { return type == Type::Atom; }
  bool operator==(const Molecule &molecule) const {
    return type == Type::Molecule;
  }
};

using EntityVariant = std::variant<Atom, Molecule>;

struct EntityVariantHash {
  std::size_t operator()(const std::variant<Atom, Molecule> &var) const {
    if (std::holds_alternative<Atom>(var)) {
      const Atom &atom = std::get<Atom>(var);
      return std::hash<int>{}(atom.index);
    } else {
      const Molecule &molecule = std::get<Molecule>(var);
      return std::hash<int>{}(molecule.index);
    }
  }
};

struct EntityVariantEqual {
  bool operator()(const std::variant<Atom, Molecule> &lhs,
                  const std::variant<Atom, Molecule> &rhs) const {
    if (lhs.index() != rhs.index()) {
      return false;
    }

    if (std::holds_alternative<Atom>(lhs)) {
      return std::get<Atom>(lhs) == std::get<Atom>(rhs);
    } else {
      return std::get<Molecule>(lhs) == std::get<Molecule>(rhs);
    }
  }
};

template <typename T>
std::vector<T>
get_entities_of_type(const std::vector<EntityVariant> &entities) {
  std::vector<T> result;
  result.reserve(entities.size());
  for (const auto &e : entities) {
    if (std::holds_alternative<T>(e)) {
      result.push_back(e);
    }
  }
  return result;
}

struct NeighbourListPacket {
  Mat3N cart_pos;
  std::vector<Entity::Type> entity_types;

  std::vector<size_t> presence_tracker;
  bool check_presence{false};
  ankerl::unordered_dense::map<size_t, size_t> index_to_canonical;

  NeighbourListPacket() = default;

  NeighbourListPacket(const std::vector<Atom> &atoms);

  NeighbourListPacket(const std::vector<Molecule> &molecules,
                      Molecule::Origin o = Molecule::CenterOfMass);

  NeighbourListPacket(const std::vector<EntityVariant> &entities,
                      Molecule::Origin o = Molecule::CenterOfMass);

  NeighbourListPacket(
      const std::vector<std::vector<EntityVariant>> &entities_vectors,
      Molecule::Origin o = Molecule::CenterOfMass);

  size_t size() const { return entity_types.size(); }

  inline bool are_same_entity(size_t idx1, size_t idx2) const {
    auto it1 = index_to_canonical.find(idx1);
    auto it2 = index_to_canonical.find(idx2);

    if (it1 == index_to_canonical.end() || it2 == index_to_canonical.end()) {
      return false;
    }

    return it1->second == it2->second;
  }

private:
  void initialise_from_entities(const std::vector<EntityVariant> &entities,
                                Molecule::Origin o = Molecule::CenterOfMass);
};

struct CellListPacket : public NeighbourListPacket {
  Mat3N wrapped_cart_pos;
  Mat3N wrapped_frac_pos;
  Vec3 side_lengths;
  std::optional<UnitCell> unit_cell;

  CellListPacket() = default;

  CellListPacket(const std::vector<Atom> &atoms,
                 const std::optional<UnitCell> &uc);

  CellListPacket(const std::vector<Molecule> &molecules,
                 const std::optional<UnitCell> &uc,
                 Molecule::Origin o = Molecule::CenterOfMass);

  CellListPacket(const std::vector<EntityVariant> &entities,
                 const std::optional<UnitCell> &uc,
                 Molecule::Origin o = Molecule::CenterOfMass);

  CellListPacket(
      const std::vector<std::vector<EntityVariant>> &entities_vectors,
      const std::optional<UnitCell> &uc,
      Molecule::Origin o = Molecule::CenterOfMass);

private:
  void initialise_from_unit_cell(const std::optional<UnitCell> &uc);
};

struct VerletListPacket : public NeighbourListPacket {
  std::optional<UnitCell> unit_cell;

  VerletListPacket() = default;

  VerletListPacket(const std::vector<Atom> &atoms,
                   const std::optional<UnitCell> &uc)
      : NeighbourListPacket(atoms), unit_cell(uc) {};

  VerletListPacket(const std::vector<Molecule> &molecules,
                   const std::optional<UnitCell> &uc,
                   Molecule::Origin o = Molecule::CenterOfMass)
      : NeighbourListPacket(molecules, o), unit_cell(uc) {};

  VerletListPacket(const std::vector<EntityVariant> &entities,
                   const std::optional<UnitCell> &uc,
                   Molecule::Origin o = Molecule::CenterOfMass)
      : NeighbourListPacket(entities, o), unit_cell(uc) {};

  VerletListPacket(
      const std::vector<std::vector<EntityVariant>> &entities_vectors,
      const std::optional<UnitCell> &uc,
      Molecule::Origin o = Molecule::CenterOfMass)
      : NeighbourListPacket(entities_vectors, o), unit_cell(uc) {};
};

using NeighbourCallback =
    std::function<void(const Entity &, const Entity &, double)>;

// base class with common interface
class NeighbourListBase {
public:
  virtual ~NeighbourListBase() = default;

  NeighbourListBase(double cutoff);

  template <typename PacketType> void update(const PacketType &packet) {
    update_impl(packet);
  }

  virtual void iterate_neighbours(const NeighbourCallback &callback) const = 0;

  inline void update_cutoff(double rcut) {
    m_cutoff = rcut;
    m_cutoffsq = rcut * rcut;
  };

  const inline double &cutoff() const { return m_cutoff; }

protected:
  double m_cutoff = 10.0, m_cutoffsq = 100.0;

  virtual void update_impl(const CellListPacket &packet) {
    throw std::runtime_error(
        "CellListPacket not supported by this neighbour list type");
  }

  virtual void update_impl(const VerletListPacket &packet) {
    throw std::runtime_error(
        "VerletListPacket not supported by this neighbour list type");
  }
};

struct CellIndex {
  size_t a, b, c;
};

class CellList; // forward declaration

class Cell {
public:
  inline void add_entity(const Entity &entity) { m_entities.push_back(entity); }
  inline void add_entity(const size_t idx, Vec3 &pos, Entity::Type type) {
    m_entities.emplace_back(idx, pos, type);
  }
  const std::vector<Entity> &get_entities() const { return m_entities; }
  void clear() { m_entities.clear(); }

private:
  std::vector<Entity> m_entities;
  CellIndex m_index;
  friend class CellList;
};

struct CellListParameters {
  size_t num_ghosts;
  size_t num_neighs;
  size_t a, b, c;
  size_t total_a, total_b, total_c;
  size_t total;
  size_t total_real;
  size_t a_upper, b_upper, c_upper;
  size_t a_end, b_end, c_end;
  CellListParameters() = default;
  CellListParameters(size_t a, size_t b, size_t c, size_t num_ghosts)
      : num_ghosts(num_ghosts),
        num_neighs(((2 * num_ghosts + 1) * (2 * num_ghosts + 1) *
                        (2 * num_ghosts + 1) -
                    1) /
                   2),
        a(a), b(b), c(c), total_a(a + 2 * num_ghosts),
        total_b(b + 2 * num_ghosts), total_c(c + 2 * num_ghosts),
        total((a + 2 * num_ghosts) * (b + 2 * num_ghosts) *
              (c + 2 * num_ghosts)),
        total_real(a * b * c), a_end(a + num_ghosts), b_end(b + num_ghosts),
        c_end(c + num_ghosts), a_upper(a - num_ghosts), b_upper(b - num_ghosts),
        c_upper(c - num_ghosts) {}
};

// cell list algorithm for neighbours
class CellList : public NeighbourListBase {
public:
  CellList(double cutoff) : NeighbourListBase(cutoff) {};

  void iterate_neighbours(const NeighbourCallback &callback) const override {
    this->cell_loop(callback);
  };

protected:
  void update_impl(const CellListPacket &clp) override;

private:
  // TODO: test non-periodic
  static constexpr size_t CELLDIVISOR = 2;
  static constexpr size_t GHOSTCELLS = CELLDIVISOR;

  CellListPacket m_clp;
  CellListParameters m_params;
  std::vector<Cell> m_cells;
  ankerl::unordered_dense::map<size_t, std::vector<size_t>> m_cell_neighs;
  std::vector<size_t> m_cell_indices;

  inline size_t linear_index(size_t a, size_t b, size_t c) const {
    return (a * m_params.total_b * m_params.total_c) + (b * m_params.total_c) +
           c;
  }
  inline Cell &cell_at(size_t a, size_t b, size_t c) {
    return m_cells[linear_index(a, b, c)];
  }
  inline const Cell &cell_at(size_t a, size_t b, size_t c) const {
    return m_cells[linear_index(a, b, c)];
  }

  CellListParameters generate_cell_params(const Vec3 &side_lengths,
                                          size_t ghost_cells) const;
  void initialise_cells(const Vec3 &side_lengths, size_t ghost_cells);
  void clear_cells();
  void cell_loop(const NeighbourCallback &callback) const;
};

// verlet/double loop algorithm for neighbours
class VerletList : public NeighbourListBase {
public:
  VerletList(double cutoff) : NeighbourListBase(cutoff) {};

  void iterate_neighbours(const NeighbourCallback &callback) const override {
    this->verlet_loop(callback);
  };

protected:
  void update_impl(const VerletListPacket &vlp) override;

private:
  VerletListPacket m_vlp;
  void verlet_loop(const NeighbourCallback &callback) const;
};

// runtime-selectable NeighbourList class
class NeighbourList {
public:
  enum class Type { Cell, Verlet };

  NeighbourList() = default;
  NeighbourList(double cutoff, Type type = Type::Cell) {

    switch (type) {
    case Type::Cell:
      m_impl = std::make_unique<CellList>(cutoff);
      m_type = Type::Cell;
      break;
    case Type::Verlet:
      m_impl = std::make_unique<VerletList>(cutoff);
      m_type = Type::Verlet;
      break;
    }
  }

  void update(const std::vector<Atom> &atoms,
              const std::optional<UnitCell> &uc);

  void update(const std::vector<Molecule> &molecules,
              const std::optional<UnitCell> &uc,
              Molecule::Origin o = Molecule::CenterOfMass);

  void update(const std::vector<EntityVariant> &entities,
              const std::optional<UnitCell> &uc,
              Molecule::Origin o = Molecule::CenterOfMass);

  void update(const std::vector<std::vector<EntityVariant>> &entities_vectors,
              const std::optional<UnitCell> &uc,
              Molecule::Origin o = Molecule::CenterOfMass);

  void iterate_neighbours(const NeighbourCallback &callback) {
    m_impl->iterate_neighbours(callback);
  }

  void update_cutoff(double rcut) { m_impl->update_cutoff(rcut); }

  inline void update_type(Type type) {
    double current_cutoff = m_impl->cutoff();
    switch (type) {
    case Type::Cell:
      m_impl = std::make_unique<CellList>(current_cutoff);
      m_type = Type::Cell;
      break;
    case Type::Verlet:
      m_impl = std::make_unique<VerletList>(current_cutoff);
      m_type = Type::Verlet;
      break;
    }
  }

protected:
  template <typename... Args> void update_impl(Args &&...args) {
    switch (m_type) {
    case Type::Cell: {
      CellListPacket clp(std::forward<Args>(args)...);
      m_impl->update(clp);
      break;
    }
    case Type::Verlet: {
      VerletListPacket vlp(std::forward<Args>(args)...);
      m_impl->update(vlp);
      break;
    }
    }
  }

private:
  Type m_type = Type::Cell;
  std::unique_ptr<NeighbourListBase> m_impl;
};
}; // namespace trajan::core
