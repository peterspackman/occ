#pragma once
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <trajan/core/atom.h>
#include <trajan/core/log.h>
#include <trajan/core/molecule.h>
#include <trajan/core/neigh.h>
#include <trajan/core/util.h>
#include <variant>
#include <vector>

namespace trajan::io {

using Atom = trajan::core::Atom;
using Molecule = trajan::core::Molecule;

struct SelectionBase {
  virtual std::string name() const = 0;
  virtual ~SelectionBase() = default;
};

template <typename T> struct Selection : public SelectionBase {
  Selection(std::vector<T> d) { data = std::move(d); }
  std::vector<T> data;

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }

  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }
};

struct AtomIndexSelection : public Selection<int> {
  using Selection<int>::Selection;
  std::string name() const override { return "AtomIndexSelection"; }
};

struct AtomTypeSelection : public Selection<std::string> {
  using Selection<std::string>::Selection;
  std::string name() const override { return "AtomTypeSelection"; }
};

struct MoleculeIndexSelection : public Selection<int> {
  using Selection<int>::Selection;
  std::string name() const override { return "MoleculeIndexSelection"; }
};

struct MoleculeTypeSelection : public Selection<std::string> {
  using Selection<std::string>::Selection;
  std::string name() const override { return "MoleculeTypeSelection"; }
};

using SelectionCriteria =
    std::variant<AtomIndexSelection, AtomTypeSelection, MoleculeIndexSelection,
                 MoleculeTypeSelection>;

template <typename T> struct SelectionTraits {};

template <> struct SelectionTraits<AtomIndexSelection> {
  using value_type = int;
  static constexpr bool allows_ranges = true;
  static constexpr char prefix = 'i';

  static std::optional<value_type> validate(const std::string &token) {
    try {
      int value = std::stoi(token);
      return value;
    } catch (...) {
      return std::nullopt;
    }
  }
};

template <> struct SelectionTraits<MoleculeIndexSelection> {
  using value_type = int;
  static constexpr bool allows_ranges = true;
  static constexpr char prefix = 'j';

  static std::optional<value_type> validate(const std::string &token) {
    try {
      int value = std::stoi(token);
      return value >= 0 ? std::optional<value_type>(value) : std::nullopt;
    } catch (...) {
      return std::nullopt;
    }
  }
};

template <> struct SelectionTraits<AtomTypeSelection> {
  using value_type = std::string;
  static constexpr bool allows_ranges = false;
  static constexpr char prefix = 'a';

  static std::optional<value_type> validate(const std::string &token) {
    if (std::all_of(token.begin(), token.end(), [](char c) {
          return std::isalnum(c) || c == '_' || c == '*';
        })) {
      return token;
    }
    return std::nullopt;
  }
};

template <> struct SelectionTraits<MoleculeTypeSelection> {
  using value_type = std::string;
  static constexpr bool allows_ranges = false;
  static constexpr char prefix = 'm';

  static std::optional<value_type> validate(const std::string &token) {
    if (std::all_of(token.begin(), token.end(), [](char c) {
          return std::isalnum(c) || c == '_' || c == '*';
        })) {
      return token;
    }
    return std::nullopt;
  }
};

class SelectionParser {
public:
  static std::optional<std::vector<SelectionCriteria>>
  parse(const std::string &input);

private:
  template <typename SelectionType>
  static std::optional<SelectionCriteria>
  parse_selection(const std::string &input);

  template <typename SelectionType>
  static std::optional<
      std::pair<typename SelectionTraits<SelectionType>::value_type,
                typename SelectionTraits<SelectionType>::value_type>>
  parse_range(const std::string &input) {
    auto parts = trajan::util::split_string(input, '-');
    if (parts.size() != 2)
      return std::nullopt;

    auto start = SelectionTraits<SelectionType>::validate(parts[0]);
    auto end = SelectionTraits<SelectionType>::validate(parts[1]);

    if (!start || !end || *start > *end)
      return std::nullopt;
    return std::make_pair(*start, *end);
  }
};

template <typename SelectionType>
std::vector<core::EntityVariant>
process_selection(const SelectionCriteria &selection,
                  const std::vector<Atom> &atoms,
                  const std::vector<Molecule> &molecules,
                  std::vector<core::EntityVariant> &entities) {

  std::visit(
      [&](const auto &sel) {
        using SelType = std::decay_t<decltype(sel)>;
        trajan::log::debug("Processing selection of type {}", sel.name());

        if constexpr (std::is_same_v<SelType, io::AtomIndexSelection>) {
          for (const Atom &atom : atoms) {
            for (const int &ai : sel) {
              if (atom.index == ai) {
                entities.push_back(atom);
              }
            }
          }
        } else if constexpr (std::is_same_v<SelType, io::AtomTypeSelection>) {
          for (const Atom &atom : atoms) {
            for (const std::string &at : sel) {
              if (atom.type == at) {
                entities.push_back(atom);
              }
            }
          }
        } else if constexpr (std::is_same_v<SelType,
                                            io::MoleculeIndexSelection>) {
          for (const Molecule &molecule : molecules) {
            for (const int &mi : sel) {
              if (molecule.index == mi) {
                entities.push_back(molecule);
              }
            }
          }
        } else if constexpr (std::is_same_v<SelType,
                                            io::MoleculeTypeSelection>) {
          for (const Molecule &molecule : molecules) {
            for (const std::string &mt : sel) {
              if (molecule.type == mt) {
                entities.push_back(molecule);
              }
            }
          }
        }
      },
      selection);
  return entities;
};

template <typename SelectionType>
std::vector<core::EntityVariant>
process_selection(const std::vector<SelectionCriteria> &selections,
                  const std::vector<Atom> &atoms,
                  const std::vector<Molecule> &molecules,
                  std::vector<core::EntityVariant> &entities) {

  for (const auto &selection : selections) {
    process_selection<SelectionType>(selection, atoms, molecules, entities);
  }

  // Remove duplicates based on entity type and index
  std::sort(entities.begin(), entities.end(), [](const auto &a, const auto &b) {
    auto get_sort_key = [](const auto &entity) {
      return std::visit(
          [](const auto &e) {
            return std::make_pair(typeid(e).hash_code(), e.index);
          },
          entity);
    };
    return get_sort_key(a) < get_sort_key(b);
  });

  entities.erase(
      std::unique(entities.begin(), entities.end(),
                  [](const auto &a, const auto &b) {
                    auto is_same_type_and_index = [](const auto &a_entity,
                                                     const auto &b_entity) {
                      return std::visit(
                          [&](const auto &a_val, const auto &b_val) {
                            using TypeA = std::decay_t<decltype(a_val)>;
                            using TypeB = std::decay_t<decltype(b_val)>;
                            if constexpr (std::is_same_v<TypeA, TypeB>) {
                              return a_val.index == b_val.index;
                            }
                            return false;
                          },
                          a_entity, b_entity);
                    };
                    return is_same_type_and_index(a, b);
                  }),
      entities.end());

  return entities;
}

inline auto selection_validator(
    std::vector<SelectionCriteria> &parsed_sel,
    std::optional<std::vector<char>> restrictions = std::nullopt) {

  return [parsed_sel = &parsed_sel, restrictions](const std::string &input) {
    auto result = SelectionParser::parse(input);
    if (!result) {
      return std::string("Invalid selection format");
    }
    if (restrictions.has_value()) {
      for (const auto &sel : result.value()) {
        std::string error;
        std::visit(
            [restrictions = &restrictions.value(), &error](const auto &s) {
              using SelType = std::decay_t<decltype(s)>;
              const char p = SelectionTraits<SelType>::prefix;
              if (std::find(restrictions->begin(), restrictions->end(), p) ==
                  restrictions->end()) {
                error = fmt::format("Selection prefix '{}' is not allowed.", p);
              }
            },
            sel);
        if (!error.empty()) {
          return error;
        }
      }
    }
    *parsed_sel = result.value();
    return std::string();
  };
};

} // namespace trajan::io
