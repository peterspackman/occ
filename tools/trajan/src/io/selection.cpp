#include <string>
#include <trajan/core/util.h>
#include <trajan/io/selection.h>

namespace trajan::io {

template <typename SelectionType>
std::optional<SelectionCriteria>
SelectionParser::parse_selection(const std::string &input) {
  using Traits = SelectionTraits<SelectionType>;
  using ValueType = typename Traits::value_type;

  try {
    std::vector<ValueType> values;
    auto tokens = trajan::util::split_string(input, ',');

    for (const auto &token : tokens) {
      if (token.find('-') != std::string::npos) {
        if (!Traits::allows_ranges)
          return std::nullopt;

        auto range = parse_range<SelectionType>(token);
        if (!range)
          return std::nullopt;

        if constexpr (std::is_integral_v<ValueType>) {
          for (ValueType i = range->first; i <= range->second; ++i) {
            values.push_back(i);
          }
        }
      } else {
        auto value = Traits::validate(token);
        if (!value)
          return std::nullopt;
        values.push_back(*value);
      }
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    if constexpr (std::is_same_v<SelectionType, AtomIndexSelection>) {
      return SelectionCriteria{AtomIndexSelection{std::move(values)}};
    } else if constexpr (std::is_same_v<SelectionType,
                                        MoleculeIndexSelection>) {
      return SelectionCriteria{MoleculeIndexSelection{std::move(values)}};
    } else if constexpr (std::is_same_v<SelectionType, AtomTypeSelection>) {
      return SelectionCriteria{AtomTypeSelection{std::move(values)}};
    }
  } catch (...) {
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<SelectionCriteria>
SelectionParser::parse(const std::string &input) {
  if (input.empty())
    return std::nullopt;

  switch (input[0]) {
  case SelectionTraits<AtomIndexSelection>::prefix:
    return parse_selection<AtomIndexSelection>(input.substr(1));
  case SelectionTraits<AtomTypeSelection>::prefix:
    return parse_selection<AtomTypeSelection>(input.substr(1));
  case SelectionTraits<MoleculeIndexSelection>::prefix:
    return parse_selection<MoleculeIndexSelection>(input.substr(1));
  default:
    return std::nullopt;
  }
}

} // namespace trajan::io
