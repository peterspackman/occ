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
    } else if constexpr (std::is_same_v<SelectionType, MoleculeTypeSelection>) {
      return SelectionCriteria{MoleculeTypeSelection{std::move(values)}};
    }
  } catch (...) {
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<std::vector<SelectionCriteria>>
SelectionParser::parse(const std::string &input) {
  if (input.empty())
    return std::nullopt;

  std::vector<SelectionCriteria> results;
  std::string current_token;
  char current_prefix = '\0';

  for (size_t i = 0; i < input.length(); i++) {
    char c = input[i];

    // Check if this is a prefix character
    if ((c == SelectionTraits<AtomIndexSelection>::prefix ||
         c == SelectionTraits<AtomTypeSelection>::prefix ||
         c == SelectionTraits<MoleculeIndexSelection>::prefix ||
         c == SelectionTraits<MoleculeTypeSelection>::prefix) &&
        (i == 0 || input[i - 1] == ',')) {

      // Process previous token if exists
      if (!current_token.empty() && current_prefix != '\0') {
        std::optional<SelectionCriteria> result;
        switch (current_prefix) {
        case SelectionTraits<AtomIndexSelection>::prefix:
          result = parse_selection<AtomIndexSelection>(current_token);
          break;
        case SelectionTraits<AtomTypeSelection>::prefix:
          result = parse_selection<AtomTypeSelection>(current_token);
          break;
        case SelectionTraits<MoleculeIndexSelection>::prefix:
          result = parse_selection<MoleculeIndexSelection>(current_token);
          break;
        case SelectionTraits<MoleculeTypeSelection>::prefix:
          result = parse_selection<MoleculeTypeSelection>(current_token);
          break;
        }
        if (!result)
          return std::nullopt;
        results.push_back(*result);
        current_token.clear();
      }
      current_prefix = c;
    } else {
      current_token += c;
    }
  }

  // Process final token
  if (!current_token.empty() && current_prefix != '\0') {
    std::optional<SelectionCriteria> result;
    switch (current_prefix) {
    case SelectionTraits<AtomIndexSelection>::prefix:
      result = parse_selection<AtomIndexSelection>(current_token);
      break;
    case SelectionTraits<AtomTypeSelection>::prefix:
      result = parse_selection<AtomTypeSelection>(current_token);
      break;
    case SelectionTraits<MoleculeIndexSelection>::prefix:
      result = parse_selection<MoleculeIndexSelection>(current_token);
      break;
    }
    if (!result)
      return std::nullopt;
    results.push_back(*result);
  }

  return results.empty() ? std::nullopt : std::make_optional(results);
}
} // namespace trajan::io
