#include <trajan/core/element.h>
#include <trajan/core/log.h>
#include <trajan/core/util.h>

namespace trajan::core {

namespace runtime_values {

ElementData *max_element =
    std::max_element(std::begin(ELEMENTDATA_TABLE), std::end(ELEMENTDATA_TABLE),
                     [](const ElementData &a, const ElementData &b) {
                       return a.covalent_radius < b.covalent_radius;
                     });
double max_cov_cutoff = max_element->covalent_radius;

} // namespace runtime_values

Element::Element(int num) : m_data(ELEMENTDATA_TABLE[0]) {
  if (num > ELEMENT_MAX) {
    trajan::log::debug("Could not get element with atomic number {}", num);
    return;
  }
  m_data = ELEMENTDATA_TABLE[num];
}

Element::Element(const std::string &s, bool exact_match)
    : m_data(ELEMENTDATA_TABLE[0]) {
  std::string symbol = trajan::util::trim_copy(s);
  size_t match_length = 0;
  auto it = std::find_if(
      std::rbegin(ELEMENTDATA_TABLE) + 1, std::rend(ELEMENTDATA_TABLE) - 1,
      [&](const ElementData &dat) {
        const size_t comp_length = dat.symbol.size();
        if (dat.symbol.compare(0, comp_length, symbol, 0, comp_length) == 0) {
          if (exact_match && dat.symbol != symbol) {
            return false;
          }
          if (dat.symbol.size() > match_length) {
            m_data = dat;
            match_length = dat.symbol.size();
          }
          return (symbol.size() == dat.symbol.size());
        }
        return false;
      });
}

std::string chemical_formula(const std::vector<Element> &els) {
  std::vector<Element> el_sorted = els;

  // put carbon first
  std::sort(el_sorted.begin(), el_sorted.end(),
            [](const Element &a, const Element &b) {
              if (a == b)
                return false;
              if (a == 6)
                return true;
              if (b == 6)
                return false;
              return a < b;
            });
  std::string result;
  int count = 1;
  std::string symbol;
  for (const auto &el : el_sorted) {
    if (el.symbol() == symbol)
      count++;
    else {
      result += symbol;
      if (count > 1)
        result += std::to_string(count);
      count = 1;
      symbol = el.symbol();
    }
  }
  result += symbol;
  if (count > 1)
    result += std::to_string(count);
  return result;
}

} // namespace trajan::core
