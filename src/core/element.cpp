#include <occ/core/element.h>
#include <occ/core/util.h>
#include <algorithm>
#include <sstream>

namespace occ::chem {

Element::Element(int atomicNumber) : m_data(ELEMENTDATA_TABLE[atomicNumber]) {}

Element::Element(const std::string &s, bool exact_match) : m_data(ELEMENTDATA_TABLE[0]) {
  // capitalize the symbol first
  auto symbol = occ::util::trim_copy(s);
  auto capitalized = occ::util::capitalize_copy(s);
  for (size_t i = ELEMENT_MAX - 1; i > 0; i--) {
    const auto dat = ELEMENTDATA_TABLE[i];
    const size_t N = dat.symbol.size();
    if (dat.symbol.compare(0, N, symbol, 0, N) == 0) {
        if(exact_match && dat.symbol != symbol) continue;
        m_data = dat;
        return;
    }
  }
}

std::string chemical_formula(const std::vector<Element> &els) {
  std::vector<Element> el_sorted;
  for (const auto &el : els)
    el_sorted.push_back(el);

  std::sort(el_sorted.begin(), el_sorted.end());
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

} // namespace occ::chem
