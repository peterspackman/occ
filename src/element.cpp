#include "element.h"
#include <algorithm>
#include <sstream>
#include "util.h"

namespace craso::chem {

Element::Element(int atomicNumber) : m_data(ELEMENTDATA_TABLE[atomicNumber])
{}

Element::Element(const std::string& s) : m_data(ELEMENTDATA_TABLE[0])
{
    //capitalize the symbol first
    auto symbol = craso::util::trim_copy(s);
    auto capitalized = craso::util::capitalize_copy(s);
    for(size_t i = 1; i < ELEMENT_MAX; i++) {
        const auto dat = ELEMENTDATA_TABLE[i];
        if(symbol == dat.symbol) {
            m_data = dat;
            return;
        }
    }
}

std::string chemical_formula(const std::vector<Element>& els)
{
    std::vector<Element> el_sorted;
    for(const auto& el : els) el_sorted.push_back(el);

    std::sort(el_sorted.begin(), el_sorted.end());
    std::string result;
    int count = 1;
    std::string symbol;
    for(const auto& el : el_sorted) {
        if(el.symbol() == symbol) count++;
        else {
            result += symbol;
            if(count > 1) result += std::to_string(count);
            count = 1;
            symbol = el.symbol();
        }
    }
    result += symbol;
    if(count > 1) result += std::to_string(count);
    return result;
}

}
