#include <algorithm>
#include <occ/core/element.h>
#include <occ/core/util.h>
#include <sstream>

namespace occ::core {

Element::Element(int atomicNumber) : m_data(ELEMENTDATA_TABLE[atomicNumber]) {}

Element::Element(const std::string &s, bool exact_match)
    : m_data(ELEMENTDATA_TABLE[0]) {
    // capitalize the symbol first
    auto symbol = occ::util::trim_copy(s);
    auto capitalized = occ::util::capitalize_copy(s);
    size_t match_length = 0;
    for (size_t i = ELEMENT_MAX - 1; i > 0; i--) {
        const auto dat = ELEMENTDATA_TABLE[i];
        const size_t N = dat.symbol.size();
        if (dat.symbol.compare(0, N, symbol, 0, N) == 0) {
            if (exact_match && dat.symbol != symbol)
                continue;
            if (dat.symbol.size() > match_length) {
                m_data = dat;
                match_length = dat.symbol.size();
            }
            if (symbol.size() == dat.symbol.size())
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

double Element::polarizability() const {
    static const std::array<double, 110> Thakkar_atomic_polarizability{
	4.50,   1.38,   164.04, 37.74,  20.43,  11.67,  7.26,   5.24,   3.70,
	2.66,   162.88, 71.22,  57.79,  37.17,  24.93,  19.37,  14.57,  11.09,
	291.10, 157.90, 142.30, 114.30, 97.30,  94.70,  75.50,  63.90,  57.70,
	51.10,  45.50,  38.35,  52.91,  40.80,  29.80,  26.24,  21.13,  16.80,
	316.20, 199.00, 153.00, 121.00, 106.00, 86.00,  77.00,  65.00,  58.00,
	32.00,  52.46,  47.55,  68.67,  57.30,  42.20,  38.10,  32.98,  27.06,
	396.00, 273.50, 210.00, 200.00, 190.00, 212.00, 203.00, 194.00, 187.00,
	159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30, 148.00, 109.00,
	88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73,  71.72,
	60.05,  48.60,  43.62,  40.73,  33.18,  315.20, 246.20, 217.00, 217.00,
	171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00,
	161.00, 123.00, 118.00, 0.00,   0.00,   0.00,   0.00,   0.00,   0.00,
	0.00,   0.00};
    return Thakkar_atomic_polarizability[m_data.atomic_number - 1];
}



} // namespace occ::core
