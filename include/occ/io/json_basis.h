#include <istream>
#include <occ/3rdparty/parallel_hashmap/phmap.h>
#include <occ/core/element.h>
#include <vector>

namespace occ::io {

using occ::core::Element;

struct ElectronShell {
    std::string function_type;
    std::string region;
    std::vector<int> angular_momentum;
    std::vector<double> exponents;
    std::vector<std::vector<double>> coefficients;
};

struct ReferenceData {
    std::string description;
    std::vector<std::string> keys;
};
struct ElementBasis {
    std::vector<ElectronShell> electron_shells;
    std::vector<ReferenceData> references;
};

using ElementMap = phmap::flat_hash_map<int, ElementBasis>;

struct JsonBasis {
    std::string version;
    std::string name;
    std::string description;
    ElementMap elements;
};

struct JsonBasisReader {
  public:
    JsonBasisReader(const std::string &);
    JsonBasisReader(std::istream &);
    const ElementMap &element_map() const;
    const ElementBasis &element_basis(int number);
    const ElementBasis &element_basis(const Element &element);
    JsonBasis json_basis;

  private:
    void parse(std::istream &);
    std::string m_filename;
};

} // namespace occ::io