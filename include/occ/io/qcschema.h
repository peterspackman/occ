#pragma once
#include <istream>
#include <occ/core/element.h>
#include <array>

namespace occ::io {

using occ::chem::Element;

struct OccInput;

struct QCSchemaBond {
    int idx_a{-1};
    int idx_b{-1};
    double bond_length{0.0};
};

struct QCSchemaMolecule {
    std::vector<std::array<double, 3>> positions;
    std::vector<Element> elements;
};

struct QCSchemaModel {
    std::string method;
    std::string basis;
};

struct QCSchemaInput {
    QCSchemaMolecule molecule;
    QCSchemaModel model;
    std::string driver;
};

class QCSchemaReader {
public:
    QCSchemaReader(const std::string &);
    QCSchemaReader(std::istream &);
    OccInput as_occ_input() const;
    void update_occ_input(OccInput &) const;
    QCSchemaInput input;
private:
    void parse(std::istream &);
    std::string m_filename;
};

} // namespace occ::io
