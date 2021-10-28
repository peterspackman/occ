#pragma once
#include <istream>
#include <occ/core/molecule.h>

namespace occ::io {

class QCSchemaReader {
public:
    QCSchemaReader(const std::string &);
    QCSchemaReader(std::istream &);
private:
    void parse(std::istream &);
    std::string m_filename;
};

} // namespace occ::io
