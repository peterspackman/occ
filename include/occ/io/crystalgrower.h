#pragma once
#include <fstream>
#include <occ/crystal/crystal.h>

namespace occ::crystal {
class Crystal;
}

namespace occ::io::crystalgrower {
class StructureWriter {
  public:
    StructureWriter(const std::string &filename);
    StructureWriter(std::ostream &);

    void write(const occ::crystal::Crystal &);

  private:
    std::ofstream m_owned_destination;
    std::ostream &m_dest;
};
} // namespace occ::io::crystalgrower
