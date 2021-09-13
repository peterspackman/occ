#pragma once
#include <fstream>
#include <vector>
#include <occ/crystal/crystal.h>

namespace occ::crystal {
class Crystal;
class CrystalDimers;
} // namespace occ::crystal

namespace occ::io::crystalgrower {
class StructureWriter {
  public:
    StructureWriter(const std::string &filename);
    StructureWriter(std::ostream &);

    void write(const occ::crystal::Crystal &,
               const occ::crystal::CrystalDimers &);
    void write_net(const occ::crystal::Crystal &,
                   const occ::crystal::CrystalDimers &);

  private:
    std::ofstream m_owned_destination;
    std::ostream &m_dest;
};

class NetWriter {
  public:
    NetWriter(const std::string &filename);
    NetWriter(std::ostream &);

    void write(const occ::crystal::Crystal &,
               const occ::crystal::CrystalDimers &);

  private:
    std::ofstream m_owned_destination;
    std::ostream &m_dest;
};
} // namespace occ::io::crystalgrower