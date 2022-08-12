#pragma once
#include <fstream>
#include <occ/crystal/crystal.h>
#include <vector>

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
