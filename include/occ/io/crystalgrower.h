#pragma once
#include <ankerl/unordered_dense.h>
#include <fstream>
#include <occ/crystal/crystal.h>

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
  using InteractionLabels =
      ankerl::unordered_dense::map<std::string, std::string>;
  NetWriter(const std::string &filename);
  NetWriter(std::ostream &);

  void write(const occ::crystal::Crystal &,
             const occ::crystal::CrystalDimers &);

  inline const InteractionLabels &interaction_labels() const {
    return m_interaction_labels;
  };

private:
  std::ofstream m_owned_destination;
  std::ostream &m_dest;
  InteractionLabels m_interaction_labels;
};
} // namespace occ::io::crystalgrower
