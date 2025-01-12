#pragma once
#include <fstream>
#include <occ/crystal/crystal.h>

namespace occ::io::kmcpp {

class InputWriter {
public:
  InputWriter(const std::string &filename);
  InputWriter(std::ostream &);

  void write(const occ::crystal::Crystal &, const occ::crystal::CrystalDimers &,
             const std::vector<double> &);

private:
  std::ofstream m_owned_destination;
  std::ostream &m_dest;
};

} // namespace occ::io::kmcpp
