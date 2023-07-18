#pragma once
#include <occ/qm/wavefunction.h>

namespace occ::io {

struct JsonWavefunctionReader {
  public:
    JsonWavefunctionReader(const std::string &);
    JsonWavefunctionReader(std::istream &);
    qm::Wavefunction wavefunction() const;
    void set_filename(const std::string &);

  private:
    void parse(std::istream &);
    std::string m_filename{"_string_data_"};
    qm::Wavefunction m_wavefunction;
};

struct JsonWavefunctionWriter {
  public:
    JsonWavefunctionWriter();
    void write(const qm::Wavefunction &, const std::string &) const;
    void write(const qm::Wavefunction &, std::ostream &) const;
    std::string to_string(const qm::Wavefunction &) const;

    void set_shiftwidth(int);

  private:
    int m_shiftwidth{2};
};

} // namespace occ::io
