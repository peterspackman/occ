#pragma once
#include <nlohmann/json.hpp>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

void from_json(const nlohmann::json &J, Energy &energy);
void to_json(nlohmann::json &J, const Energy &energy);

} // namespace occ::qm

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
    enum Format {
	JSON,
	UBJSON,
	CBOR,
	BSON,
	MSGPACK,
    };
    JsonWavefunctionWriter();
    void write(const qm::Wavefunction &, const std::string &) const;
    void write(const qm::Wavefunction &, std::ostream &) const;
    std::string to_string(const qm::Wavefunction &) const;

    inline void set_format(Format format) {
	m_format = format;
    }

    void set_shiftwidth(int);

  private:
    Format m_format{JSON};
    int m_shiftwidth{2};
};

} // namespace occ::io
