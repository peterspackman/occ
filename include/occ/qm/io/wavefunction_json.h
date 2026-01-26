#pragma once
#include <nlohmann/json.hpp>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

void from_json(const nlohmann::json &J, Energy &energy);
void to_json(nlohmann::json &J, const Energy &energy);

} // namespace occ::qm

namespace occ::io {

enum class JsonFormat {
  JSON,
  UBJSON,
  CBOR,
  BSON,
  MSGPACK,
};

inline JsonFormat json_format(const std::string &str) {
  if (str == ".json" || str == "json") {
    return JsonFormat::JSON;
  } else if (str == ".ubjson" || str == "ubjson") {
    return JsonFormat::UBJSON;
  } else if (str == ".cbor" || str == "cbor") {
    return JsonFormat::CBOR;
  } else if (str == ".bson" || str == "bson") {
    return JsonFormat::BSON;
  } else if (str == ".msgpack" || str == "msgpack") {
    return JsonFormat::MSGPACK;
  }
  return JsonFormat::JSON;
}

inline bool valid_json_format_string(const std::string &str) {
  return (str == ".json" || str == "json" || str == ".ubjson" ||
          str == "ubjson" || str == ".cbor" || str == "cbor" ||
          str == ".bson" || str == "bson" || str == ".msgpack" ||
          str == "msgpack");
}

struct JsonWavefunctionReader {
public:
  JsonWavefunctionReader(const std::string &,
                         JsonFormat fmt = JsonFormat::JSON);
  JsonWavefunctionReader(std::istream &);
  qm::Wavefunction wavefunction() const;
  void set_filename(const std::string &);

private:
  JsonFormat m_format{JsonFormat::JSON};
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

  inline void set_format(const std::string &ext) {
    m_format = json_format(ext);
  }

  inline void set_format(JsonFormat format) { m_format = format; }

  void set_shiftwidth(int);

private:
  JsonFormat m_format{JsonFormat::JSON};
  int m_shiftwidth{2};
};

} // namespace occ::io
