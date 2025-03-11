#include <cstdlib>
#include <occ/core/data_directory.h>

namespace occ {

namespace {
static std::string data_directory_override{""};
}

void set_data_directory(const std::string &s) { data_directory_override = s; }

const char *get_data_directory() {
  return data_directory_override.empty() ? std::getenv("OCC_DATA_PATH")
                                         : data_directory_override.c_str();
}

} // namespace occ
