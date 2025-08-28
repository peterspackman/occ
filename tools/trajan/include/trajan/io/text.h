#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <string>
#include <vector>

class TextFileWriter {
private:
  std::ofstream file_stream;
  std::string delimiter;
  bool is_open;

public:
  inline TextFileWriter(const std::string &delimiter = ",")
      : delimiter(delimiter), is_open(false) {}
  inline ~TextFileWriter() {
    if (!is_open) {
      return;
    }
    file_stream.close();
  }

  inline bool open(const std::string &filename) {
    file_stream.open(filename);
    is_open = file_stream.is_open();
    return is_open;
  }

  inline void close() {
    if (!is_open) {
      return;
    }
    file_stream.close();
    is_open = false;
  }

  inline bool write_row(const std::vector<std::string> &data) {
    if (!is_open) {
      return false;
    }

    for (size_t i = 0; i < data.size(); ++i) {
      file_stream << data[i];
      if (i < data.size() - 1) {
        file_stream << delimiter;
      }
    }
    file_stream << std::endl;

    return file_stream.good();
  }

  template <typename... Args>
  inline bool write_formatted_row(const std::string &format_str,
                                  Args &&...args) {
    if (!is_open) {
      return false;
    }

    std::string formatted_row =
        fmt::format(format_str, std::forward<Args>(args)...);
    file_stream << formatted_row << std::endl;

    return file_stream.good();
  }

  template <typename... Args>
  inline bool write_line(const std::string &format_str, Args &&...args) {
    if (!is_open) {
      return false;
    }

    file_stream << fmt::format(fmt::runtime(format_str),
                               std::forward<Args>(args)...)
                << std::endl;
    return file_stream.good();
  }

  template <typename Collection>
  inline bool
  write_delimited_collection(const Collection &items,
                             const std::string &item_format_str = "{}") {
    if (!is_open) {
      return false;
    }

    bool first = true;
    for (const auto &item : items) {
      if (!first) {
        file_stream << delimiter;
      }
      file_stream << fmt::format(item_format_str, item);
      first = false;
    }
    file_stream << std::endl;

    return file_stream.good();
  }

  inline bool is_file_open() const { return is_open; }
};
