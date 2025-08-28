#pragma once
#include <string_view>
#include <trajan/io/file_handler.h>

namespace trajan::io {

constexpr std::string_view PDB_CRYST_FMT_READ =
    "%6c%9lf%9lf%9lf%7lf%7lf%7lf%11c%4c";
constexpr std::string_view PDB_LINE_FMT_READ =
    // AAAAAABBBBBXCCCCDEEEXFGGGGHXXXIIIIIIIIJJJJJJJJKKKKKKKKLLLLLLMMMMMMXXXXXXXXXXNNOO
    "%6c%5d%1c%4c%1c%3c%1c%1c%4d%1c%3c%8lf%8lf%8lf%6f%6f%10c%2c%2c";

constexpr std::string_view PDB_LINE_FMT_WRITE =
    "{:<6}{:>5}{:1}{:>4}{:1}{:>3}{:1}{:1}{:>4}{:1}{:>3}{:8.3f}{:8.3f}{:8.3f}{:"
    "6.2f}{:6.2f}{:10}{:>2}{:>2}";

class PDBHandler : public FileHandler {
public:
  inline FileType file_type() const override { return FileType::PDB; }
  bool parse_pdb(core::Frame &frame);
  bool read_next_frame(core::Frame &frame) override;
  bool write_next_frame(const core::Frame &frame) override;

private:
  bool _initialise() override;
  void _finalise() override;
  bool m_has_read{false};
};

} // namespace trajan::io
