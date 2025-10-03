#pragma once

#include <trajan/io/file_handler.h>

namespace trajan::io {

class XYZHandler : public FileHandler {
public:
  inline FileType file_type() const override { return FileType::XYZ; }

protected:
  bool _initialise() override;
  void _finalise() override;
  bool read_next_frame(core::Frame &frame) override;
  bool write_next_frame(const core::Frame &frame) override;
};

} // namespace trajan::io
