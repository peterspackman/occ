#pragma once

#include <libxtc/xtc_reader.h>
#include <libxtc/xtc_writer.h>
#include <memory>
#include <trajan/io/file_handler.h>

namespace trajan::io {

class XTCHandler : public FileHandler {
public:
  inline FileType file_type() const override { return FileType::XTC; }

protected:
  bool _initialise() override;
  void _finalise() override;
  bool read_next_frame(core::Frame &frame) override;
  bool write_next_frame(const core::Frame &frame) override;

private:
  std::unique_ptr<XTCReader> m_xtcreader;
  std::unique_ptr<XTCWriter> m_xtcwriter;
};

} // namespace trajan::io
