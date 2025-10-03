#pragma once
#include <trajan/io/file_handler.h>

namespace trajan::io {

class DCDHandler : public FileHandler {
public:
  inline FileType file_type() const override { return FileType::DCD; }
  bool read_next_frame(core::Frame &frame) override;
  bool write_next_frame(const core::Frame &frame) override;

private:
  bool _initialise() override;
  void _finalise() override;
  bool _parse_dcd_header();
  bool parse_dcd_header();
  bool _parse_dcd(core::Frame &frame);
  bool parse_dcd(core::Frame &frame);
  bool write_dcd_header(const core::Frame &frame);
  bool write_unit_cell_data(const core::Frame &frame);
  bool write_dcd_frame(const core::Frame &frame);

  size_t m_current_frame = 0;
  size_t m_total_frames = 0;
  size_t m_num_atoms = 0;

  template <typename T> bool read_binary(T &value);
  template <typename T> bool write_binary(const T &value);
  bool read_fortran_record(std::vector<char> &buffer);
  bool write_fortran_record(const std::vector<char> &buffer);
  bool skip_fortran_record();

  int m_dcd_version = 0;
  bool m_has_extra_block = false;
  bool m_has_4d_coords = false;
  bool m_is_charmm_format = false;
  double m_timestep = 1.0;

  std::vector<float> m_x_coords;
  std::vector<float> m_y_coords;
  std::vector<float> m_z_coords;
};

} // namespace trajan::io
