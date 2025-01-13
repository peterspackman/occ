#pragma once
#include <chrono>
#include <occ/core/kalman_estimator.h>
#include <string>
#include <vector>

namespace occ::core {

struct TerminalSize {
  int rows{0};
  int cols{0};
  static TerminalSize get_current_size();
};

class ProgressTracker {
public:
  ProgressTracker(int total);

  void set_tty(bool);
  void update(int progress, int total, const std::string description = "");
  void reset();

  void clear();

  std::chrono::duration<double> time_taken() const;

private:
  void estimate_time_remaining();
  void move_cursor_to_bottom();
  void save_cursor();
  void restore_cursor();
  void clear_progress_line();

  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  int m_total{0};
  int m_current{0};
  int m_window{5};
  int m_current_progress{0};
  bool m_started{false};
  std::vector<TimePoint> m_time_points;
  std::string m_name{"Progress"};
  std::chrono::duration<double> m_average_time;
  std::chrono::duration<double> m_estimated_time_remaining;
  std::chrono::duration<double> m_estimated_time_uncertainty;
  TerminalSize m_tsize;
  KalmanEstimator m_time_estimator;
  bool m_is_tty{false};
};

} // namespace occ::core
