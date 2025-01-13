#include <occ/core/progress.h>
#include <occ/core/util.h>

#include <chrono>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <iostream>
#include <occ/core/log.h>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <windows.h>
inline COORD occ_saved_cursor_position;
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <sys/ioctl.h>
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

namespace occ::core {

inline bool stdout_is_tty() { return ISATTY(FILENO(stdout)); }

#undef ISATTY
#undef FILENO

TerminalSize TerminalSize::get_current_size() {

#if defined(_WIN32) || defined(_WIN64)
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(hStdOut, &csbi);
  return TerminalSize{csbi.dwSize.X, csbi.dwSize.Y};
#else
  struct winsize size;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
  return TerminalSize{size.ws_row, size.ws_col};
#endif
}

ProgressTracker::ProgressTracker(int total) : m_total(total) {
  m_is_tty = stdout_is_tty();
  m_tsize = TerminalSize::get_current_size();
  m_time_points.push_back(std::chrono::high_resolution_clock::now());
}

void ProgressTracker::save_cursor() {
#if defined(_WIN32) || defined(_WIN64)
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(hStdOut, &csbi);
  occ_saved_cursor_position = csbi.dwCursorPosition;
#else
  fmt::print("\033[s");
#endif
}

void ProgressTracker::restore_cursor() {
#if defined(_WIN32) || defined(_WIN64)
  SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE),
                           occ_saved_cursor_position);
#else
  fmt::print("\033[u"); // Restore cursor position
#endif
}

void ProgressTracker::move_cursor_to_bottom() {
#if defined(_WIN32) || defined(_WIN64)
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(hStdOut, &csbi);

  COORD bottomCoord;
  bottomCoord.X = 0;
  bottomCoord.Y = m_tsize.rows;
  SetConsoleCursorPosition(hStdOut, bottomCoord);
#else
  fmt::print("\033[{};1H", m_tsize.rows);
#endif
}

void ProgressTracker::update(int progress, int total,
                             const std::string description) {
  if (!m_started) {
    m_started = true;
    m_time_points.push_back(std::chrono::high_resolution_clock::now());
  }

  if (progress != m_current_progress) {
    m_time_points.push_back(std::chrono::high_resolution_clock::now());
    m_current_progress = progress;
  }

  estimate_time_remaining();

  std::string eta_string;
  if (m_time_points.size() > 1) {
    eta_string = occ::util::human_readable_time(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            m_estimated_time_remaining));
  }

  float percent = static_cast<float>(progress) / total;
  if (!m_is_tty) {
    occ::log::info("{: <40s} {}/{} {: >3d}% {}", description, progress, total,
                   static_cast<int>(percent * 100.0), eta_string);
    return;
  }

  std::string left =
      fmt::format("{: <40s} {}/{}  [", description, progress, total);
  std::string right = fmt::format(
      "| {: >3d}% {}", static_cast<int>(percent * 100.0), eta_string);

  const int bar_width = m_tsize.cols - left.size() - right.size();
  int pos = bar_width * percent;

  save_cursor();
  move_cursor_to_bottom();

  fmt::print("{}", left);
  for (int i = 0; i < bar_width; ++i) {
    if (i <= pos)
      fmt::print("#");
    else
      fmt::print(".");
  }
  fmt::print("{}\r", right);
  std::cout.flush();
  restore_cursor();
}

void ProgressTracker::clear_progress_line() {
  if (!m_is_tty)
    return;
  save_cursor();
  move_cursor_to_bottom();
  for (int i = 0; i < m_tsize.cols; ++i) {
    fmt::print(" ");
  }
  restore_cursor();
}

void ProgressTracker::clear() { clear_progress_line(); }

void ProgressTracker::set_tty(bool value) { m_is_tty = value; }

void ProgressTracker::estimate_time_remaining() {
  if (m_time_points.empty())
    return;

  auto now = std::chrono::high_resolution_clock::now();
  double current_t =
      std::chrono::duration<double>(now - m_time_points.front()).count();

  // Calculate progress volatility if we have enough data points
  if (m_time_points.size() >= 3) {
    Eigen::VectorXd velocities(m_time_points.size() - 1);
    for (size_t i = 1; i < m_time_points.size(); ++i) {
      auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    m_time_points[i] - m_time_points[i - 1])
                    .count();
      velocities[i - 1] = dt > 0 ? 1.0 / dt : 0.0;
    }

    double mean = velocities.mean();
    double variance = (velocities.array() - mean).square().mean();

    // Adjust noise parameters based on observed variance
    double process_noise = std::clamp(variance * 0.1, 0.01, 1.0);
    double measurement_noise = std::clamp(variance, 0.1, 10.0);

    m_time_estimator.adjust_noise(process_noise, measurement_noise);
  }

  m_time_estimator.update(m_current_progress, current_t);
  double remaining_seconds = m_time_estimator.estimate_remaining(m_total);
  m_estimated_time_remaining = std::chrono::duration<double>(remaining_seconds);

  // If you want to use the uncertainty:
  double uncertainty_seconds = m_time_estimator.time_uncertainty(m_total);
  // Convert to chrono duration only when needed for display
  auto uncertainty = std::chrono::duration<double>(uncertainty_seconds);
}

std::chrono::duration<double> ProgressTracker::time_taken() const {
  return m_time_points.back() - m_time_points.front();
}

} // namespace occ::core
