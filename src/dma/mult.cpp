#include <cmath>
#include <fmt/format.h>
#include <occ/dma/mult.h>

namespace occ::dma {

Mult::Mult(int n) : max_rank(n) { q = Vec::Zero(121); }

Mult::Mult() : max_rank(10) { q = Vec::Zero(121); }

std::string Mult::to_string(int lm) const {
  std::string result;

  // Component labels
  constexpr const char *ql[] = {"Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                                "Q8", "Q9", "QA", "QB", "QC", "QD", "QE", "QF"};
  constexpr const char *qm[] = {"0 ", "1c", "1s", "2c", "2s", "3c", "3s", "4c",
                                "4s", "5c", "5s", "6c", "6s", "7c", "7s", "8c",
                                "8s", "9c", "9s", "Ac", "As", "Bc", "Bs", "Cc",
                                "Cs", "Dc", "Ds", "Ec", "Es", "Fc", "Fs"};

  result += fmt::format("                   Q00  ={:11.6f}\n", Q00());

  int k = 1;

  for (int l = 1; l <= lm; l++) {
    int ll1 = 2 * l + 1; // Number of components for this l

    // Find significant components and calculate magnitude
    std::vector<int> significant_indices;
    double qsq = 0.0;

    for (int i = 0; i < ll1; i++) {
      double val = q(k + i); // Convert to 0-based indexing
      qsq += val * val;
      if (std::abs(val) >= 5e-7) {
        significant_indices.push_back(i);
      }
    }

    double qs = std::sqrt(qsq);
    bool big = (qs >= 1e3);

    // Format output based on magnitude and number of significant components
    if (!significant_indices.empty() && big) {
      // Scientific notation for large values
      result += fmt::format("|{}| ={:11.3e}", ql[l], qs);

      for (size_t i = 0; i < significant_indices.size(); i++) {
        int idx = significant_indices[i];
        double val = q(k + idx);

        if (i % 3 == 0 && i > 0) {
          result += fmt::format(
              "{}\n", std::string(17, ' ')); // New line with indentation
        }
        result += fmt::format("  {}{} ={:11.3e}", ql[l], qm[idx], val);
      }
      result += "\n";

    } else if (significant_indices.empty() && !big) {
      // Just magnitude for small values with no significant components
      result += fmt::format("|{}| ={:11.6f}\n", ql[l], qs);

    } else if (!significant_indices.empty() && !big) {
      // Fixed point notation for normal values
      result += fmt::format("|{}| ={:11.6f}", ql[l], qs);

      for (size_t i = 0; i < significant_indices.size(); i++) {
        int idx = significant_indices[i];
        double val = q(k + idx);

        if (i % 3 == 0 && i > 0) {
          result += "\n" + std::string(17, ' '); // New line with indentation
        }
        result += fmt::format("  {}{} ={:11.6f}", ql[l], qm[idx], val);
      }
      result += "\n";
    }

    k += ll1; // Move to next angular momentum level
  }

  return result;
}

double Mult::get_multipole(int l, int m) const {
  // Calculate the index in the q array for a given (l, m) pair
  // For rank l, components are ordered as: Q_l0, Q_l1c, Q_l1s, Q_l2c, Q_l2s, ..., Q_llc, Q_lls
  // Starting index for rank l = l^2
  if (l < 0 || l > max_rank || m < -l || m > l) return 0.0;
  
  int base_index = l * l;  // Starting index for this rank
  int component_index;
  
  if (m == 0) {
    component_index = 0;  // Q_l0 is first
  } else if (m > 0) {
    component_index = 2 * m - 1;  // Q_lmc (cosine components)
  } else {
    component_index = 2 * (-m);   // Q_lms (sine components)
  }
  
  int final_index = base_index + component_index;
  if (final_index >= q.size()) return 0.0;
  
  return q(final_index);
}

double& Mult::get_multipole(int l, int m) {
  // Same logic as const version, but return reference
  if (l < 0 || l > max_rank || m < -l || m > l) {
    static double dummy = 0.0;
    return dummy;
  }
  
  int base_index = l * l;
  int component_index;
  
  if (m == 0) {
    component_index = 0;
  } else if (m > 0) {
    component_index = 2 * m - 1;
  } else {
    component_index = 2 * (-m);
  }
  
  int final_index = base_index + component_index;
  if (final_index >= q.size()) {
    static double dummy = 0.0;
    return dummy;
  }
  
  return q(final_index);
}

std::pair<int, int> Mult::component_name_to_lm(const std::string& name) {
  // Handle special case
  if (name == "charge") return {0, 0};
  
  // Parse component name like "Q21c", "Q30", "Q11s"
  if (name.length() < 3 || name[0] != 'Q') return {-1, 0};  // Invalid
  
  // Extract rank (can be 1 or 2 characters)
  int rank;
  std::string m_part;
  
  if (name.length() == 3) {
    // Single digit rank: Q10, Q20, etc.
    rank = name[1] - '0';
    m_part = name.substr(2);
  } else if (name.length() == 4) {
    // Could be single digit rank with 2-char m part (Q11c, Q22s) or hex rank
    if (name[1] >= '0' && name[1] <= '9' && (name[3] == 'c' || name[3] == 's')) {
      // Single digit rank with 2-char m part: Q11c, Q22s, etc.
      rank = name[1] - '0';
      m_part = name.substr(2);
    } else if (name[1] >= 'A' && name[1] <= 'F') {
      // Hex digit (A=10, B=11, etc.)
      rank = 10 + (name[1] - 'A');
      m_part = name.substr(2);
    } else {
      return {-1, 0};  // Invalid
    }
  } else if (name.length() == 5) {
    // Two digit rank: Q10c, Q44s, etc.
    if (name[1] >= '0' && name[1] <= '9' && name[2] >= '0' && name[2] <= '9') {
      rank = (name[1] - '0') * 10 + (name[2] - '0');
      m_part = name.substr(3);
    } else {
      return {-1, 0};  // Invalid
    }
  } else {
    return {-1, 0};  // Invalid
  }
  
  // Convert m_part to m value
  if (m_part == "0") {
    return {rank, 0};
  } else if (m_part.length() == 2 && m_part[1] == 'c') {
    // Cosine component (positive m)
    int m = m_part[0] - '0';
    if (m_part[0] >= 'A') m = 10 + (m_part[0] - 'A');
    return {rank, m};
  } else if (m_part.length() == 2 && m_part[1] == 's') {
    // Sine component (negative m)
    int m = m_part[0] - '0';
    if (m_part[0] >= 'A') m = 10 + (m_part[0] - 'A');
    return {rank, -m};
  }
  
  return {-1, 0};  // Invalid
}

double Mult::get_component(const std::string& name) const {
  auto [l, m] = component_name_to_lm(name);
  if (l < 0) return 0.0;
  return get_multipole(l, m);
}

double& Mult::get_component(const std::string& name) {
  auto [l, m] = component_name_to_lm(name);
  if (l < 0) {
    static double dummy = 0.0;
    return dummy;
  }
  return get_multipole(l, m);
}

} // namespace occ::dma
