#include <cmath>
#include <fmt/format.h>
#include <occ/dma/mult.h>

namespace occ::dma {

std::string Mult::to_string(int lm) const {
  std::string result;

  // Component labels
  constexpr const char *ql[] = {"Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                                "Q8", "Q9", "QA", "QB", "QC", "QD", "QE", "QF"};
  constexpr const char *qm[] = {"0 ", "1c", "1s", "2c", "2s", "3c", "3s", "4c",
                                "4s", "5c", "5s", "6c", "6s", "7c", "7s", "8c",
                                "8s", "9c", "9s", "Ac", "As", "Bc", "Bs", "Cc",
                                "Cs", "Dc", "Ds", "Ec", "Es", "Fc", "Fs"};

  result += fmt::format("                    Q00  = {:11.6f}\n", Q00());

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
      result += fmt::format("|{}| = {:11.3e}", ql[l], qs);

      for (size_t i = 0; i < significant_indices.size(); i++) {
        int idx = significant_indices[i];
        double val = q(k + idx);

        if (i % 3 == 0 && i > 0) {
          result += fmt::format("{}\n", std::string(17, ' ')); // New line with indentation
        }
        result += fmt::format("  {}{} = {:11.3e}", ql[l], qm[idx], val);
      }
      result += "\n";

    } else if (significant_indices.empty() && !big) {
      // Just magnitude for small values with no significant components
      result += fmt::format("|{}| = {:11.6f}\n", ql[l], qs);

    } else if (!significant_indices.empty() && !big) {
      // Fixed point notation for normal values
      result += fmt::format("|{}| = {:11.6f}", ql[l], qs);

      for (size_t i = 0; i < significant_indices.size(); i++) {
        int idx = significant_indices[i];
        double val = q(k + idx);

        if (i % 3 == 0 && i > 0) {
          result += "\n" + std::string(17, ' '); // New line with indentation
        }
        result += fmt::format("  {}{} = {:11.6f}", ql[l], qm[idx], val);
      }
      result += "\n";
    }

    k += ll1; // Move to next angular momentum level
  }

  return result;
}

} // namespace occ::dma
