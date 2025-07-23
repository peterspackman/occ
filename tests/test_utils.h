#pragma once
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <fmt/core.h>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>

// Custom Catch2 matcher for matrix comparison with detailed debugging
template<typename ExpectedType>
class MatrixApproxMatcher : public Catch::Matchers::MatcherGenericBase {
    ExpectedType const& m_expected;
    double m_atol;
    double m_rtol;
    mutable std::string m_description;

public:
    MatrixApproxMatcher(ExpectedType const& expected, double atol = 1e-6, double rtol = 1e-6)
        : m_expected(expected), m_atol(atol), m_rtol(rtol) {}

    template<typename ActualType>
    bool match(ActualType const& actual) const {
        if (actual.rows() != m_expected.rows() || actual.cols() != m_expected.cols()) {
            m_description = fmt::format("Size mismatch: actual {}x{} vs expected {}x{}", 
                                      actual.rows(), actual.cols(), 
                                      m_expected.rows(), m_expected.cols());
            return false;
        }

        std::vector<std::tuple<int, int, double, double, double>> failures;
        
        for (int i = 0; i < actual.rows(); ++i) {
            for (int j = 0; j < actual.cols(); ++j) {
                double a = actual(i, j);
                double e = m_expected(i, j);
                double abs_diff = std::abs(a - e);
                double rel_diff = std::abs(e) > 0 ? abs_diff / std::abs(e) : abs_diff;
                
                if (abs_diff > m_atol && rel_diff > m_rtol) {
                    failures.emplace_back(i, j, a, e, abs_diff);
                }
            }
        }
        
        if (!failures.empty()) {
            std::ostringstream oss;
            oss << fmt::format("Matrix comparison failed (atol={:.2e}, rtol={:.2e}):\n", m_atol, m_rtol);
            
            if (failures.size() <= 10) {
                oss << fmt::format("  {} mismatched elements:\n", failures.size());
                for (const auto& [i, j, a, e, diff] : failures) {
                    oss << fmt::format("    [{:2d},{:2d}]: actual={:12.8f}, expected={:12.8f}, diff={:.2e}\n", 
                                     i, j, a, e, diff);
                }
            } else {
                oss << fmt::format("  First 10 of {} mismatched elements:\n", failures.size());
                for (size_t k = 0; k < 10; ++k) {
                    const auto& [i, j, a, e, diff] = failures[k];
                    oss << fmt::format("    [{:2d},{:2d}]: actual={:12.8f}, expected={:12.8f}, diff={:.2e}\n", 
                                     i, j, a, e, diff);
                }
                oss << fmt::format("    ... and {} more\n", failures.size() - 10);
            }
            
            // Show small matrices in full
            if (actual.rows() <= 8 && actual.cols() <= 8) {
                oss << "\n  Actual matrix:\n";
                for (int i = 0; i < actual.rows(); ++i) {
                    oss << "    ";
                    for (int j = 0; j < actual.cols(); ++j) {
                        oss << fmt::format("{:10.6f} ", actual(i, j));
                    }
                    oss << "\n";
                }
                
                oss << "\n  Expected matrix:\n";
                for (int i = 0; i < m_expected.rows(); ++i) {
                    oss << "    ";
                    for (int j = 0; j < m_expected.cols(); ++j) {
                        oss << fmt::format("{:10.6f} ", m_expected(i, j));
                    }
                    oss << "\n";
                }
            }
            
            m_description = oss.str();
            return false;
        }
        
        return true;
    }

    std::string describe() const override {
        if (m_description.empty()) {
            return fmt::format("is approximately equal to expected matrix (atol={:.2e}, rtol={:.2e})", m_atol, m_rtol);
        }
        return m_description;
    }
};

// Helper function to create the matcher
template<typename T>
inline MatrixApproxMatcher<T> IsApproxMatrix(T const& expected, double atol = 1e-6, double rtol = 1e-6) {
    return MatrixApproxMatcher<T>(expected, atol, rtol);
}