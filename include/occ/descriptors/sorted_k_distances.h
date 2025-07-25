/**
 * \brief High-performance helper class to maintain k smallest distances in
 * sorted order
 *
 * Template parameter K allows compile-time optimizations and stack allocation
 * Automatically uses NEON on ARM or AVX2 on x86
 *
 * Optimizations:
 * - No dynamic allocation - everything on stack
 * - SIMD operations for comparisons and shifts (NEON or AVX2)
 * - Batch insertion support for processing multiple values at once
 * - Binary search for larger k values
 * - Compile-time loop unrolling
 * - Branchless operations where possible
 */

// Platform detection and SIMD includes
#if !defined(OCC_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#define HAS_NEON 1
#define HAS_AVX2 0
#elif !defined(OCC_DISABLE_SIMD) && (defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)))
#include <immintrin.h>
#define HAS_NEON 0
#define HAS_AVX2 1
#else
#define HAS_NEON 0
#define HAS_AVX2 0
#endif

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace occ::descriptors {

template <int K> class SortedKDistances {
  static_assert(K > 0, "K must be positive");

private:
  alignas(32) std::array<double, K> m_data;
  int m_size = 0;

  // Threshold for switching from linear to binary search
  static constexpr int BINARY_SEARCH_THRESHOLD = 32;

  // Find insertion position using linear search (for small k)
  template <bool UseUnroll = (K >= 4)>
  inline int find_position_linear(double distance) const {
    if constexpr (UseUnroll) {
      int pos = 0;
      // Unroll loop for better performance
      while (pos + 3 < m_size) {
        if (m_data[pos] >= distance)
          return pos;
        if (m_data[pos + 1] >= distance)
          return pos + 1;
        if (m_data[pos + 2] >= distance)
          return pos + 2;
        if (m_data[pos + 3] >= distance)
          return pos + 3;
        pos += 4;
      }
      while (pos < m_size && m_data[pos] < distance) {
        pos++;
      }
      return pos;
    } else {
      // Simple linear search for very small K
      int pos = 0;
      while (pos < m_size && m_data[pos] < distance) {
        pos++;
      }
      return pos;
    }
  }

  // Find insertion position using binary search (for large k)
  inline int find_position_binary(double distance) const {
    int left = 0;
    int right = m_size;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (m_data[mid] < distance) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }

  // Optimized shift using memmove
  inline void shift_right(int from, int to) {
    if (from < to) {
      std::memmove(&m_data[from + 1], &m_data[from],
                   (to - from) * sizeof(double));
    }
  }

public:
  SortedKDistances() {
    // Initialize with infinity for easier comparisons
    m_data.fill(std::numeric_limits<double>::infinity());
  }

  void clear() {
    m_size = 0;
    // Reset to infinity for consistent behavior
    m_data.fill(std::numeric_limits<double>::infinity());
  }

  // Optimized single insertion
  inline bool try_insert(double distance) {
    // Early exit if distance is too large
    if (m_size == K && distance >= m_data[K - 1]) {
      return false;
    }

    int pos;
    if constexpr (K < BINARY_SEARCH_THRESHOLD) {
      pos = find_position_linear(distance);
    } else {
      pos = (m_size < BINARY_SEARCH_THRESHOLD) ? find_position_linear(distance)
                                               : find_position_binary(distance);
    }

    if (m_size < K) {
      // Not full yet
      shift_right(pos, m_size);
      m_data[pos] = distance;
      m_size++;
      return true;
    } else if (pos < K) {
      // Replace worst distance
      shift_right(pos, K - 1);
      m_data[pos] = distance;
      return true;
    }
    return false;
  }

#if HAS_NEON
  // NEON version for ARM - processes 2 doubles at a time (128-bit registers)
  inline int try_insert_batch4(const double *distances) {
    // Load 4 distances using 2 NEON registers
    float64x2_t batch_low = vld1q_f64(distances);
    float64x2_t batch_high = vld1q_f64(distances + 2);

    // If we're not full, insert all that fit
    if (m_size + 4 <= K) {
      // Simple case: just insert all 4 in sorted order
      alignas(16) double temp[4];
      vst1q_f64(temp, batch_low);
      vst1q_f64(temp + 2, batch_high);
      int count = 0;
      for (int i = 0; i < 4; ++i) {
        if (try_insert(temp[i]))
          count++;
      }
      return count;
    }

    // Complex case: need to check each value
    double worst =
        (m_size == K) ? m_data[K - 1] : std::numeric_limits<double>::infinity();
    float64x2_t worst_vec = vdupq_n_f64(worst);

    // Compare with worst value
    uint64x2_t mask_low = vcltq_f64(batch_low, worst_vec);
    uint64x2_t mask_high = vcltq_f64(batch_high, worst_vec);

    // Extract and insert values that passed the test
    alignas(16) double temp[4];
    vst1q_f64(temp, batch_low);
    vst1q_f64(temp + 2, batch_high);

    // Check each value based on mask
    int count = 0;
    if (vgetq_lane_u64(mask_low, 0) && try_insert(temp[0]))
      count++;
    if (vgetq_lane_u64(mask_low, 1) && try_insert(temp[1]))
      count++;
    if (vgetq_lane_u64(mask_high, 0) && try_insert(temp[2]))
      count++;
    if (vgetq_lane_u64(mask_high, 1) && try_insert(temp[3]))
      count++;
    return count;
  }

  // NEON optimized batch for 2 values (more natural for NEON)
  inline int try_insert_batch2(const double *distances) {
    float64x2_t batch = vld1q_f64(distances);

    if (m_size + 2 <= K) {
      alignas(16) double temp[2];
      vst1q_f64(temp, batch);
      int count = 0;
      if (try_insert(temp[0]))
        count++;
      if (try_insert(temp[1]))
        count++;
      return count;
    }

    double worst =
        (m_size == K) ? m_data[K - 1] : std::numeric_limits<double>::infinity();
    float64x2_t worst_vec = vdupq_n_f64(worst);
    uint64x2_t mask = vcltq_f64(batch, worst_vec);

    alignas(16) double temp[2];
    vst1q_f64(temp, batch);

    int count = 0;
    if (vgetq_lane_u64(mask, 0) && try_insert(temp[0]))
      count++;
    if (vgetq_lane_u64(mask, 1) && try_insert(temp[1]))
      count++;
    return count;
  }

#elif HAS_AVX2
  // AVX2 version for x86 - processes 4 doubles at a time (256-bit registers)
  inline int try_insert_batch4(const double *distances) {
    // Load 4 distances
    __m256d batch = _mm256_loadu_pd(distances);

    // If we're not full, insert all that fit
    if (m_size + 4 <= K) {
      // Simple case: just insert all 4 in sorted order
      alignas(32) double temp[4];
      _mm256_store_pd(temp, batch);
      int count = 0;
      for (int i = 0; i < 4; ++i) {
        if (try_insert(temp[i]))
          count++;
      }
      return count;
    }

    // Complex case: need to check each value
    double worst =
        (m_size == K) ? m_data[K - 1] : std::numeric_limits<double>::infinity();
    __m256d worst_vec = _mm256_set1_pd(worst);
    __m256d mask = _mm256_cmp_pd(batch, worst_vec, _CMP_LT_OQ);

    // Extract mask to check which values to insert
    int mask_int = _mm256_movemask_pd(mask);

    // Insert values that passed the test
    alignas(32) double temp[4];
    _mm256_store_pd(temp, batch);
    int count = 0;
    for (int i = 0; i < 4; ++i) {
      if ((mask_int & (1 << i)) && try_insert(temp[i])) {
        count++;
      }
    }
    return count;
  }

  // For compatibility with NEON version
  inline int try_insert_batch2(const double *distances) {
    __m128d batch = _mm_loadu_pd(distances);

    if (m_size + 2 <= K) {
      alignas(16) double temp[2];
      _mm_store_pd(temp, batch);
      int count = 0;
      if (try_insert(temp[0]))
        count++;
      if (try_insert(temp[1]))
        count++;
      return count;
    }

    double worst =
        (m_size == K) ? m_data[K - 1] : std::numeric_limits<double>::infinity();
    __m128d worst_vec = _mm_set1_pd(worst);
    __m128d mask = _mm_cmplt_pd(batch, worst_vec);

    alignas(16) double temp[2];
    _mm_store_pd(temp, batch);
    int mask_int = _mm_movemask_pd(mask);

    int count = 0;
    if ((mask_int & 1) && try_insert(temp[0]))
      count++;
    if ((mask_int & 2) && try_insert(temp[1]))
      count++;
    return count;
  }

#else
  // Fallback scalar version
  inline int try_insert_batch4(const double *distances) {
    int count = 0;
    for (int i = 0; i < 4; ++i) {
      if (m_size < K || distances[i] < m_data[K - 1]) {
        if (try_insert(distances[i]))
          count++;
      }
    }
    return count;
  }

  inline int try_insert_batch2(const double *distances) {
    int count = 0;
    for (int i = 0; i < 2; ++i) {
      if (m_size < K || distances[i] < m_data[K - 1]) {
        if (try_insert(distances[i]))
          count++;
      }
    }
    return count;
  }
#endif

  // Platform-independent sorting network batch insertion
  inline int try_insert_sorted_batch4(double d0, double d1, double d2,
                                      double d3) {
    // Sort the 4 values using a sorting network (optimal for 4 elements)
    if (d0 > d1)
      std::swap(d0, d1);
    if (d2 > d3)
      std::swap(d2, d3);
    if (d0 > d2)
      std::swap(d0, d2);
    if (d1 > d3)
      std::swap(d1, d3);
    if (d1 > d2)
      std::swap(d1, d2);

    // Now d0 <= d1 <= d2 <= d3
    // Insert in order (early exit when possible)
    int count = 0;
    if (m_size == K && d0 >= m_data[K - 1])
      return 0;
    if (try_insert(d0))
      count++;
    if (m_size == K && d1 >= m_data[K - 1])
      return count;
    if (try_insert(d1))
      count++;
    if (m_size == K && d2 >= m_data[K - 1])
      return count;
    if (try_insert(d2))
      count++;
    if (m_size == K && d3 >= m_data[K - 1])
      return count;
    if (try_insert(d3))
      count++;
    return count;
  }

  // Specialized version for very small K (compile-time optimization)
  template <int N = K>
  inline typename std::enable_if<N <= 4, bool>::type
  try_insert_specialized(double distance) {
    if constexpr (K == 1) {
      if (m_size == 0 || distance < m_data[0]) {
        m_data[0] = distance;
        m_size = 1;
        return true;
      }
      return false;
    } else if constexpr (K == 2) {
      if (m_size == 0) {
        m_data[0] = distance;
        m_size = 1;
        return true;
      } else if (m_size == 1) {
        if (distance < m_data[0]) {
          m_data[1] = m_data[0];
          m_data[0] = distance;
        } else {
          m_data[1] = distance;
        }
        m_size = 2;
        return true;
      } else if (distance < m_data[1]) {
        if (distance < m_data[0]) {
          m_data[1] = m_data[0];
          m_data[0] = distance;
        } else {
          m_data[1] = distance;
        }
        return true;
      }
      return false;
    } else {
      // Use regular insert for K = 3 or 4
      return try_insert(distance);
    }
  }

  // Getters
  constexpr size_t size() const { return static_cast<size_t>(m_size); }
  constexpr bool empty() const { return m_size == 0; }
  double back() const {
    return m_size > 0 ? m_data[m_size - 1]
                      : std::numeric_limits<double>::infinity();
  }
  double operator[](size_t idx) const { return m_data[idx]; }

  // For compatibility with Eigen
  Eigen::Map<const Eigen::ArrayXd> as_array(int k) const {
    return Eigen::Map<const Eigen::ArrayXd>(m_data.data(), std::min(k, m_size));
  }

  // Direct access to raw data for maximum performance
  const double *raw_data() const { return m_data.data(); }

  // Check if a distance would be inserted (without actually inserting)
  inline bool would_insert(double distance) const {
    return m_size < K || distance < m_data[m_size - 1];
  }

  // Get the worst (largest) distance that would still be inserted
  inline double worst_acceptable() const {
    return (m_size == K) ? m_data[K - 1]
                         : std::numeric_limits<double>::infinity();
  }

  // Reserve-like functionality (no-op for stack-based storage, but kept for API
  // compatibility)
  void reserve(size_t) const {}

  // Query which SIMD implementation is being used
  static constexpr const char *simd_type() {
#if HAS_NEON
    return "NEON";
#elif HAS_AVX2
    return "AVX2";
#else
    return "scalar";
#endif
  }
};

// Convenience aliases for common sizes
using SortedK1Distance = SortedKDistances<1>;
using SortedK5Distances = SortedKDistances<5>;
using SortedK10Distances = SortedKDistances<10>;
using SortedK20Distances = SortedKDistances<20>;
using SortedK50Distances = SortedKDistances<50>;
using SortedK100Distances = SortedKDistances<100>;
} // namespace occ::descriptors
