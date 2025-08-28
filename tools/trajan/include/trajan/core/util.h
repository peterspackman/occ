#pragma once
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cctype>
#include <filesystem>
#include <occ/crystal/unitcell.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <trajan/core/linear_algebra.h>
#include <vector>

namespace trajan::util {

namespace fs = std::filesystem;

using occ::crystal::UnitCell;

// static inline void capitalize(std::string &s) {
//   s[0] = std::toupper(s[0]);
//   std::transform(s.begin() + 1, s.end(), s.begin() + 1,
//                  [](unsigned char c) { return std::tolower(c); });
// }
//
// static inline std::string capitalize_copy(std::string s) {
//   capitalize(s);
//   return s;
// }
//
// static inline void to_upper(std::string &s) {
//   std::transform(s.begin(), s.end(), s.begin(),
//                  [](unsigned char c) { return std::toupper(c); });
// }
//
// static inline std::string to_upper_copy(std::string s) {
//   to_upper(s);
//   return s;
// }
//
// static inline void to_lower(std::string &s) {
//   std::transform(s.begin(), s.end(), s.begin(),
//                  [](unsigned char c) { return std::tolower(c); });
// }
//
// static inline std::string to_lower_copy(std::string s) {
//   to_lower(s);
//   return s;
// }
//
// // trim from start (in place)
// static inline void ltrim(std::string &s) {
//   s.erase(s.begin(), std::find_if(s.begin(), s.end(),
//                                   [](int ch) { return !std::isspace(ch); }));
// }
//
// // trim from end (in place)
// static inline void rtrim(std::string &s) {
//   s.erase(std::find_if(s.rbegin(), s.rend(),
//                        [](int ch) { return !std::isspace(ch); })
//               .base(),
//           s.end());
// }
//
// // trim from both ends (in place)
// static inline void trim(std::string &s) {
//   ltrim(s);
//   rtrim(s);
// }
//
// // trim from start (copying)
// static inline std::string ltrim_copy(std::string s) {
//   ltrim(s);
//   return s;
// }
//
// // trim from end (copying)
// static inline std::string rtrim_copy(std::string s) {
//   rtrim(s);
//   return s;
// }
//
// // trim from both ends (copying)
// static inline std::string trim_copy(std::string s) {
//   trim(s);
//   return s;
// }
//
// static inline std::vector<std::string> split_string(const std::string &input,
//                                                     char delimiter) {
//   std::vector<std::string> tokens;
//   std::stringstream ss(input);
//   std::string token;
//
//   while (std::getline(ss, token, delimiter)) {
//     token.erase(0, token.find_first_not_of(" \t"));
//     token.erase(token.find_last_not_of(" \t") + 1);
//     if (!token.empty()) {
//       tokens.push_back(token);
//     }
//   }
//   return tokens;
// }
//
// template <typename T>
// constexpr bool is_close(T a, T b,
//                         const T rtol =
//                         Eigen::NumTraits<T>::dummy_precision(), const T atol
//                         = Eigen::NumTraits<T>::epsilon()) {
//   return abs(a - b) <= (atol + rtol * abs(b));
// }

static inline double square_distance(Vec3 &p1, Vec3 &p2) {
  double dx = p1.x() - p2.x(), dy = p1.y() - p2.y(), dz = p1.z() - p2.z();
  return dx * dx + dy * dy + dz * dz;
}

template <typename T>
static inline const std::vector<T>
combine_vectors(const std::vector<std::vector<T>> &vecs) {
  std::vector<T> combined_vec;
  for (const std::vector<T> &vec : vecs) {
    combined_vec.reserve(combined_vec.size() + vec.size());
    combined_vec.insert(combined_vec.end(), vec.begin(), vec.end());
  }
  return combined_vec;
}

template <typename T>
static inline std::vector<T> deduplicate(const std::vector<T> &input) {
  std::vector<T> result;
  ankerl::unordered_dense::set<T> seen;

  result.reserve(input.size());
  for (const auto &element : input) {
    if (seen.insert(element).second) {
      result.push_back(element);
    }
  }

  return result;
}

template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
static inline std::vector<T> deduplicate_variant(const std::vector<T> &input,
                                                 Hash hasher = Hash(),
                                                 Equal equal = Equal()) {
  std::vector<T> result;
  ankerl::unordered_dense::set<T, Hash, Equal> seen(0, hasher, equal);

  result.reserve(input.size());

  for (const auto &element : input) {
    if (seen.insert(element).second) {
      result.push_back(element);
    }
  }

  return result;
}

template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
static inline std::pair<std::vector<T>, std::vector<std::bitset<8>>>
combine_map(const std::vector<std::vector<T>> &vecs, Hash hasher = Hash(),
            Equal equal = Equal()) {
  std::vector<T> result;
  size_t total_size = 0;
  for (const auto &vec : vecs) {
    total_size += vec.size();
  }
  result.reserve(total_size);

  std::vector<std::bitset<8>> presence_tracker;
  presence_tracker.reserve(total_size);

  for (size_t vec_idx = 0; vec_idx < vecs.size(); ++vec_idx) {
    for (const auto &element : vecs[vec_idx]) {
      result.push_back(element);
      std::bitset<8> presence;
      presence.set(vec_idx);
      presence_tracker.push_back(presence);
    }
  }

  return {result, presence_tracker};
}

template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
static inline std::tuple<std::vector<T>, std::vector<size_t>,
                         ankerl::unordered_dense::map<size_t, size_t>>
combine_map_check(const std::vector<std::vector<T>> &vecs, Hash hasher = Hash(),
                  Equal equal = Equal()) {
  std::vector<T> result;
  size_t total_size = 0;
  for (const auto &vec : vecs) {
    total_size += vec.size();
  }
  result.reserve(total_size);
  std::vector<size_t> presence_tracker;
  presence_tracker.reserve(total_size);
  ankerl::unordered_dense::map<T, std::vector<size_t>, Hash, Equal>
      entity_to_indices;
  ankerl::unordered_dense::map<size_t, size_t> index_to_canonical;

  for (size_t vec_idx = 0; vec_idx < vecs.size(); ++vec_idx) {
    for (const auto &element : vecs[vec_idx]) {
      size_t current_index = result.size();
      result.push_back(element);
      presence_tracker.push_back(vec_idx);

      auto it = entity_to_indices.find(element);
      if (it == entity_to_indices.end()) {
        entity_to_indices[element].push_back(current_index);
        index_to_canonical[current_index] = current_index;
      } else {
        size_t canonical_idx = it->second[0];
        entity_to_indices[element].push_back(current_index);
        index_to_canonical[current_index] = canonical_idx;
      }
    }
  }

  return {result, presence_tracker, index_to_canonical};
}

static inline bool
is_cross_section(const std::vector<size_t> &indices,
                 const std::vector<size_t> &presence_tracker) {
  ankerl::unordered_dense::set<size_t> seen_vectors;
  for (size_t idx : indices) {
    size_t original_vector = presence_tracker[idx];
    if (seen_vectors.find(original_vector) != seen_vectors.end()) {
      return false;
    }
    seen_vectors.insert(original_vector);
  }
  return true;
}

template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
static inline std::pair<std::vector<T>, std::vector<std::bitset<8>>>
combine_deduplicate_map(const std::vector<std::vector<T>> &vecs,
                        Hash hasher = Hash(), Equal equal = Equal()) {

  std::vector<T> result;
  size_t total_size = 0;
  for (const auto &vec : vecs) {
    total_size += vec.size();
  }
  result.reserve(total_size);
  ankerl::unordered_dense::map<T, size_t, Hash, Equal> element_to_index;
  std::vector<std::bitset<8>> presence_tracker;
  for (size_t vec_idx = 0; vec_idx < vecs.size(); ++vec_idx) {
    for (const auto &element : vecs[vec_idx]) {
      auto it = element_to_index.find(element);
      if (it == element_to_index.end()) {
        result.push_back(element);
        std::bitset<8> presence;
        presence.set(vec_idx);
        presence_tracker.push_back(presence);
        element_to_index[element] = result.size() - 1;

      } else {
        presence_tracker[it->second].set(vec_idx);
      }
    }
  }
  return {result, presence_tracker};
}

template <size_t NumIndices>
inline bool cross_section_fold(size_t num_sources, const std::bitset<8> *bts) {
  std::bitset<8> combined;

  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((combined |= bts[I]), ...);
  }(std::make_index_sequence<NumIndices>{});

  for (size_t i = 0; i < num_sources; ++i) {
    if (!combined.test(i))
      return false;
  }
  return true;
}

struct Opts {
  size_t num_threads;
  std::vector<fs::path> infiles;
};

inline std::pair<Mat3N, Mat3N> wrap_coordinates(Mat3N &cart_pos, UnitCell &uc,
                                                bool dummy_unit_cell = false) {
  if (dummy_unit_cell) {
    Vec3 min_vals = cart_pos.rowwise().minCoeff();
    Mat3N shifted_cart_pos = cart_pos.colwise() - min_vals;
    Mat3N frac_pos = uc.to_fractional(shifted_cart_pos);
    return {frac_pos, cart_pos};
  }
  Mat3N frac_pos = uc.to_fractional(cart_pos);
  frac_pos = frac_pos.array() - frac_pos.array().floor();
  Mat3N wrapped_cart_pos = uc.to_cartesian(frac_pos);
  return {frac_pos, wrapped_cart_pos};
}

// class DummyUnitCell : public UnitCell {
//
// };

} // namespace trajan::util
