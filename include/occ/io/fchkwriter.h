#pragma once
#include <ankerl/unordered_dense.h>
#include <array>
#include <fstream>
#include <istream>
#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <variant>
#include <vector>

namespace occ::io {

namespace impl {
using fchk_vector = std::variant<std::vector<double>, std::vector<int>,
                                 std::vector<bool>, std::vector<std::string>>;

using fchk_scalar = std::variant<double, int, bool, std::string>;

struct FchkScalarWriter {
  std::ostream &destination;
  std::string key;
  void operator()(int);
  void operator()(double);
  void operator()(const std::string &);
  void operator()(bool);
};

struct FchkVectorWriter {
  std::ostream &destination;
  std::string key;
  void operator()(const std::vector<int> &);
  void operator()(const std::vector<double> &);
  void operator()(const std::vector<std::string> &);
  void operator()(const std::vector<bool> &);
};

} // namespace impl

class FchkWriter {
public:
  enum FchkType {
    SP,
    FOPT,
    POPT,
    FTS,
    PTS,
    FSADDLE,
    PSADDLE,
    FORCE,
    FREQ,
    SCAN,
    GUESS_ONLY,
    LST,
    STABILITY,
    REARCHIVE_MS_RESTART,
    MIXED
  };

  FchkWriter(const std::string &filename);
  FchkWriter(std::ostream &);

  void set_title(const std::string &title) { m_title = title; }
  void set_fchk_type(FchkType t) { m_type = t; }
  void set_method(const std::string &method) { m_method = method; }
  void set_basis_name(const std::string &basis) { m_basis_name = basis; }

  void set_basis(const occ::qm::AOBasis &);

  template <typename T>
  void set_scalar(const std::string &key, const T &value) {
    if constexpr (std::is_integral<T>::value) {
      if constexpr (std::is_same<decltype(value), bool>::value) {
        m_scalars[key] = value;
      } else {
        m_scalars[key] = static_cast<int>(value);
      }
    } else if constexpr (std::is_floating_point<T>::value) {
      m_scalars[key] = static_cast<double>(value);
    } else {
      m_scalars[key] = std::string(value);
    }
  }

  template <typename T>
  void set_vector(const std::string &key, const Eigen::DenseBase<T> &mat) {
    if constexpr (std::is_integral<typename T::Scalar>::value) {
      if constexpr (std::is_same<typename T::Scalar, bool>::value) {
        std::vector<bool> vals;
        vals.reserve(mat.size());
        for (size_t c = 0; c < mat.cols(); c++) {
          for (size_t r = 0; r < mat.rows(); r++) {
            vals.push_back(mat(r, c));
          }
        }
        m_vectors[key] = vals;
      } else {
        std::vector<int> vals;
        vals.reserve(mat.size());
        for (size_t c = 0; c < mat.cols(); c++) {
          for (size_t r = 0; r < mat.rows(); r++) {
            vals.push_back(static_cast<int>(mat(r, c)));
          }
        }
        m_vectors[key] = vals;
      }
    } else if constexpr (std::is_floating_point<typename T::Scalar>::value) {
      std::vector<double> vals;
      vals.reserve(mat.size());
      for (size_t c = 0; c < mat.cols(); c++) {
        for (size_t r = 0; r < mat.rows(); r++) {
          vals.push_back(static_cast<double>(mat(r, c)));
        }
      }
      m_vectors[key] = vals;
    }
  }

  template <typename InputType, typename CastType = InputType>
  void set_vector(const std::string &key, const std::vector<InputType> &vec) {
    if constexpr (std::is_integral<CastType>::value) {
      if constexpr (std::is_same<CastType, bool>::value) {
        std::vector<bool> vals;
        vals.reserve(vec.size());
        for (const auto &x : vec)
          vals.push_back(x);
        m_vectors[key] = vals;
      } else {
        std::vector<int> vals;
        vals.reserve(vec.size());
        for (const auto &x : vec)
          vals.push_back(static_cast<int>(x));
        m_vectors[key] = vals;
      }
    } else if constexpr (std::is_floating_point<CastType>::value) {
      std::vector<double> vals;
      vals.reserve(vec.size());
      for (const auto &x : vec)
        vals.push_back(static_cast<double>(x));
      m_vectors[key] = vals;
    } else {
      std::vector<std::string> vals;
      for (const auto &x : vec)
        vals.push_back(std::string(x));
      m_vectors[key] = vals;
    }
  }

  void write();

private:
  std::string m_title{"fchk produced by OCC"};
  std::string m_method{"unknown_method"};
  std::string m_basis_name{"unknown_basis"};
  FchkType m_type{SP};
  ankerl::unordered_dense::map<std::string, impl::fchk_scalar> m_scalars;
  ankerl::unordered_dense::map<std::string, impl::fchk_vector> m_vectors;
  std::ofstream m_owned_destination;
  std::ostream &m_dest;
};

} // namespace occ::io
