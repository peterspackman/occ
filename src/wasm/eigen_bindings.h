#pragma once
#include <Eigen/Dense>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <string>

namespace emscripten {

// Base TypeConverter declaration
template <typename T> struct TypeConverter;

namespace internal {

template <typename T> struct typed_array_name {};

template <> struct typed_array_name<double> {
  static constexpr const char *name = "Float64Array";
};

template <> struct typed_array_name<float> {
  static constexpr const char *name = "Float32Array";
};

template <> struct typed_array_name<int> {
  static constexpr const char *name = "Int32Array";
};

} // namespace internal

// Specialization for Eigen::Matrix
template <typename T, int Rows, int Cols>
struct TypeConverter<Eigen::Matrix<T, Rows, Cols>> {
  using MatrixType = Eigen::Matrix<T, Rows, Cols>;

  static val to_val(const MatrixType &mat) {
    const bool is_vector = (Cols == 1) || (Rows == 1);
    const int rows = mat.rows();
    const int cols = mat.cols();

    if (is_vector) {
      val array = val::array();
      for (int i = 0; i < mat.size(); ++i) {
        array.set(i, mat.data()[i]);
      }
      return array;
    }

    val array = val::array();
    for (int i = 0; i < rows; ++i) {
      val row = val::array();
      for (int j = 0; j < cols; ++j) {
        row.set(j, mat(i, j));
      }
      array.set(i, row);
    }
    return array;
  }

  static MatrixType from_val(const val &v) {
    const bool is_array = v.instanceof(val::global("Array"));

    if (!is_array) {
      throw std::runtime_error("Expected array input for matrix conversion");
    }

    const bool is_nested = v[0].instanceof(val::global("Array"));

    if (is_nested) {
      // Matrix case
      const int rows = v["length"].as<int>();
      const int cols = v[0]["length"].as<int>();
      MatrixType result;
      if (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
        result.resize(rows, cols);
      }
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          result(i, j) = v[i][j].as<T>();
        }
      }
      return result;
    } else {
      // Vector case
      const int size = v["length"].as<int>();
      MatrixType result;
      if (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic) {
        if (Cols == 1)
          result.resize(size, 1);
        else
          result.resize(1, size);
      }
      for (int i = 0; i < size; ++i) {
        result.data()[i] = v[i].as<T>();
      }
      return result;
    }
  }
};

// Specialization for Vector3
template <typename T> struct TypeConverter<Eigen::Matrix<T, 3, 1>> {
  using Vector3Type = Eigen::Matrix<T, 3, 1>;

  static val to_val(const Vector3Type &vec) {
    val array = val::array();
    array.set(0, vec.x());
    array.set(1, vec.y());
    array.set(2, vec.z());
    return array;
  }

  static Vector3Type from_val(const val &v) {
    if (v.instanceof(val::global("Array"))) {
      return Vector3Type(v[0].as<T>(), v[1].as<T>(), v[2].as<T>());
    }
    throw std::runtime_error("Invalid input type for Vector3 conversion");
  }
};

// Register common Eigen types with value semantics
template <typename T, int Rows, int Cols>
void register_matrix(const char *name) {
  typedef Eigen::Matrix<T, Rows, Cols> MatType;
  class_<MatType>(name)
      .constructor<>()
      .function("rows", &MatType::rows)
      .function("cols", &MatType::cols)
      .function("resize",
                optional_override([](MatType &self, int rows, int cols) {
                  self.resize(rows, cols);
                }))
      .function("data", optional_override([](const MatType &self) {
                  return TypeConverter<MatType>::to_val(self);
                }))
      .function("data", optional_override([](MatType &self, const val &data) {
                  self = TypeConverter<MatType>::from_val(data);
                }));
}

} // namespace emscripten
