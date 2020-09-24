#include "symmetryoperation.h"
#include "fraction.h"
#include "util.h"
#include <regex>

namespace craso::crystal {

using craso::numeric::Fraction;
using craso::util::tokenize;

void decode_int(int code, Mat4 &seitz) {
  /*Decode an integer encoded symmetry operation.

  A space group operation is compressed using ternary numerical system for
  rotation and duodecimal system for translation. This is achieved because
  each element of rotation matrix can have only one of {-1,0,1}, and the
  translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
  3^9 * 12^3 = 34012224 different values can map space group operations. In
  principle, octal numerical system can be used for translation, but
  duodecimal system is more convenient.
  */
  seitz.block(3, 0, 1, 3).setZero();
  seitz(3, 3) = 1.0;
  int r = code % 19683; // 19683 = 3**9
  int shift = 6561;     // 6561 = 3**8
  int t = code / 19683;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // we need integer division here
      seitz(i, j) = (r % (shift * 3)) / shift - 1;
      shift /= 3;
    }
  }

  shift = 144;
  for (int i = 0; i < 3; i++) {
    // we need integer division here by shift
    seitz(i, 3) = ((t % (shift * 12)) / shift) / 12.0;
    shift /= 12;
  }
}

std::vector<std::string> get_symbols(std::string s) {
  std::vector<std::string> symbols;
  std::regex re(".*?([+-]*[xyzXYZ0-9/\\.]+)");
  auto matches_begin = std::sregex_iterator(s.begin(), s.end(), re);
  auto matches_end = std::sregex_iterator();
  for (std::sregex_iterator it = matches_begin; it != matches_end; ++it) {
    std::smatch match = *it;
    symbols.push_back(match.str());
  }
  return symbols;
}

void decode_string(std::string code, Mat4 &seitz) {
  seitz.fill(0.0);
  auto tokens = tokenize(code, ",");
  int i = 0;
  int idx = 0;
  for (const auto &x : tokens) {
    int fac = 1;
    auto symbols = get_symbols(x);
    for (const auto &symbol : symbols) {
      fac = (symbol.find('-') != std::string::npos) ? -1 : 1;
      if (symbol.find('x') != std::string::npos) {
        idx = 0;
        seitz(i, idx) = fac;
      } else if (symbol.find('y') != std::string::npos) {
        idx = 1;
        seitz(i, idx) = fac;
      } else if (symbol.find('z') != std::string::npos) {
        idx = 2;
        seitz(i, idx) = fac;
      } else {
        seitz(i, 3) = fmod(Fraction(symbol).cast<double>(), 1.0);
      }
    }
    i++;
  }
  seitz.block(3, 0, 1, 3).setZero();
  seitz(3, 3) = 1.0;
}

std::string encode_string(const Mat4 &seitz) {
  /* Encode a rotation matrix (of -1, 0, 1s) and (rational) translation vector
  into string form e.g. 1/2-x,z-1/3,-y-1/6
  */
  using craso::util::join;
  std::string symbols = "xyz";
  std::vector<std::string> res;
  for (int i = 0; i < 3; i++) {
    auto t = Fraction(seitz(i, 3)).limit_denominator(12);
    std::string v = "";
    if (!(t == 0)) {
      v += t.to_string();
    }
    for (int j = 0; j < 3; j++) {
      auto c = seitz(i, j);
      if (c < 0) {
        v += "-" + symbols.substr(j, 1);
      } else if (c > 0) {
        v += "+" + symbols.substr(j, 1);
      }
    }
    res.push_back(v);
  }
  return join(res, ",");
}

int encode_int(const Mat4 &seitz) {
  /* Encode an integer encoded symmetry from a rotation matrix and translation
  vector.

  A space group operation is compressed using ternary numerical system for
  rotation and duodecimal system for translation. This is achieved because
  each element of rotation matrix can have only one of {-1,0,1}, and the
  translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
  3^9 * 12^3 = 34012224 different values can map space group operations. In
  principle, octal numerical system can be used for translation, but
  duodecimal system is more convenient.
  */

  int r = 0, t = 0, shift = 1;
  // encode rotation component
  for (int i = 2; i >= 0; i--) {
    for (int j = 2; j >= 0; j--) {
      r += (static_cast<int>(seitz.coeff(i, j)) + 1) * shift;
      shift *= 3;
    }
  }
  shift = 1;
  for (int i = 2; i >= 0; i--) {
    t += static_cast<int>(seitz.coeff(i, 3) * 12) * shift;
    shift *= 12;
  }
  return r + t * 19683;
}

SymmetryOperation::SymmetryOperation(int code) { set_from_int(code); }

SymmetryOperation::SymmetryOperation(const std::string &str) {
  set_from_string(str);
}

SymmetryOperation SymmetryOperation::translated(const Vec3 &t) const {
  SymmetryOperation result = *this;
  result.m_seitz.block(0, 3, 3, 1) += t;
  result.update_from_seitz();
  result.m_str = encode_string(result.m_seitz);
  result.m_int = encode_int(result.m_seitz);
  return result;
}

SymmetryOperation SymmetryOperation::inverted() const {
  SymmetryOperation result = *this;
  result.m_seitz.block(0, 0, 3, 4) *= -1;
  result.update_from_seitz();
  result.m_str = encode_string(result.m_seitz);
  result.m_int = encode_int(result.m_seitz);
  return result;
}

void SymmetryOperation::set_from_int(int code) {
  if (code == m_int)
    return;
  m_int = code;
  decode_int(code, m_seitz);
  update_from_seitz();
  m_str = encode_string(m_seitz);
}

void SymmetryOperation::set_from_string(const std::string &code) {
  if (code == m_str)
    return;
  decode_string(code, m_seitz);
  m_str = code;
  update_from_seitz();
  m_int = encode_int(m_seitz);
}

void SymmetryOperation::update_from_seitz() {
  m_rotation = m_seitz.block(0, 0, m_rotation.rows(), m_rotation.cols());
  m_translation =
      m_seitz.block(0, 3, m_translation.rows(), m_translation.cols());
}

} // namespace craso::crystal
