#pragma once
#include <tonto/core/linear_algebra.h>

namespace tonto::linalg {

tonto::Mat3 kabsch_rotation_matrix(const tonto::Mat3N&, const tonto::Mat3N&);

}
