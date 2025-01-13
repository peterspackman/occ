#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <occ/geometry/index_cache.h>
#include <vector>

namespace occ::geometry {

template <typename T> void set_zero(T &t) {
  for (size_t i = 0; i < t.size(); ++i)
    t[i] = 0;
}

IndexCache::IndexCache(size_t s)
    : size_x(s), size_y(s), layer0(size_x, size_y), layer1(size_x, size_y),
      row0(size_x), row1(size_x) {}

IndexCache::IndexCache(size_t x, size_t y)
    : size_x(x), size_y(y), layer0(size_x, size_y), layer1(size_x, size_y),
      row0(size_x), row1(size_x) {}

void IndexCache::put(size_t x, size_t y, size_t edge, size_t index) {
  if (edge >= 4 && edge <= 7)
    layer1(x, y).element[edge - 4] = index;

  switch (edge) {
  case 6:
    row1(x).element[0] = index;
    break;
  case 11:
    row1(x).element[1] = index;
    break;
  case 10:
    row1(x).element[2] = index;
    break;
  default:
    break;
  }
  switch (edge) {
  case 5:
  case 10:
    cell1[0] = index;
    break;
  default:
    break;
  }
  current_cell[edge] = index;
}

uint32_t IndexCache::get(size_t x, size_t y, size_t edge) {
  uint32_t result{0};
  switch (edge) {
  case 0:
  case 1:
  case 2:
  case 3:
    result = layer0(x, y).element[edge];
    break;
  case 4:
    result = row0(x).element[0];
    break;
  case 8:
    result = row0(x).element[1];
    break;
  case 9:
    result = row0(x).element[2];
    break;
  case 7:
  case 11:
    result = cell0[1];
    break;
  default:
    result = 0;
    break;
  }

  if (result > 0)
    return result;
  return current_cell[edge];
}

void IndexCache::advance_cell() {
  std::swap(cell0, cell1);
  set_zero(current_cell);
}

void IndexCache::advance_row() {
  std::swap(row0, row1);
  set_zero(cell0);
}

void IndexCache::advance_layer() {
  std::swap(layer0, layer1);
  set_zero(cell0);
  for (int i = 0; i < row0.rows(); i++) {
    set_zero(row0(i).element);
  }
}

} // namespace occ::geometry
