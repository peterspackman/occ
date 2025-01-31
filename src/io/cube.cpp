#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/io/cube.h>
#include <scn/scan.h>

namespace occ::io {

Cube::Cube() : basis(Mat3::Identity()), m_grid(11, 11, 11) {}

void Cube::write_header_to_stream(std::ostream &out) {
  // Write the header
  fmt::print(out, "{}\n{}\n", name, description);
  fmt::print(out, "{:5d} {:12.6f} {:12.6f} {:12.6f}\n", atoms.size(),
             origin.x(), origin.y(), origin.z());
  for (int i = 0; i < 3; i++) {
    fmt::print(out, "{:5d} {:12.6f} {:12.6f} {:12.6f}\n", steps(i), basis(0, i),
               basis(1, i), basis(2, i));
  }

  auto gc = [&](int i) { return charges.size() > i ? charges[i] : 0.0; };
  // Write atoms
  int idx = 0;
  for (const auto &atom : atoms) {
    fmt::print(out, "{:5d} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n",
               atom.atomic_number, gc(idx), atom.x, atom.y, atom.z);
    idx++;
  }
}

void Cube::save(const std::string &destination) {
  std::filebuf buf;
  buf.open(destination, std::ios::out);
  std::ostream out(&buf);
  if (out.fail())
    throw std::runtime_error("Could not open file for writing: " + destination);
  write_header_to_stream(out);
  write_data_to_stream(out);
}

void Cube::save(std::ostream &out) {
  write_header_to_stream(out);
  write_data_to_stream(out);
}

void Cube::write_data_to_stream(std::ostream &out) {
  if (m_grid.size() < 1)
    throw std::runtime_error("No data in cube");

  for (int x = 0, i = 0; x < steps(0); x++) {
    for (int y = 0; y < steps(1); y++) {
      for (int z = 0; z < steps(2); z++, i++) {
        fmt::print(out, "{:12.6f} ", m_grid(x, y, z));
        if (z % 6 == 5) {
          fmt::print(out, "\n");
        }
      }
      fmt::print(out, "\n");
    }
  }
}

void Cube::center_molecule() {
  Vec3 center = 0.5 * basis * steps.cast<double>();

  Vec3 centroid = Vec3::Zero();
  for (const auto &atom : atoms) {
    centroid += Vec3(atom.x, atom.y, atom.z);
  }
  centroid /= atoms.size();

  origin = centroid - center;
}

template <typename T>
std::unique_ptr<T[]> read_block(std::istream &stream, size_t count) {
  auto data = std::make_unique<T[]>(count);
  size_t idx = 0;
  std::string line;
  while (idx < count) {
    std::getline(stream, line);
    auto input = scn::ranges::subrange{line};
    while (auto result = scn::scan<T>(input, "{}")) {
      data[idx++] = result->value();
      input = result->range();
      if (idx >= count)
        break;
    }
  }
  return data;
}

Cube Cube::load(std::istream &input) {
  Cube cube;
  std::string line;

  // Read title lines
  std::getline(input, cube.name);
  std::getline(input, cube.description);

  std::getline(input, line);
  auto header =
      scn::scan<int, double, double, double>(line, "    {}    {}    {}    {}");
  if (!header)
    throw std::runtime_error("Failed to parse cube header");

  auto [natoms, x, y, z] = header->values();
  cube.origin = Vec3(x, y, z);

  for (int i = 0; i < 3; i++) {
    std::getline(input, line);
    auto result = scn::scan<int, double, double, double>(
        line, "    {}    {}    {}    {}");
    if (!result)
      throw std::runtime_error("Failed to parse grid dimensions");

    auto &[n, x, y, z] = result->values();
    cube.steps(i) = n;
    cube.basis(0, i) = x;
    cube.basis(1, i) = y;
    cube.basis(2, i) = z;
  }

  cube.atoms.reserve(natoms);
  cube.charges = Vec::Zero(natoms);
  for (int i = 0; i < natoms; i++) {
    std::getline(input, line);
    auto atom_result = scn::scan<int, double, double, double, double>(
        line, "    {}    {}    {}    {}    {}");
    if (!atom_result)
      throw std::runtime_error("Failed to parse atom data");

    core::Atom atom;
    auto [n, q, x, y, z] = atom_result->values();
    atom.atomic_number = n;
    cube.charges(i) = q;
    atom.x = x;
    atom.y = y;
    atom.z = z;
    cube.atoms.push_back(atom);
  }

  const size_t total_points = cube.steps(0) * cube.steps(1) * cube.steps(2);

  auto data = read_block<float>(input, total_points);

  std::array<size_t, 3> dims{static_cast<size_t>(cube.steps(0)),
                             static_cast<size_t>(cube.steps(1)),
                             static_cast<size_t>(cube.steps(2))};
  cube.m_grid = geometry::VolumeGrid(std::move(data), dims);

  return cube;
}

Cube Cube::load(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  return load(file);
}

} // namespace occ::io
