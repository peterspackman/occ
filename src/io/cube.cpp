#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/io/cube.h>

namespace occ::io {

Cube::Cube() : basis(Mat3::Identity()) {}

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

void Cube::write_data_to_file(const std::string &destination) {
  std::filebuf buf;
  buf.open(destination, std::ios::out);
  std::ostream out(&buf);
  if (out.fail())
    throw std::runtime_error("Could not open file for writing: " + destination);
  write_header_to_stream(out);
  write_data_to_stream(out);
}

void Cube::write_data_to_stream(std::ostream &out) {
  if (data.size() < 1)
    throw std::runtime_error("No data in cube");
  for (int x = 0, i = 0; x < steps(0); x++) {
    for (int y = 0; y < steps(1); y++) {
      for (int z = 0; z < steps(2); z++, i++) {
        fmt::print(out, "{:12.6f} ", data(i));
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

} // namespace occ::io
