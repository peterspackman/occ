#pragma once
#include <occ/isosurface/isosurface.h>
#include <nlohmann/json.hpp>
#include <string>

namespace occ::io {

// Convert isosurface to JSON
nlohmann::json isosurface_to_json(const isosurface::Isosurface &surf);

// Write isosurface to JSON file
void write_isosurface_json(const std::string &filename, 
                          const isosurface::Isosurface &surf);

// Write isosurface to JSON string
std::string isosurface_to_json_string(const isosurface::Isosurface &surf);

} // namespace occ::io