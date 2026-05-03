#include "d3_data.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <stdexcept>

namespace occ::disp::d3_data {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string locate_refdata() {
  const char *base = occ::get_data_directory();
  if (base) {
    fs::path p = fs::path(base) / "dftd3" / "refdata.json";
    if (fs::exists(p)) return p.string();
  }
  if (fs::exists("dftd3/refdata.json")) return "dftd3/refdata.json";
  if (fs::exists("refdata.json")) return "refdata.json";
  throw std::runtime_error(
      "Cannot locate DFT-D3 reference data file (looked at "
      "share/dftd3/refdata.json, dftd3/refdata.json, refdata.json). "
      "Set OCC_DATA_PATH or run from a directory containing dftd3/refdata.json.");
}

ReferenceData load_from_json(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open D3 reference data: " + path);
  }
  json j;
  in >> j;

  ReferenceData out;
  const int n_elem = j.value("max_elements", 94);
  const int n_ref = j.value("max_references", 5);
  if (n_elem != N_ELEMENTS || n_ref != MAX_REF) {
    throw std::runtime_error("D3 refdata: unexpected dimensions (max_elements=" +
                             std::to_string(n_elem) + ", max_references=" +
                             std::to_string(n_ref) + ")");
  }

  // number_of_references[ia-1] is the count for element Z=ia.
  const auto &nor = j.at("number_of_references");
  for (int Z = 1; Z <= N_ELEMENTS; ++Z) {
    out.nref[Z] = nor[Z - 1];
  }

  // reference_cn[Z-1][iref] is the reference covalent CN.
  const auto &rcn = j.at("reference_cn");
  for (int Z = 1; Z <= N_ELEMENTS; ++Z) {
    const auto &row = rcn[Z - 1];
    for (int k = 0; k < MAX_REF; ++k) out.ref_cn[Z][k] = row[k];
  }

  // c6ab is a flat list of pair blocks; pair_index uses xtb's 1-indexed
  // lower-triangle linearization but the JSON is 0-indexed (drop dummy
  // pair 0 — see extract_d3_data.py).
  const auto &c6ab = j.at("c6ab");
  const std::size_t expected = static_cast<std::size_t>(N_ELEMENTS) *
                                (N_ELEMENTS + 1) / 2;
  if (c6ab.size() != expected) {
    throw std::runtime_error("D3 refdata: c6ab has " +
                             std::to_string(c6ab.size()) + " pairs (expected " +
                             std::to_string(expected) + ")");
  }
  // Allocate with index 0 unused so pair_index() (1-based) maps directly.
  out.c6ab.resize(expected + 1);
  for (std::size_t p = 0; p < expected; ++p) {
    const auto &block = c6ab[p];
    for (int i = 0; i < MAX_REF; ++i)
      for (int k = 0; k < MAX_REF; ++k)
        out.c6ab[p + 1][i][k] = block[i][k];
  }
  return out;
}

} // namespace

const ReferenceData &reference_data() {
  static const ReferenceData data = [] {
    const std::string path = locate_refdata();
    occ::log::debug("Loading DFT-D3 reference data from {}", path);
    return load_from_json(path);
  }();
  return data;
}

} // namespace occ::disp::d3_data
