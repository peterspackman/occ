#include "d4_data.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <stdexcept>

namespace occ::disp::d4_data {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string locate_refdata() {
  // Search order:
  //   1. ${OCC_DATA_PATH}/dftd4/refdata.json
  //   2. ./dftd4/refdata.json
  //   3. ./refdata.json
  // Mirrors the convention used by Gfn2Parameters::load_default.
  const char *base = occ::get_data_directory();
  if (base) {
    fs::path p = fs::path(base) / "dftd4" / "refdata.json";
    if (fs::exists(p)) return p.string();
  }
  if (fs::exists("dftd4/refdata.json")) return "dftd4/refdata.json";
  if (fs::exists("refdata.json")) return "refdata.json";
  throw std::runtime_error(
      "Cannot locate DFT-D4 reference data file (looked at "
      "share/dftd4/refdata.json, dftd4/refdata.json, refdata.json). "
      "Set OCC_DATA_PATH or run from a directory containing dftd4/refdata.json.");
}

ReferenceData load_from_json(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open D4 reference data: " + path);
  }
  json j;
  in >> j;

  ReferenceData out;
  // Casimir-Polder weights.
  const auto &cpw = j.at("casimir_polder_weights");
  if (cpw.size() != static_cast<std::size_t>(N_FREQ)) {
    throw std::runtime_error("D4 refdata: casimir_polder_weights must have " +
                             std::to_string(N_FREQ) + " entries");
  }
  for (int k = 0; k < N_FREQ; ++k) out.casimir_polder_weights[k] = cpw[k];

  // Secondary atoms (entries 1..17 in xtb; we store at indices 1..17).
  const auto &sec = j.at("secondary");
  const auto &secq_gfn2 = sec.at("secq_gfn2");
  const auto &secq_dft = sec.at("secq_dft");
  const auto &sscale = sec.at("sscale");
  const auto &secaiw = sec.at("secaiw");
  if (secq_gfn2.size() != static_cast<std::size_t>(N_SECONDARY) ||
      secq_dft.size() != static_cast<std::size_t>(N_SECONDARY) ||
      sscale.size() != static_cast<std::size_t>(N_SECONDARY) ||
      secaiw.size() != static_cast<std::size_t>(N_SECONDARY)) {
    throw std::runtime_error("D4 refdata: secondary tables must have " +
                             std::to_string(N_SECONDARY) + " entries");
  }
  for (int i = 0; i < N_SECONDARY; ++i) {
    out.secq_gfn2[i + 1] = secq_gfn2[i];
    out.secq_dft[i + 1] = secq_dft[i];
    out.sscale[i + 1] = sscale[i];
    const auto &aiw = secaiw[i];
    if (aiw.size() != static_cast<std::size_t>(N_FREQ)) {
      throw std::runtime_error("D4 refdata: secaiw row " + std::to_string(i) +
                               " has " + std::to_string(aiw.size()) +
                               " entries (expected 23)");
    }
    for (int k = 0; k < N_FREQ; ++k) out.secaiw[i + 1][k] = aiw[k];
  }

  // Per-element data.
  const auto &elements = j.at("elements");
  for (int Z = 1; Z <= N_ELEMENTS; ++Z) {
    const std::string key = std::to_string(Z);
    if (!elements.contains(key)) continue; // leave defaults
    const auto &e = elements.at(key);
    out.zeff[Z] = e.at("zeff");
    out.gam[Z] = e.at("gam");
    out.sqrt_zr4r2[Z] = e.at("sqrt_zr4r2");
    auto &ref = out.elements[Z];
    ref.refn = e.at("refn");
    if (ref.refn < 0 || ref.refn > MAX_REF) {
      throw std::runtime_error("D4 refdata: element Z=" + std::to_string(Z) +
                               " has refn=" + std::to_string(ref.refn) +
                               " (max " + std::to_string(MAX_REF) + ")");
    }
    auto fill_int = [&](const char *name, std::array<int, MAX_REF> &dst) {
      const auto &arr = e.at(name);
      for (std::size_t i = 0; i < arr.size() && i < MAX_REF; ++i)
        dst[i] = arr[i];
    };
    auto fill_dbl = [&](const char *name, std::array<double, MAX_REF> &dst) {
      const auto &arr = e.at(name);
      for (std::size_t i = 0; i < arr.size() && i < MAX_REF; ++i)
        dst[i] = arr[i];
    };
    fill_int("refsys", ref.refsys);
    fill_dbl("refcovcn", ref.refcovcn);
    fill_dbl("refq_gfn2", ref.refq_gfn2);
    fill_dbl("refh_gfn2", ref.refh_gfn2);
    fill_dbl("refq_dft", ref.refq_dft);
    fill_dbl("refh_dft", ref.refh_dft);
    fill_dbl("ascale", ref.ascale);
    fill_dbl("hcount", ref.hcount);
    const auto &aiw_all = e.at("alphaiw");
    for (std::size_t i = 0; i < aiw_all.size() && i < MAX_REF; ++i) {
      const auto &row = aiw_all[i];
      for (int k = 0; k < N_FREQ; ++k) ref.alphaiw[i][k] = row[k];
    }
  }
  return out;
}

} // namespace

const ReferenceData &reference_data() {
  static const ReferenceData data = [] {
    const std::string path = locate_refdata();
    occ::log::debug("Loading DFT-D4 reference data from {}", path);
    return load_from_json(path);
  }();
  return data;
}

} // namespace occ::disp::d4_data
