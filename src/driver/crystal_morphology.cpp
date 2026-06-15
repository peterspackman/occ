#include <Eigen/Geometry>
#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <occ/core/log.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/surface.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/driver/crystal_morphology.h>
#include <occ/geometry/wulff.h>

namespace occ::driver {

using occ::Mat3;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using occ::crystal::HKL;
using occ::geometry::WulffConstruction;

namespace {

constexpr double KJ_PER_MOL_TO_J_PER_M2 = 0.16604390671;

HKL reduced_hkl(const HKL &h) {
  int g = std::gcd(std::gcd(std::abs(h.h), std::abs(h.k)), std::abs(h.l));
  if (g == 0)
    g = 1;
  return HKL{h.h / g, h.k / g, h.l / g};
}

std::string hkl_key(const HKL &h) {
  HKL r = reduced_hkl(h);
  return fmt::format("{},{},{}", r.h, r.k, r.l);
}

// The unit-scale particle polyhedron with active-facet geometry extracted from
// WulffConstruction (works for both equilibrium gamma and user support distances).
// One symmetry-unique facet: its energy, the optimal cut offset occ found, and d-spacing.
struct UniqueFacet {
  HKL hkl;
  double gamma{0.0};
  double offset{0.0};    // optimal cut offset (fraction of d-spacing)
  double dspacing{0.0};  // interplanar spacing (Angstrom); 0 disables snapping
};

struct ParticleShape {
  std::vector<Vec3> normals;     // active facet unit normals
  std::vector<double> distances; // active facet support distances (gamma)
  std::vector<HKL> hkls;         // active facet (representative) Miller indices
  std::vector<double> areas;     // active facet area (unit scale)
  std::vector<double> offsets;   // optimal cut offset per active facet
  std::vector<double> dspacings; // interplanar spacing per active facet
  struct Edge {
    int fa, fb;
    double length;
  };
  std::vector<Edge> edges;
  struct Corner {
    Vec3 pos;
    std::vector<int> facets;
  };
  std::vector<Corner> corners;
  double volume{0.0};

  // Support distance of facet f at scale s, snapped to its optimal molecular cut
  // (the cut plane sits at n.x mod d == offset*d, nearest gamma*s) when d-spacing is known.
  double face_distance(size_t f, double s) const {
    double base = distances[f] * s;
    if (dspacings[f] <= 0.0)
      return base;
    double d = dspacings[f], o = offsets[f] * d;
    return o + d * std::round((base - o) / d);
  }
  std::vector<double> face_distances(double s) const {
    std::vector<double> D(normals.size());
    for (size_t f = 0; f < normals.size(); ++f)
      D[f] = face_distance(f, s);
    return D;
  }
  bool inside(const Vec3 &x, const std::vector<double> &D) const {
    for (size_t f = 0; f < normals.size(); ++f)
      if (normals[f].dot(x) > D[f] + 1e-9)
        return false;
    return true;
  }
};

// Reduce surface-energy facets to one (lowest positive energy) cut per reduced hkl,
// keeping the optimal offset and computing the d-spacing for snapping.
std::vector<UniqueFacet> unique_facets(const std::vector<FacetEnergies> &facets,
                                       const Crystal &crystal) {
  std::map<std::string, UniqueFacet> best;
  std::map<std::string, double> min_energy; // including non-positive cuts
  for (const auto &f : facets) {
    std::string key = hkl_key(f.hkl);
    auto [it_min, inserted] = min_energy.try_emplace(key, f.energy);
    if (!inserted)
      it_min->second = std::min(it_min->second, f.energy);
    if (f.energy <= 1e-6)
      continue;
    auto it = best.find(key);
    if (it == best.end() || f.energy < it->second.gamma)
      best[key] = UniqueFacet{f.hkl, f.energy, f.offset, 0.0};
  }
  for (const auto &[key, emin] : min_energy) {
    if (emin > 1e-6)
      continue;
    if (best.count(key))
      occ::log::warn("Morphology: facet ({}) has a non-positive cut energy "
                     "({:.4f} J/m^2); using its lowest positive cut",
                     key, emin);
    else
      occ::log::warn("Morphology: facet ({}) has no positive cut energy "
                     "({:.4f} J/m^2); excluded from equilibrium shape",
                     key, emin);
  }
  std::vector<UniqueFacet> out;
  for (auto &[k, v] : best) {
    double gnorm = occ::crystal::Surface(v.hkl, crystal).d(); // |G| = 1/d
    v.dspacing = (gnorm > 1e-9) ? 1.0 / gnorm : 0.0;
    out.push_back(v);
  }
  return out;
}

ParticleShape build_shape(const Crystal &crystal,
                          const std::vector<UniqueFacet> &facets) {
  const size_t n = facets.size();
  Mat3N hkl(3, n);
  Vec energies(n);
  for (size_t i = 0; i < n; ++i) {
    hkl(0, i) = facets[i].hkl.h;
    hkl(1, i) = facets[i].hkl.k;
    hkl(2, i) = facets[i].hkl.l;
    energies(i) = facets[i].gamma;
  }
  // (hkl) plane normal in cartesian is reciprocal * hkl; symmetry rotations
  // are applied in (direct) fractional coordinates (matches write_wulff)
  const Mat3 recip = crystal.unit_cell().reciprocal();
  Mat3N hkl_frac = crystal.to_fractional(recip * hkl);
  auto [symop_id, expanded_frac] = crystal.space_group().apply_rotations(hkl_frac);
  const size_t n_sym = crystal.space_group().symmetry_operations().size();
  Vec expanded_energies = energies.replicate(n_sym, 1);
  Mat3N directions = crystal.to_cartesian(expanded_frac);
  directions.colwise().normalize();

  std::vector<HKL> expanded_hkl(directions.cols());
  std::vector<double> expanded_offset(directions.cols()), expanded_d(directions.cols());
  for (int j = 0; j < directions.cols(); ++j) {
    const auto &rep = facets[j % n]; // apply_rotations orders by symop block
    expanded_hkl[j] = rep.hkl;       // representative hkl of the form
    expanded_d[j] = rep.dspacing;
    // a cut at offset o on face (hkl) maps under symop (R, t) to a cut at
    // o + hkl'.t (mod 1) on the rotated face hkl' = R^-T hkl, so faces of a
    // form related by screw axes / glides have shifted optimal terminations
    occ::crystal::SymmetryOperation symop(static_cast<int>(symop_id(j)));
    Vec3 hkl_rep(rep.hkl.h, rep.hkl.k, rep.hkl.l);
    Vec3 hkl_rot = symop.rotation().inverse().transpose() * hkl_rep;
    double o = rep.offset + hkl_rot.dot(symop.translation());
    expanded_offset[j] = o - std::floor(o);
  }

  WulffConstruction wulff(directions, expanded_energies);

  ParticleShape shape;
  std::vector<int> active_of_wulff(wulff.facets().size(), -1);
  std::map<int, std::vector<int>> vertex_facets; // vertex -> active facet indices
  for (size_t f = 0; f < wulff.facets().size(); ++f) {
    const auto &facet = wulff.facets()[f];
    if (facet.point_index.empty())
      continue;
    int k = shape.normals.size();
    active_of_wulff[f] = k;
    shape.normals.push_back(facet.normal);
    shape.distances.push_back(facet.energy);
    shape.hkls.push_back(expanded_hkl[f]);
    shape.areas.push_back(wulff.facet_area(f));
    shape.offsets.push_back(expanded_offset[f]);
    shape.dspacings.push_back(expanded_d[f]);
    for (int v : facet.point_index)
      vertex_facets[v].push_back(k);
  }
  for (const auto &e : wulff.edges()) {
    int a = active_of_wulff[e.facet_a], b = active_of_wulff[e.facet_b];
    if (a >= 0 && b >= 0)
      shape.edges.push_back({a, b, e.length});
  }
  for (const auto &[v, fs] : vertex_facets) {
    if (fs.size() >= 3)
      shape.corners.push_back({wulff.vertices().col(v), fs});
  }
  for (size_t f = 0; f < shape.normals.size(); ++f)
    shape.volume += shape.distances[f] * shape.areas[f];
  shape.volume /= 3.0;
  return shape;
}

// A neighbour interaction of a unit-cell molecule, identified exactly by the
// neighbour's unit-cell molecule index and integer cell offset (no float keys).
struct Bond {
  int target;
  Eigen::Vector3i shift;
  double energy;
};

inline int64_t encode(int u, const Eigen::Vector3i &cell, int64_t n_uc_mols) {
  constexpr int64_t B = 1 << 16, OFF = 1 << 15;
  int64_t c = ((cell.x() + OFF) * B + (cell.y() + OFF)) * B + (cell.z() + OFF);
  return c * n_uc_mols + u;
}

struct ClusterData {
  std::vector<int> uc_idx;            // inside molecules' unit-cell index
  std::vector<Eigen::Vector3i> cell;  // and their integer cell
  std::vector<double> D;              // snapped face support distances at this scale
  ankerl::unordered_dense::set<int64_t> keys;
};

ClusterData tile_inside(const ParticleShape &shape, double s,
                        const Mat3 &direct, const Mat3 &inv_direct,
                        const std::vector<Vec3> &uc_frac) {
  std::vector<double> D = shape.face_distances(s); // snapped to optimal cuts
  // integer cell range covering the scaled shape (plus one-cell pad)
  Vec3 lo = Vec3::Constant(1e30), hi = Vec3::Constant(-1e30);
  for (const auto &c : shape.corners) {
    Vec3 fr = inv_direct * (c.pos * s);
    lo = lo.cwiseMin(fr);
    hi = hi.cwiseMax(fr);
  }
  Eigen::Vector3i nlo = (lo.array() - 1.0).floor().cast<int>();
  Eigen::Vector3i nhi = (hi.array() + 1.0).ceil().cast<int>();

  const int64_t n_uc = uc_frac.size();
  ClusterData cd;
  cd.D = D;
  for (int nx = nlo.x(); nx <= nhi.x(); ++nx)
    for (int ny = nlo.y(); ny <= nhi.y(); ++ny)
      for (int nz = nlo.z(); nz <= nhi.z(); ++nz) {
        Eigen::Vector3i cell(nx, ny, nz);
        for (size_t u = 0; u < uc_frac.size(); ++u) {
          Vec3 cart = direct * (uc_frac[u] + cell.cast<double>());
          if (shape.inside(cart, D)) {
            cd.uc_idx.push_back(u);
            cd.cell.push_back(cell);
            cd.keys.insert(encode(u, cell, n_uc));
          }
        }
      }
  return cd;
}

} // namespace

MorphologyResult compute_crystal_morphology(
    const Crystal &crystal, const CrystalDimers &uc_dimers,
    const CrystalSurfaceEnergies &surface_energies,
    const occ::cg::CrystalGrowthResult &growth_result,
    const MorphologyOptions &options) {

  MorphologyResult result;
  result.shape = options.user_shifts.empty() ? "wulff" : "user";

  // ---- shape -------------------------------------------------------
  std::vector<UniqueFacet> facets;
  if (options.user_shifts.empty()) {
    facets = unique_facets(surface_energies.facets, crystal);
  } else {
    // user/growth morphology: fixed support distances, no optimal-cut snapping
    for (const auto &[hkl, shift] : options.user_shifts)
      facets.push_back(UniqueFacet{hkl, shift, 0.0, 0.0});
  }
  ParticleShape shape = build_shape(crystal, facets);
  if (shape.normals.size() < 4 || shape.volume <= 0) {
    occ::log::warn("Morphology: shape is degenerate/unbounded ({} facets) - "
                   "need more surface energies",
                   shape.normals.size());
    return result;
  }

  // ---- bulk reference & molecular volume ---------------------------
  double crystal_energy_sum = 0.0;
  for (const auto &mr : growth_result.molecule_results)
    crystal_energy_sum += mr.total.crystal_energy;
  result.mu_bulk = 0.5 * crystal_energy_sum / growth_result.molecule_results.size();
  const auto &uc_mols = crystal.unit_cell_molecules();
  result.molecular_volume = crystal.volume() / uc_mols.size();

  // ---- neighbour bonds (from stamped uc_dimers) --------------------
  const Mat3 direct = crystal.unit_cell().direct();
  const Mat3 inv_direct = direct.inverse();
  std::vector<Vec3> uc_frac(uc_mols.size());
  for (size_t i = 0; i < uc_mols.size(); ++i)
    uc_frac[i] = inv_direct * uc_mols[i].centroid();
  std::vector<std::vector<Bond>> bonds(uc_mols.size());
  bool any_energy = false;
  for (size_t i = 0; i < uc_dimers.molecule_neighbors.size() && i < bonds.size();
       ++i) {
    for (const auto &srd : uc_dimers.molecule_neighbors[i]) {
      const auto &mol_a = srd.dimer.a();
      const auto &mol_b = srd.dimer.b();
      int ia = mol_a.unit_cell_molecule_idx();
      int ib = mol_b.unit_cell_molecule_idx();
      Eigen::Vector3i shift =
          (mol_b.cell_shift() - uc_mols[ib].cell_shift()) -
          (mol_a.cell_shift() - uc_mols[ia].cell_shift());
      double energy = srd.dimer.interaction_energy("Total");
      any_energy = any_energy || energy != 0.0;
      bonds[i].push_back({ib, shift, energy});
    }
  }
  if (!any_energy) {
    occ::log::warn("Morphology: all dimer interaction energies are zero - "
                   "uc_dimers must carry energies (InteractionMapper)");
  }

  // ---- facet/edge/corner geometry report ---------------------------
  // group active facets by representative hkl
  std::map<std::string, FacetMorphology> facet_map;
  for (size_t f = 0; f < shape.normals.size(); ++f) {
    auto &fm = facet_map[hkl_key(shape.hkls[f])];
    fm.hkl = reduced_hkl(shape.hkls[f]);
    fm.gamma = shape.distances[f];
    fm.area += shape.areas[f];
  }
  for (auto &[k, v] : facet_map)
    result.facets.push_back(v);

  // edge / corner type lengths and counts (unit scale)
  std::map<std::string, double> edge_type_length;
  std::map<std::string, std::pair<HKL, HKL>> edge_type_hkls;
  for (const auto &e : shape.edges) {
    std::string ka = hkl_key(shape.hkls[e.fa]), kb = hkl_key(shape.hkls[e.fb]);
    if (kb < ka)
      std::swap(ka, kb);
    std::string key = ka + "|" + kb;
    edge_type_length[key] += e.length;
    edge_type_hkls[key] = {reduced_hkl(shape.hkls[e.fa]),
                           reduced_hkl(shape.hkls[e.fb])};
  }
  std::map<std::string, int> corner_type_count;
  std::map<std::string, std::vector<HKL>> corner_type_hkls;
  auto corner_key = [&](const std::vector<int> &fs) {
    std::vector<std::string> ks;
    for (int f : fs)
      ks.push_back(hkl_key(shape.hkls[f]));
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    std::string key;
    for (auto &k : ks)
      key += k + ";";
    return key;
  };
  for (const auto &c : shape.corners) {
    std::string key = corner_key(c.facets);
    corner_type_count[key]++;
    if (!corner_type_hkls.count(key)) {
      std::vector<HKL> hs;
      std::vector<std::string> seen;
      for (int f : c.facets) {
        std::string k = hkl_key(shape.hkls[f]);
        if (std::find(seen.begin(), seen.end(), k) == seen.end()) {
          seen.push_back(k);
          hs.push_back(reduced_hkl(shape.hkls[f]));
        }
      }
      corner_type_hkls[key] = hs;
    }
  }

  // ---- size-dependent excess energy --------------------------------
  // Each facet is snapped to its optimal molecular cut (the equilibrium termination),
  // so a single tiling per size gives the minimum-energy particle - no registry scan.
  std::map<std::string, double> edge_bucket_best, corner_bucket_best;
  const int64_t n_uc = uc_frac.size();
  double best_s = 0.0;
  for (int n_target : options.sizes) {
    double s = std::cbrt(n_target * result.molecular_volume / shape.volume);
    ClusterData cd = tile_inside(shape, s, direct, inv_direct, uc_frac);
    int n_inside = cd.uc_idx.size();
    double e_surf = 0.0, e_edge = 0.0, e_corner = 0.0;
    size_t n_unattributed = 0;
    std::map<std::string, double> edge_bucket, corner_bucket;
    // attribute each broken bond by which facets the outside neighbour violates
    // (where the bond exits the particle): 1 -> surface, 2 -> edge, >=3 -> corner.
    for (size_t m = 0; m < cd.uc_idx.size(); ++m) {
      for (const auto &b : bonds[cd.uc_idx[m]]) {
        Eigen::Vector3i ncell = cd.cell[m] + b.shift;
        if (cd.keys.contains(encode(b.target, ncell, n_uc)))
          continue; // neighbour inside -> bond not broken
        Vec3 ncart = direct * (uc_frac[b.target] + ncell.cast<double>());
        std::vector<int> viol;
        for (size_t f = 0; f < shape.normals.size(); ++f)
          if (shape.normals[f].dot(ncart) > cd.D[f] + 1e-9)
            viol.push_back(f);
        if (viol.empty()) {
          // outside the cluster but inside every face plane - should not
          // happen with exact integer bookkeeping
          n_unattributed++;
          e_surf += b.energy;
        } else if (viol.size() == 1) {
          e_surf += b.energy;
        } else if (viol.size() == 2) {
          e_edge += b.energy;
          std::string ka = hkl_key(shape.hkls[viol[0]]),
                      kb = hkl_key(shape.hkls[viol[1]]);
          if (kb < ka)
            std::swap(ka, kb);
          edge_bucket[ka + "|" + kb] += b.energy;
        } else {
          e_corner += b.energy;
          corner_bucket[corner_key(viol)] += b.energy;
        }
      }
    }
    double f = options.sign * 0.5;
    double total_area = 0.0, total_len = 0.0, e_surf_analytic = 0.0;
    for (size_t fi = 0; fi < shape.areas.size(); ++fi) {
      total_area += shape.areas[fi];
      e_surf_analytic +=
          shape.distances[fi] / KJ_PER_MOL_TO_J_PER_M2 * shape.areas[fi] * s * s;
    }
    for (const auto &e : shape.edges)
      total_len += e.length;
    if (n_unattributed > 0) {
      occ::log::warn("Morphology: {} broken bonds violated no face plane at "
                     "size {} (counted as surface)",
                     n_unattributed, n_target);
    }
    result.samples.push_back(ParticleSample{
        s, n_inside, f * (e_surf + e_edge + e_corner), f * e_surf, f * e_edge,
        f * e_corner, e_surf_analytic, total_area * s * s, total_len * s,
        int(shape.corners.size())});
    if (s > best_s) { // edge/corner energies come from the largest particle
      best_s = s;
      edge_bucket_best = edge_bucket;
      corner_bucket_best = corner_bucket;
    }
  }

  // ---- named edge/corner energies (from the largest particle) ------
  double f = options.sign * 0.5;
  for (auto &[key, hkls] : edge_type_hkls) {
    double bucket = edge_bucket_best.count(key) ? edge_bucket_best[key] : 0.0;
    double length = edge_type_length[key] * best_s;
    EdgeMorphology em;
    em.hkl_a = hkls.first;
    em.hkl_b = hkls.second;
    em.length = edge_type_length[key];
    em.lambda = length > 0 ? f * bucket / length : 0.0;
    result.edges.push_back(em);
  }
  for (auto &[key, hkls] : corner_type_hkls) {
    double bucket = corner_bucket_best.count(key) ? corner_bucket_best[key] : 0.0;
    int count = corner_type_count[key];
    CornerMorphology cm;
    cm.hkls = hkls;
    cm.count = count;
    cm.epsilon = count > 0 ? f * bucket / count : 0.0;
    result.corners.push_back(cm);
  }
  return result;
}

void to_json(nlohmann::json &j, const MorphologyResult &m) {
  j["shape"] = m.shape;
  j["mu_bulk"] = m.mu_bulk;
  j["molecular_volume"] = m.molecular_volume;
  nlohmann::json facets = nlohmann::json::array();
  for (const auto &f : m.facets)
    facets.push_back({{"hkl", {f.hkl.h, f.hkl.k, f.hkl.l}},
                      {"gamma", f.gamma},
                      {"area", f.area}});
  j["facets"] = facets;
  nlohmann::json edges = nlohmann::json::array();
  for (const auto &e : m.edges)
    edges.push_back({{"hkl_a", {e.hkl_a.h, e.hkl_a.k, e.hkl_a.l}},
                     {"hkl_b", {e.hkl_b.h, e.hkl_b.k, e.hkl_b.l}},
                     {"length", e.length},
                     {"lambda", e.lambda}});
  j["edges"] = edges;
  nlohmann::json corners = nlohmann::json::array();
  for (const auto &c : m.corners) {
    nlohmann::json hkls = nlohmann::json::array();
    for (const auto &h : c.hkls)
      hkls.push_back({h.h, h.k, h.l});
    corners.push_back(
        {{"hkls", hkls}, {"count", c.count}, {"epsilon", c.epsilon}});
  }
  j["corners"] = corners;
  nlohmann::json samples = nlohmann::json::array();
  for (const auto &s : m.samples)
    samples.push_back({{"size_scale", s.size_scale},
                       {"n_molecules", s.n_molecules},
                       {"e_excess", s.e_excess},
                       {"e_surface", s.e_surface},
                       {"e_edge", s.e_edge},
                       {"e_corner", s.e_corner},
                       {"e_surface_analytic", s.e_surface_analytic},
                       {"area", s.area},
                       {"edge_length", s.edge_length},
                       {"n_corners", s.n_corners}});
  j["samples"] = samples;
}

} // namespace occ::driver
