#include "d3_data.h"
#include "d4_data.h" // shared sqrt_zr4r2 table
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/disp/d3.h>
#include <stdexcept>
#include <unordered_map>

namespace occ::disp {

namespace {

using namespace d3_data;

// ============================================================================
// D3 covalent coordination number — erf-counted, no EN factor (k1=16).
// xtb's `xtb_disp_ncoord::ncoord_d3` (and the Grimme s-dftd3 implementation).
// Pyykko & Atsumi 2009 covalent radii (×4/3, in Bohr) like the D4 path, but
// without the Pauling-EN damping factor.
// ============================================================================

namespace cn_consts {
constexpr double k1 = 16.0; // erf steepness (D3 default; D4 uses kn=7.5)
constexpr double bohr_per_angstrom = 1.0 / 0.52917726;

// Pyykko covalent radii (Å), values for metals reduced 10%. Same table used
// in d4.cpp; duplicated here to keep modules self-contained. Index by Z.
constexpr std::array<double, 95> rcov_angstrom = {
    0.0,
    0.32, 0.46,
    1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67,
    1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96,
    1.76, 1.54,
    1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09,
    1.12, 1.09, 1.15, 1.10, 1.14, 1.17,
    1.89, 1.67,
    1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23,
    1.28, 1.26, 1.26, 1.23, 1.32, 1.31,
    2.09, 1.76,
    1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51,
    1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
    1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32,
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42,
    2.01, 1.81,
    1.67, 1.58, 1.52, 1.53, 1.54, 1.55,
};

constexpr double rcov_bohr(int Z) {
  return (4.0 / 3.0) * rcov_angstrom[Z] * bohr_per_angstrom;
}

inline double exp_count(double k, double r, double r0) {
  // D3's ncoord uses 1 / (1 + exp(-k(r0/r - 1))) — the "exponential" counting
  // function (NOT erf, despite the D4 name being similar).
  return 1.0 / (1.0 + std::exp(-k * (r0 / r - 1.0)));
}

} // namespace cn_consts

Vec d3_coordination_numbers(const std::vector<core::Atom> &atoms,
                             double cn_cutoff) {
  using namespace cn_consts;
  const int n = static_cast<int>(atoms.size());
  Vec cn = Vec::Zero(n);
  const double cutoff2 = cn_cutoff * cn_cutoff;
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    if (Zi < 1 || Zi >= (int)rcov_angstrom.size())
      throw std::runtime_error("D3: unsupported element Z=" + std::to_string(Zi));
    const double rci = rcov_bohr(Zi);
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2) continue;
      const double r = std::sqrt(r2);
      const double r0 = rci + rcov_bohr(Zj);
      const double c = exp_count(k1, r, r0);
      cn(i) += c;
      cn(j) += c;
    }
  }
  return cn;
}

// ============================================================================
// Get C6_AB(CN_A, CN_B) from the reference C6 table by Gaussian-weighted
// interpolation over the (refcn_i, refcn_j) grid points. Standard D3 form
// (Grimme et al. 2010, eq. 19).
//
//   C6_AB = Σ_{ki, kj} W_ki·W_kj · C6_ref[Zi, Zj, ki, kj]
//   W_ki  = exp(-k3 · (CN_A − refcn[Zi, ki])^2) / Σ_l exp(-k3 · (CN_A − refcn[Zi, l])^2)
//
// The k3=4 width parameter is part of D3's definition.
// ============================================================================

constexpr double k3 = 4.0;

double get_c6_pair(int Zi, int Zj, double cn_i, double cn_j) {
  const auto &rd = reference_data();
  const int ipair = ReferenceData::pair_index(Zi, Zj);
  const auto &block = rd.c6ab[ipair];
  // Block is indexed as [iref of Z=Zi][iref of Z=Zj] when Zi <= Zj. Need to
  // transpose if Zi > Zj.
  const bool transposed = Zi > Zj;
  const int Zlo = std::min(Zi, Zj);
  const int Zhi = std::max(Zi, Zj);
  const int nlo = rd.nref[Zlo];
  const int nhi = rd.nref[Zhi];
  const auto &cn_lo_table = rd.ref_cn[Zlo];
  const auto &cn_hi_table = rd.ref_cn[Zhi];
  const double cn_lo_value = transposed ? cn_j : cn_i;
  const double cn_hi_value = transposed ? cn_i : cn_j;

  // Per-axis Gaussian weights.
  std::array<double, MAX_REF> w_lo{}, w_hi{};
  for (int k = 0; k < nlo; ++k) {
    const double d = cn_lo_value - cn_lo_table[k];
    w_lo[k] = std::exp(-k3 * d * d);
  }
  for (int k = 0; k < nhi; ++k) {
    const double d = cn_hi_value - cn_hi_table[k];
    w_hi[k] = std::exp(-k3 * d * d);
  }
  // c6ab block layout: rows = high-Z ref, cols = low-Z ref (the iref index
  // for the LOWER atomic number is the inner Fortran index → column under our
  // row-major C++ storage). Only weight components attached to a NON-ZERO
  // reference contribute — both numerator AND denominator exclude missing
  // entries (xtb's get_c6 / Grimme's original D3 formulation).
  double num = 0.0, denom = 0.0;
  for (int kh = 0; kh < nhi; ++kh) {
    for (int kl = 0; kl < nlo; ++kl) {
      const double cref = block[kh][kl];
      if (cref == 0.0) continue;
      const double w = w_lo[kl] * w_hi[kh];
      num += w * cref;
      denom += w;
    }
  }
  if (denom > 0.0) return num / denom;
  // All weights collapsed — fall back to the closest non-zero reference.
  double best = std::numeric_limits<double>::infinity();
  double cref_best = 0.0;
  for (int kh = 0; kh < nhi; ++kh) {
    for (int kl = 0; kl < nlo; ++kl) {
      const double cref = block[kh][kl];
      if (cref == 0.0) continue;
      const double dlo = cn_lo_value - cn_lo_table[kl];
      const double dhi = cn_hi_value - cn_hi_table[kh];
      const double dist = dlo * dlo + dhi * dhi;
      if (dist < best) { best = dist; cref_best = cref; }
    }
  }
  return cref_best;
}

// ============================================================================
// 2-body BJ-damped energy + ATM 3-body (same form as D4).
// sqrt_zr4r2 comes from the D4 reference-data JSON (shared between D3 and D4
// — both use Grimme's r4/r2 expectation values).
// ============================================================================

inline double sqrtZr4r2(int Z) {
  return ::occ::disp::d4_data::reference_data().sqrt_zr4r2[Z];
}

double dispersion_2body(const std::vector<core::Atom> &atoms,
                         const D3Damping &dp, const Vec &cn,
                         double cutoff_bohr) {
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double qj = sqrtZr4r2(Zj);
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-12) continue;
      const double r4r2ij = 3.0 * qi * qj;
      const double r0 = dp.a1 * std::sqrt(r4r2ij) + dp.a2;
      const double r0_2 = r0 * r0;
      const double r0_6 = r0_2 * r0_2 * r0_2;
      const double r0_8 = r0_6 * r0_2;
      const double r6 = r2 * r2 * r2;
      const double r8 = r6 * r2;
      const double cij = get_c6_pair(Zi, Zj, cn(i), cn(j));
      const double c8ij = r4r2ij * cij;
      e -= dp.s6 * cij / (r6 + r0_6) + dp.s8 * c8ij / (r8 + r0_8);
    }
  }
  return e;
}

double dispersion_3body(const std::vector<core::Atom> &atoms,
                         const D3Damping &dp, const Vec &cn,
                         double cutoff_bohr) {
  if (dp.s9 == 0.0) return 0.0;
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  // Pre-compute pairwise C6 + R0 (small overhead; pays off in the triple loop).
  Mat c6(n, n), r0ij(n, n);
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j <= i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double qj = sqrtZr4r2(Zj);
      const double cij = get_c6_pair(Zi, Zj, cn(i), cn(j));
      c6(i, j) = cij;
      c6(j, i) = cij;
      const double r0 = dp.a1 * std::sqrt(3.0 * qi * qj) + dp.a2;
      r0ij(i, j) = r0;
      r0ij(j, i) = r0;
    }
  }
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      const double r2ij = std::pow(atoms[i].x - atoms[j].x, 2) +
                          std::pow(atoms[i].y - atoms[j].y, 2) +
                          std::pow(atoms[i].z - atoms[j].z, 2);
      if (r2ij > cutoff2 || r2ij < 1e-12) continue;
      for (int k = 0; k < j; ++k) {
        const double r2ik = std::pow(atoms[i].x - atoms[k].x, 2) +
                            std::pow(atoms[i].y - atoms[k].y, 2) +
                            std::pow(atoms[i].z - atoms[k].z, 2);
        if (r2ik > cutoff2 || r2ik < 1e-12) continue;
        const double r2jk = std::pow(atoms[j].x - atoms[k].x, 2) +
                            std::pow(atoms[j].y - atoms[k].y, 2) +
                            std::pow(atoms[j].z - atoms[k].z, 2);
        if (r2jk > cutoff2 || r2jk < 1e-12) continue;
        const double rij = std::sqrt(r2ij);
        const double rik = std::sqrt(r2ik);
        const double rjk = std::sqrt(r2jk);
        const double c9 = std::sqrt(std::abs(c6(i, j) * c6(i, k) * c6(j, k)));
        const double r2ijk_sq = r2ij * r2ik * r2jk;
        const double rijk_p3 = (rij * rik * rjk) * r2ijk_sq;
        const double angular = (0.375 * (r2ij + r2jk - r2ik) *
                                (r2ij + r2ik - r2jk) *
                                (r2ik + r2jk - r2ij) / r2ijk_sq +
                                1.0) / rijk_p3;
        const double r0_prod = r0ij(i, j) * r0ij(i, k) * r0ij(j, k);
        const double r_prod = rij * rik * rjk;
        const double damp =
            1.0 / (1.0 + 6.0 * std::pow(r0_prod / r_prod, dp.alp / 3.0));
        e += dp.s9 * c9 * angular * damp;
      }
    }
  }
  return e;
}

// ============================================================================
// Functional damping database for DFT-D3(BJ).
// ============================================================================

const std::unordered_map<std::string, D3Damping> &functional_table() {
  using nlohmann::json;
  static const auto table = [] {
    std::unordered_map<std::string, D3Damping> m;
    namespace fs = std::filesystem;
    auto find_path = []() -> std::string {
      const char *base = occ::get_data_directory();
      if (base) {
        fs::path p = fs::path(base) / "dftd3" / "functionals.json";
        if (fs::exists(p)) return p.string();
      }
      if (fs::exists("dftd3/functionals.json")) return "dftd3/functionals.json";
      if (fs::exists("functionals.json")) return "functionals.json";
      throw std::runtime_error(
          "Cannot locate DFT-D3 functional parameter file (looked at "
          "share/dftd3/functionals.json, dftd3/functionals.json, "
          "functionals.json). Set OCC_DATA_PATH or run from a directory "
          "containing dftd3/functionals.json.");
    };
    std::ifstream in(find_path());
    json j;
    in >> j;
    for (const auto &[name, p] : j.at("functionals").items()) {
      D3Damping d{
          p.value("s6", 1.0), p.value("s8", 0.0), p.value("s9", 1.0),
          p.value("a1", 0.0), p.value("a2", 0.0), p.value("alp", 14),
      };
      m.emplace(name, d);
    }
    return m;
  }();
  return table;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

DispersionD3::DispersionD3(std::vector<core::Atom> atoms)
    : m_atoms(std::move(atoms)) {
  for (const auto &a : m_atoms) {
    if (a.atomic_number < 1 || a.atomic_number > N_ELEMENTS) {
      throw std::runtime_error(
          "DispersionD3: element Z=" + std::to_string(a.atomic_number) +
          " is outside the supported range (1..94)");
    }
  }
  (void)d3_data::reference_data();
}

void DispersionD3::set_functional(const std::string &functional) {
  const auto &table = functional_table();
  auto it = table.find(functional);
  if (it == table.end()) {
    throw std::runtime_error("Unknown DFT-D3 functional: '" + functional +
                             "' (check share/dftd3/functionals.json)");
  }
  m_damping = it->second;
}

void DispersionD3::update_positions(const std::vector<core::Atom> &atoms) {
  if (atoms.size() != m_atoms.size()) {
    throw std::runtime_error("DispersionD3::update_positions: count mismatch");
  }
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (atoms[i].atomic_number != m_atoms[i].atomic_number) {
      throw std::runtime_error(
          "DispersionD3::update_positions: atomic_number changed at index " +
          std::to_string(i));
    }
    m_atoms[i].x = atoms[i].x;
    m_atoms[i].y = atoms[i].y;
    m_atoms[i].z = atoms[i].z;
  }
}

Vec DispersionD3::coordination_numbers() const {
  return d3_coordination_numbers(m_atoms, m_cutoff_cn);
}

double DispersionD3::energy() const {
  const Vec cn = coordination_numbers();
  const double e2 = dispersion_2body(m_atoms, m_damping, cn, m_cutoff_disp2);
  const double e3 = dispersion_3body(m_atoms, m_damping, cn, m_cutoff_disp3);
  return e2 + e3;
}

std::pair<double, Mat3N> DispersionD3::energy_and_gradient() const {
  // Numerical gradient — analytical implementation is a follow-up.
  const double e = energy();
  const int n = static_cast<int>(m_atoms.size());
  Mat3N g = Mat3N::Zero(3, n);
  DispersionD3 tmp(*this);
  constexpr double h = 1.0e-4;
  for (int a = 0; a < n; ++a) {
    for (int k = 0; k < 3; ++k) {
      auto displaced = m_atoms;
      auto eval = [&](double dh) {
        displaced[a].x = m_atoms[a].x + (k == 0 ? dh : 0.0);
        displaced[a].y = m_atoms[a].y + (k == 1 ? dh : 0.0);
        displaced[a].z = m_atoms[a].z + (k == 2 ? dh : 0.0);
        tmp.update_positions(displaced);
        return tmp.energy();
      };
      const double e_p2 = eval(2 * h);
      const double e_p1 = eval(h);
      const double e_m1 = eval(-h);
      const double e_m2 = eval(-2 * h);
      g(k, a) = (-e_p2 + 8 * e_p1 - 8 * e_m1 + e_m2) / (12 * h);
    }
  }
  return {e, g};
}

} // namespace occ::disp
