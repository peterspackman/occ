#pragma once
#include <array>
#include <string>
#include <vector>

namespace occ::xtb {

struct ShellParam {
  int n;                // principal quantum number
  int l;                // angular momentum (0=s, 1=p, 2=d, 3=f)
  int n_prim;           // number of Gaussian primitives in STO-NG expansion
  bool is_valence;      // false → polarization-only, ref_occ = 0
  double self_energy_ev;
  double slater_exponent;
  double kcn_au;        // CN-shift coefficient (already × 0.1)
  double shell_poly;    // distance-polynomial coefficient
  double ref_occ;       // reference atomic occupation
  double shell_hardness_au;  // (already × 0.1) — usually 0 in GFN2
};

struct ElementParam {
  int z;
  std::string ao;          // shell pattern, e.g. "2s2p3d"
  double pauling_en;
  double atomic_hardness;          // GAM
  double third_order_atom_au;      // GAM3 × 0.1
  double rep_alpha;                // REPA
  double rep_zeff;                 // REPB
  double dip_kernel;               // DPOL × 0.01
  double quad_kernel;              // QPOL × 0.01
  std::vector<ShellParam> shells;
};

struct GlobalParam {
  // H0 shell scalings (s, p, d, f)
  std::array<double, 4> kshell{};
  double ksp{0.0};
  double ksd{0.0};
  double kpd{0.0};
  double kdiff{0.0};

  // Electronegativity scaling for H0
  double enscale{0.0};
  double enscale4{0.0};

  // IP/EA scissor (already × 0.1)
  double ipeashift_au{0.0};

  // Third-order shell weights, gam3shell[l][which] (which=0 or 1; for d
  // these can differ — GFN2 splits as d1/d2)
  std::array<std::array<double, 2>, 4> gam3shell{};

  // Anisotropic electrostatic damping
  double aesshift{0.0};
  double aesexp{0.0};
  double aesrmax{0.0};
  double aesdmp3{0.0};
  double aesdmp5{0.0};

  // Klopman/gamma exponent (alphaj in xtb)
  double alphaj{0.0};

  // D4 dispersion damping
  double a1{0.0};
  double a2{0.0};
  double s6{1.0};
  double s8{0.0};
  double s9{0.0};

  // Repulsion exponents
  double kexp{0.0};
  double kexplight{0.0};
};

class Gfn2Parameters {
public:
  // Load from the bundled share/xtb/gfn2.json (or a user-supplied JSON path).
  // Throws std::runtime_error if the file cannot be read or parsed.
  static Gfn2Parameters load_default();
  static Gfn2Parameters load(const std::string &path);

  const std::string &method() const { return m_method; }
  const std::string &doi() const { return m_doi; }
  int max_z() const { return static_cast<int>(m_elements.size()); }

  const GlobalParam &globals() const { return m_globals; }

  // Element access by atomic number Z (1-based). Returns nullptr if missing.
  const ElementParam *element(int z) const;

  const std::vector<ElementParam> &elements() const { return m_elements; }

  // Mutators used by the JSON loader; not part of the user-facing API.
  void set_method(std::string s) { m_method = std::move(s); }
  void set_doi(std::string s) { m_doi = std::move(s); }
  void set_globals(GlobalParam g) { m_globals = std::move(g); }
  void add_element(ElementParam e) { m_elements.push_back(std::move(e)); }

private:
  std::string m_method;
  std::string m_doi;
  GlobalParam m_globals;
  std::vector<ElementParam> m_elements;
};

} // namespace occ::xtb
