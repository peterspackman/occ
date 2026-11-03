#pragma once
#include <array>
#include <memory>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::gto {
class AOBasis;
}

/// Effective core potential (ECP) integrals over Cartesian Gaussians.
///
/// The operator about an ECP centre C is
///
///   U(r) = U_L(r) + sum_{l<L} sum_m |lm> U_l(r) <lm|,
///   U_l(r) = sum_k d_k r^(n_k - 2) exp(-zeta_k r^2),
///
/// with the local channel L the highest angular momentum on that centre and
/// the channels below it stored as U_l - U_L (the BSE/Gaussian convention).
///
/// The local (type-1) term is a three-centre integral done analytically with
/// McMurchie-Davidson Hermite expansions: r^0 is a plain three-Gaussian
/// overlap, r^-1 reduces to Boys functions and r^-2 to the analogous
/// G_m(T) = int_0^1 (1-v^2)^m exp(-T(1-v^2)) dv.
///
/// The semilocal (type-2) term uses the half-numerical scheme of
/// Flores-Moreno et al., J. Comput. Chem. 27, 1009 (2006): each Gaussian is
/// expanded about C in real spherical harmonics with the plane-wave
/// expansion, the angular integrals are exact (tabulated from harmonic
/// polynomials), and the radial integrals
///
///   int r^(N+n) exp(-zeta r^2 - a(r-|A|)^2 - b(r-|B|)^2)
///       i~_la(2a|A|r) i~_lb(2b|B|r) dr
///
/// (i~ the exponentially scaled modified spherical Bessel function) are done
/// per primitive triple with fixed-order Gauss-Legendre on a window placed
/// around the peak of that integrand. The window depends smoothly on the
/// geometry and the order is fixed, so the integrals are smooth functions of
/// the nuclear positions.
///
/// There is no fixed angular momentum limit. Derivatives with respect to the
/// basis centres come from shells with l raised or lowered (l +- 1, l +- 2)
/// and exponent-weighted coefficients; derivatives involving C follow from
/// translational invariance.
namespace occ::ints {

struct EcpPrimitive {
  int n{2};          // r^(n-2)
  double zeta{0.0};  // exponent
  double coeff{0.0}; // d_k
};

/// All ECP terms on one atom.
struct EcpCentre {
  int atom{-1};
  Vec3 origin{0, 0, 0};
  int L{0};                                      // local channel
  std::vector<EcpPrimitive> local;               // U_L
  std::vector<std::vector<EcpPrimitive>> semilocal; // U_l - U_L, l = 0..L-1
  double min_zeta{0.0};
};

/// A contracted Cartesian AO shell as the ECP code sees it. `coeffs` are
/// occ's (libcint-normalised) contraction coefficients with the libcint s/p
/// factors 1/(2 sqrt(pi)) and sqrt(3)/(2 sqrt(pi)) folded in, so that
/// coeffs[i] * (x-Ax)^i (y-Ay)^j (z-Az)^k exp(-a_i r_A^2) summed over
/// primitives is the Cartesian basis function occ's other integrals use.
struct EcpBasisShell {
  int l{0};
  int atom{-1};
  Vec3 origin{0, 0, 0};
  std::vector<double> exponents;
  std::vector<double> coeffs;
};

std::vector<EcpCentre> ecp_centres_from_basis(const occ::gto::AOBasis &basis);
std::vector<EcpBasisShell> ecp_shells_from_basis(const occ::gto::AOBasis &basis);

/// Index of the Cartesian component (i,j,k) in occ's default ordering.
inline constexpr int ecp_cart_index(int i, int j, int k) {
  const int l = i + j + k;
  return ((l - i) * (l - i + 1)) / 2 + (l - i - j);
}

/// Position of d2/dX_i dX_j in the packed xx, xy, xz, yy, yz, zz order of
/// the same-centre second-derivative blocks.
inline constexpr int ecp_sym_index(int i, int j) {
  constexpr int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  return idx[i][j];
}

class EcpEvaluator {
public:
  struct Settings {
    /// Primitive (pair, ECP term) contributions whose bound falls below this
    /// are skipped. Kept far below any target so the screening introduces no
    /// visible steps in finite differences.
    double screening_threshold{1e-20};
  };

  EcpEvaluator(std::vector<EcpBasisShell> shells,
               std::vector<EcpCentre> centres, int max_deriv = 2);
  explicit EcpEvaluator(const occ::gto::AOBasis &basis, int max_deriv = 2);
  ~EcpEvaluator();
  EcpEvaluator(EcpEvaluator &&) noexcept;
  EcpEvaluator &operator=(EcpEvaluator &&) noexcept;

  const std::vector<EcpCentre> &centres() const { return m_centres; }
  const std::vector<EcpBasisShell> &shells() const { return m_shells; }
  int max_basis_l() const { return m_max_basis_l; }
  int max_deriv() const { return m_max_deriv; }
  Settings &settings() { return m_settings; }
  const Settings &settings() const { return m_settings; }

  /// Angular data for every AO shell about one ECP centre, built once and
  /// shared read-only between threads.
  struct CentreData;
  std::shared_ptr<const CentreData> prepare(int centre, int deriv) const;

  /// Thread-local scratch space.
  struct Workspace;
  std::shared_ptr<Workspace> make_workspace() const;

  /// Can the pair (sa, sb) see the centre at all? Both shells must pass the
  /// bound on their reach to C computed in prepare().
  bool significant(const CentreData &, int sa, int sb) const;

  /// V block, row-major ncart(la) x ncart(lb).
  void compute(const CentreData &, int sa, int sb, Workspace &,
               Eigen::Ref<MatRM> out) const;

  /// First derivatives with respect to the basis centres, C held fixed.
  /// out[0..2] = d/dA_{x,y,z}, out[3..5] = d/dB_{x,y,z}; each block row-major
  /// ncart(la) x ncart(lb). Blocks not requested are left untouched.
  void compute_first_derivatives(const CentreData &, int sa, int sb,
                                 bool want_a, bool want_b, Workspace &,
                                 std::array<MatRM, 6> &out) const;

  /// Second derivatives with respect to the basis centres, C held fixed.
  /// aa: xx,xy,xz,yy,yz,zz on A; ab: 3*i + j with i on A and j on B;
  /// bb: as aa on B. Blocks not requested are left untouched.
  void compute_second_derivatives(const CentreData &, int sa, int sb,
                                  bool want_aa, bool want_ab, bool want_bb,
                                  Workspace &, std::array<MatRM, 6> &aa,
                                  std::array<MatRM, 9> &ab,
                                  std::array<MatRM, 6> &bb) const;

  /// Convenience wrappers (prepare + workspace per call): for tests.
  MatRM shell_pair(int centre, int sa, int sb) const;
  /// Layout [Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz], with the C blocks from
  /// translational invariance.
  std::array<MatRM, 9> shell_pair_first_derivatives(int centre, int sa,
                                                    int sb) const;
  /// AA (6), AB (9), BB (6).
  void shell_pair_second_derivatives(int centre, int sa, int sb,
                                     std::array<MatRM, 6> &aa,
                                     std::array<MatRM, 9> &ab,
                                     std::array<MatRM, 6> &bb) const;

  /// One variant of the integral: shells with angular momenta la' = la + dla,
  /// lb' = lb + dlb and primitive coefficients scaled by alpha^ka, beta^kb.
  struct Variant {
    int dla{0}, dlb{0};
    int ka{0}, kb{0};
  };
  void compute_variants(const CentreData &, int sa, int sb,
                        const std::vector<Variant> &variants, Workspace &,
                        std::vector<MatRM> &out) const;

  struct Tables;

private:
  std::vector<EcpBasisShell> m_shells;
  std::vector<EcpCentre> m_centres;
  int m_max_basis_l{0};
  int m_max_deriv{2};
  int m_max_semilocal_l{-1};
  Settings m_settings;
  std::shared_ptr<const Tables> m_tables;
  std::vector<double> m_shell_log_cmax; // max ln|c| per shell
};

namespace ecp_detail {
/// Exponentially scaled modified spherical Bessel functions of the first kind,
/// out[l] = exp(-z) i_l(z) for l = 0..lmax. Relative accuracy ~1e-14.
void scaled_bessel_i(int lmax, double z, double *out);
/// Boys functions F_m(T), m = 0..mmax.
void boys(int mmax, double T, double *out);
/// G_m(T) = int_0^1 (1-v^2)^m exp(-T(1-v^2)) dv, m = 0..mmax.
void gfunc(int mmax, double T, double *out);
} // namespace ecp_detail

} // namespace occ::ints
