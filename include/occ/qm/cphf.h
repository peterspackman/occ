#pragma once
#include <functional>
#include <occ/core/linear_algebra.h>
#include <occ/qm/mo.h>
#include <vector>

/// Coupled-perturbed SCF (CPHF / CPKS).
///
/// The solver is method-agnostic: a procedure supplies one operation, the
/// response Fock matrices G[D1] for a batch of symmetric first-order AO
/// densities D1, and everything else here is linear algebra in the
/// virtual-occupied space. For Hartree-Fock G = 2J[D1] - K[D1] (occ's
/// restricted D is the per-spin density); a hybrid scales K and adds the XC
/// kernel; solvent adds the reaction-field response.
///
/// Real, static perturbations, restricted or unrestricted: per spin channel s
///
///   (e_a - e_i) U^s_ai + [C_v^s^T G^s[D(U)] C_o^s]_ai = B^s_ai
///
/// with D^s(U) = C_v^s U^s C_o^s^T + C_o^s U^s^T C_v^s^T, i.e. the orbital
/// Hessian (A + B) applied to U. The channels couple only through G (the
/// Coulomb and, for DFT, the XC kernel). All right-hand sides share each
/// batched G evaluation; converged ones drop out of the batch.
namespace occ::qm::cphf {

/// The occupied and virtual orbitals of one spin channel.
struct SpinChannel {
  Mat Co, Cv;
  Vec eo, ev;
};

/// The spin channels of an SCF reference: one for restricted (two electrons
/// per orbital), alpha then beta for unrestricted.
std::vector<SpinChannel> spin_channels(const MolecularOrbitals &mo);

/// The occupied coefficients of every channel side by side, nbf x
/// (n_alpha [+ n_beta]): the layout of perturbed occupied coefficients Cx
/// and of the responses G[D1] C_o.
Mat occupied_orbitals(const MolecularOrbitals &mo);

/// F^s C_o^s for every channel, side by side in that layout. F is nbf x nbf
/// (one operator for both spins) or, unrestricted, alpha rows over beta rows.
Mat contract_occupied(const Mat &F, const MolecularOrbitals &mo);

/// The response for a batch of perturbed occupied coefficients Cx (in the
/// layout of occupied_orbitals): per channel G^s[D1] C_o^s with
/// D1^s = Cx^s C_o^s^T + C_o^s Cx^s^T, the first-order per-spin density
/// (restricted, G = 2J[D1] - K[D1]) or spin density (unrestricted,
/// G^s = J[D1^a + D1^b] - K[D1^s]). A CPHF rotation U gives Cx^s = C_v^s U^s;
/// an occupied-occupied perturbation M gives Cx^s = C_o^s M^s / 2. The
/// response contracted with C_o is all CPHF and the Hessian ever use (its vo
/// and oo blocks), and lets a method avoid nbf x nbf matrices.
using ResponseFunction =
    std::function<std::vector<Mat>(const std::vector<Mat> &)>;

struct Settings {
  /// Converged when max_ai |residual_ai| < tolerance for every RHS.
  double tolerance{1e-9};
  int max_iterations{100};
};

/// Vectors in the virtual-occupied space (U, B, residuals) are single
/// matrices: nvirt x nocc for a restricted reference; for unrestricted the
/// alpha (nvirt_a x n_alpha) and beta blocks one after the other, column-major,
/// as one column. Solver::block views either per channel.
struct Result {
  std::vector<Mat> U;            ///< one per RHS
  std::vector<double> residuals; ///< final max |residual| per RHS
  std::vector<Mat> R;            ///< final residual B - A U per RHS
  /// G[D(U)] C_o for the final U (the response is linear, so it is carried
  /// along with U at no extra cost).
  std::vector<Mat> G;
  int iterations{0};
  int response_builds{0}; ///< number of D1 densities passed to G in total
  bool converged{false};
};

class Solver {
public:
  Solver(const MolecularOrbitals &mo, ResponseFunction G,
         Settings settings = {});

  /// Solve for every right-hand side B (a vo vector each).
  /// U0, if given, is the starting guess (e.g. a solution with a cheaper,
  /// approximate response).
  Result solve(const std::vector<Mat> &B,
               const std::vector<Mat> *U0 = nullptr) const;

  /// The orbital Hessian (A + B) applied to each U; exposed for testing.
  std::vector<Mat> apply(const std::vector<Mat> &U) const;
  /// As apply, also returning the response G[D(U)] C_o.
  std::vector<Mat> apply(const std::vector<Mat> &U,
                         std::vector<Mat> &G) const;

  /// C_v^s^T X C_o^s for every channel, as a vo vector (X as in
  /// contract_occupied).
  Mat vo_block(const Mat &X) const;
  /// A zero vo vector.
  Mat vo_zero() const;

  /// Channel s of a vo vector, nvirt_s x nocc_s.
  Eigen::Map<Mat> block(Mat &U, int s) const;
  Eigen::Map<const Mat> block(const Mat &U, int s) const;

  /// Electrons per orbital: 2 restricted, 1 unrestricted. A second-order
  /// property summed over vo vectors carries it, e.g. the polarisability
  /// alpha_xy = -2 occupation() vo_block(r_x) . U_y.
  inline double occupation() const { return m_channels.size() == 1 ? 2 : 1; }

  inline const std::vector<SpinChannel> &channels() const { return m_channels; }
  /// e_a - e_i, a vo vector.
  inline const Mat &orbital_energy_differences() const { return m_delta; }

private:
  std::vector<SpinChannel> m_channels;
  std::vector<Eigen::Index> m_offset; ///< channel s starts here in a vo vector
  std::vector<Eigen::Index> m_column; ///< and here in the Cx layout
  Mat m_delta;
  ResponseFunction m_G;
  Settings m_settings;
};

} // namespace occ::qm::cphf
