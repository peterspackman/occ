#include <occ/core/constants.h>
#include <occ/elastic_fit/monkhorst_pack.h>
#include <occ/elastic_fit/pes.h>
#include <iomanip>

namespace occ::elastic_fit {

using occ::units::EV_TO_KJ_PER_MOL;
using occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
using occ::units::PI;

inline void save_matrix(const occ::Mat &matrix, const std::string &filename,
                        std::vector<std::string> comments = {},
                        bool upper_triangle_only = false, int width = 6) {
  std::ofstream file(filename);
  occ::log::info("Writing matrix to file {}", filename);
  for (const auto &comment : comments) {
    file << "# " << comment << std::endl;
  }
  file << std::fixed << std::setprecision(4);

  if (upper_triangle_only) {
    int count = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = i; j < matrix.cols(); ++j) {
        file << std::setw(12) << matrix(i, j);
        count++;
        if (count % width == 0) {
          file << std::endl;
        } else {
          file << " ";
        }
      }
    }
    if (count % width != 0) {
      file << std::endl;
    }
    file.close();
    return;
  }

  int count = 0;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << std::setw(12) << matrix(i, j);
      count++;
      if (count % width == 0) {
        file << std::endl;
      } else {
        file << " ";
      }
    }
  }
  if (count % width != 0) {
    file << std::endl;
  }

  file.close();
}

inline void print_vector(const occ::Vec &vec, int per_line) {
  std::string line;

  occ::Vec sorted_vec = vec;
  std::sort(sorted_vec.begin(), sorted_vec.end());

  for (size_t i = 0; i < sorted_vec.size(); ++i) {
    double val = sorted_vec(i);
    line += fmt::format("{:9.3f}", val);
    if ((i + 1) % per_line == 0 || i == vec.size() - 1) {
      spdlog::info("{}", line);
      line.clear();
    }
  }
}

occ::Mat PES::solve_linear_system(const occ::Mat &A, const occ::Mat &B,
                                  LinearSolverType solver_type,
                                  double svd_threshold) {
  switch (solver_type) {
  case LinearSolverType::LU:
    return A.lu().solve(B);
  case LinearSolverType::SVD: {
    Eigen::JacobiSVD<occ::Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(svd_threshold); // Added to control svd_threshold  
    return svd.solve(B);
  }
  case LinearSolverType::QR:
    return A.householderQr().solve(B);
  case LinearSolverType::LDLT:
    return A.ldlt().solve(B);
  default:
    throw std::runtime_error("Unknown linear solver type");
  }
}

occ::Mat6 PES::compute_elastic_tensor(double volume,
                                      LinearSolverType solver_type,
                                      double svd_threshold,
                                      bool save_debug_matrices) {

  size_t n_molecules = this->num_unique_molecules();
  size_t dim = 3 * n_molecules;

  occ::Mat6 D_ee = occ::Mat6::Zero();       // Strain-Strain
  occ::Mat D_ei = occ::Mat::Zero(6, dim);   // Strain-Cart
  occ::Mat D_ij = occ::Mat::Zero(dim, dim); // Cart-Cart

  for (const auto &pot : m_potentials) {
    const occ::Vec3 &r_hat = pot->r_hat;
    const occ::Vec3 &r_vector = pot->r_vector;
    double r = pot->r0;
    double r2 = r * r;
    double dU_dr = pot->first_derivative();
    double d2U_dr2 = pot->second_derivative();
    const auto [idx_0, idx_1] = pot->uc_pair_indices();

    for (int p = 0; p < 6; p++) {
      auto [alpha, beta] = this->voigt_notation(p);
      for (int q = 0; q < 6; q++) {
        auto [gamma, delta] = this->voigt_notation(q);

        double term1 = (r_vector[alpha] * r_vector[beta] * r_vector[gamma] *
                        r_vector[delta]) *
                       d2U_dr2 / r2;

        // NOTE: term2 will be zero for us by definition.
        // double term2 = dU_dr / r *
        //                ((alpha == gamma ? r_hat[beta] * r_hat[delta] * r2:
        //                0) +
        //                 (alpha == delta ? r_hat[beta] * r_hat[gamma] * r2:
        //                 0) + (beta == gamma ? r_hat[alpha] * r_hat[delta] *
        //                 r2: 0) + (beta == delta ? r_hat[alpha] *
        //                 r_hat[gamma] * r2: 0));
        // D_ee(p, q) += term1 + term2;

        D_ee(p, q) += term1;
      }
      for (int coord = 0; coord < 3; coord++) {
        double mixed_term =
            d2U_dr2 * r_vector[alpha] * r_vector[beta] * r_vector[coord] / r2;

        // NOTE: zero by definition.
        // mixed_term += dU_dr / r *
        //               ((alpha == coord ? r_vector[beta]: 0) +
        //                (beta == coord ? r_vector[alpha]: 0));

        D_ei(p, idx_0 * 3 + coord) -= mixed_term;
        D_ei(p, idx_1 * 3 + coord) += mixed_term;
      }
    }
    for (int alpha = 0; alpha < 3; alpha++) {
      for (int beta = 0; beta < 3; beta++) {
        double e = d2U_dr2 * r_hat[alpha] * r_hat[beta];

        // NOTE: zero by definition
        // if (alpha == beta) {
        //   e += dU_dr / r;
        // }

        D_ij(idx_0 * 3 + alpha, idx_0 * 3 + beta) += e;
        D_ij(idx_1 * 3 + alpha, idx_1 * 3 + beta) += e;
        D_ij(idx_0 * 3 + alpha, idx_1 * 3 + beta) -= e;
        D_ij(idx_1 * 3 + alpha, idx_0 * 3 + beta) -= e;
      }
    }
  }

  D_ee /= 2;
  D_ei /= 2;
  D_ij /= 2;

  occ::Mat mass_inv_ij = this->inv_mass_matrix();
  occ::Mat Dyn_ij = mass_inv_ij.cwiseProduct(D_ij);

  occ::Mat X = solve_linear_system(D_ij, D_ei.transpose(), solver_type,
                                   svd_threshold); // D_ij * X = D_ei^T
  occ::Mat6 correction = D_ei * X; // This gives D_ei * D_ij^(-1) * D_ei^T
  occ::Mat6 C = (D_ee - correction) / volume * KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

  if (save_debug_matrices) {
    save_matrix(D_ij, "D_ij.txt",
                {"Cartesian-cartesian Hessian "
                 "(kJ/mol/Ang**2)"});
    save_matrix(D_ei.transpose(), "D_ei.txt",
                {"Strain-cartesian second derivative matrix "
                 " (kJ/mol/Ang)"},
                false);
    save_matrix(D_ee, "D_ee.txt",
                {"Strain-strain second derivative matrix "
                 "(kJ/mol)"},
                false);
    save_matrix(Dyn_ij, "Dyn_ij.txt",
                {"Dynamical cartesian-cartesian Hessian "
                 "(kJ/Ang**2/kg)"},
                false);
    save_matrix(D_ij / EV_TO_KJ_PER_MOL, "D_ij_gulp.txt",
                {"Cartesian-cartesian Hessian"
                 "(eV/Ang**2)"},
                false, 3);
    save_matrix(D_ee / EV_TO_KJ_PER_MOL, "D_ee_gulp.txt",
                {"Strain-strain second derivative matrix (eV)"}, false, 3);
    save_matrix(D_ei.transpose() / EV_TO_KJ_PER_MOL, "D_ei_gulp.txt",
                {"Strain-cartesian second derivative "
                 "(eV/Ang)"},
                false, 3);
  }

  return C;
}

void PES::phonons(const occ::IVec3 &shrinking_factors, const occ::Vec3 shift,
                  bool animate) {
  MonkhorstPack mp(shrinking_factors, shift);
  size_t n_kpoints = mp.size();
  double weight = 1 / static_cast<double>(n_kpoints);
  size_t n_molecules = this->num_unique_molecules();
  const occ::Mat &inv_m_ij = this->inv_mass_matrix();
  double temp = this->get_temperature();

  double Uvib = 0;
  double kBT = temp * occ::constants::boltzmann<double> *
               occ::constants::avogadro<double> / 1000;
  double zpe = 0;

  std::vector<occ::Vec> all_freqs;
  std::vector<occ::Mat> all_eigvecs;
  all_freqs.reserve(n_kpoints);
  all_eigvecs.reserve(n_kpoints);

  occ::log::info("Phonon frequencies (cm-1): ");
  for (const auto &kpoint : mp) {
    const auto &fm = this->compute_fm_at_kpoint(kpoint);
    occ::CMat dyn = fm.cwiseProduct(inv_m_ij);
    occ::CMat dyn_gulp = dyn / 96486.256;
    const auto &[freqs, eigvecs] = this->compute_phonons_at_kpoint(dyn);
    occ::log::info(" @ k-point {:8.4f} {:8.4f} {:8.4f} Weight = {:8.4f}:",
                   kpoint[0], kpoint[1], kpoint[2], weight);
    // print_matrix_full(dyn_gulp);
    occ::log::info("");
    print_vector(freqs, 6);
    occ::log::info("");
    for (size_t f_idx = 0; f_idx < freqs.size(); f_idx++) {
      double f_wvn = freqs[f_idx];
      if (f_wvn < 1e-2) {
        occ::log::debug("Skipping imaginary or translational phonon mode with "
                        "frequency {:.3f} cm-1 in Uvib calc.",
                        f_wvn);
        continue;
      }
      double f_Hz =
          occ::constants::speed_of_light<double> / ((1 / f_wvn) / 100);
      double Uf = f_Hz * occ::constants::planck<double> *
                  occ::constants::avogadro<double> / 1000;
      double f_zpe = 0.5 * Uf;
      zpe += f_zpe * weight;
      if (kBT > 0.0) {
        Uvib += (f_zpe + Uf / (std::exp(Uf / kBT) - 1)) * weight;
      }
    }

    if (!animate) {
      continue;
    }

    this->animate_phonons(freqs, eigvecs, kpoint);
  }

  occ::log::info("Zero point energy                           = {:8.3f} kJ/mol",
                 zpe);
  if (!(kBT > 0.0)) {
    return;
  }
  occ::log::info("Vibrational energy properties at {:.2f} K: ", temp);
  occ::log::info("Uvib (excluding rovibrations)               = {:8.3f} kJ/mol",
                 Uvib);
  double rovib = 3 * kBT * n_molecules;
  occ::log::info("Uvib (including equipartition rovibrations) = {:8.3f} kJ/mol",
                 Uvib + rovib);
  occ::log::info("                                            = {:8.3f} "
                 "kJ/mol/molecule",
                 (Uvib + rovib) / n_molecules);
  occ::log::info("                                            = {:8.3f} "
                 "kBT/molecule",
                 (Uvib + rovib) / kBT / n_molecules);
  double equip = 6 * kBT * n_molecules;
  occ::log::info("Uvib (equipartition)                        = {:8.3f} kJ/mol",
                 equip);
  occ::log::info("");
}

occ::Mat PES::inv_mass_matrix() {
  size_t n_molecules = this->num_unique_molecules();
  size_t dim = 3 * n_molecules;
  occ::Mat mass_inv_ij = occ::Mat::Zero(dim, dim);

  for (const auto &pot : m_potentials) {
    const auto [idx_0, idx_1] = pot->uc_pair_indices();
    const auto [m0, m1] = pot->pair_mass();

    for (int alpha = 0; alpha < 3; alpha++) {
      for (int beta = 0; beta < 3; beta++) {

        mass_inv_ij(idx_0 * 3 + alpha, idx_0 * 3 + beta) =
            1 / std::sqrt(m0 * m0);
        mass_inv_ij(idx_1 * 3 + alpha, idx_1 * 3 + beta) =
            1 / std::sqrt(m1 * m1);
        mass_inv_ij(idx_0 * 3 + alpha, idx_1 * 3 + beta) =
            1 / std::sqrt(m0 * m1);
        mass_inv_ij(idx_1 * 3 + alpha, idx_0 * 3 + beta) =
            1 / std::sqrt(m1 * m0);
      }
    }
  }
  return mass_inv_ij;
}

occ::CMat PES::compute_fm_at_kpoint(const occ::Vec3 &kp) {

  size_t n_molecules = this->num_unique_molecules();
  size_t dim = 3 * n_molecules;
  const auto &recip = this->crystal().unit_cell().reciprocal();

  occ::CMat D_ij = occ::Mat::Zero(dim, dim);

  for (const auto &pot : m_potentials) {
    const occ::Vec3 &r_hat = pot->r_hat;
    const occ::Vec3 &r_vector = pot->r_vector;
    auto q = recip * kp * 2 * PI;
    std::complex<double> phase =
        std::exp(std::complex<double>(0.0, q.dot(r_vector)));
    double d2U_dr2 = pot->second_derivative();
    const auto [idx_0, idx_1] = pot->uc_pair_indices();

    for (int alpha = 0; alpha < 3; alpha++) {
      for (int beta = 0; beta < 3; beta++) {
        std::complex<double> e = d2U_dr2 * r_hat[alpha] * r_hat[beta];

        // NOTE: zero by definition
        // if (alpha == beta) {
        //   e += dU_dr / r;
        // }

        D_ij(idx_0 * 3 + alpha, idx_0 * 3 + beta) += e;
        D_ij(idx_1 * 3 + alpha, idx_1 * 3 + beta) += e;
        D_ij(idx_0 * 3 + alpha, idx_1 * 3 + beta) -= e * phase;
        D_ij(idx_1 * 3 + alpha, idx_0 * 3 + beta) -= e * phase;
      }
    }
  }

  D_ij /= 2;

  return D_ij;
}

std::pair<occ::Vec, occ::Mat>
PES::compute_phonons_at_kpoint(const occ::CMat &Dyn_ij) {

  Eigen::ComplexEigenSolver<occ::CMat> es(Dyn_ij);
  occ::CVec eigenvalues = es.eigenvalues();
  occ::CMat eigenvectors = es.eigenvectors();
  occ::Vec frequencies;
  frequencies.resize(eigenvalues.size());
  for (size_t i = 0; i < eigenvalues.size(); ++i) {
    std::complex<double> eval = eigenvalues(i);
    if (eval.real() < 0) {
      frequencies(i) = -std::sqrt(-eval.real());
    } else {
      frequencies(i) = std::sqrt(eval.real());
    }
  }

  occ::Vec freq_Hz = frequencies * std::sqrt(1e23);
  occ::Vec freq_wavenumbers =
      freq_Hz / (occ::constants::speed_of_light<double> * 100.0) / (2.0 * PI);

  size_t n = eigenvalues.size();
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int i, int j) {
    return freq_wavenumbers(i) < freq_wavenumbers(j);
  });
  occ::Vec sorted_freq_wavenumbers(n);
  occ::Mat sorted_eigenvectors(eigenvectors.rows(), n);
  for (size_t i = 0; i < n; ++i) {
    sorted_freq_wavenumbers(i) = freq_wavenumbers(indices[i]);
    sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]).real();
  }

  return std::pair(sorted_freq_wavenumbers, sorted_eigenvectors);
}

void PES::animate_phonons(const occ::Vec &frequencies,
                          const occ::Mat &eigenvectors,
                          const occ::Vec3 &kpoint) {

  double amplitude = 0.5;
  size_t n_frames = 50;
  double period = 2.0 * PI;
  size_t n_modes = frequencies.size();
  size_t n_molecules = this->num_unique_molecules();
  size_t n_atoms = 0;

  occ::log::info("Animating {} phonon modes over {} frames with amplitude {} "
                 "Angstroms\n@ kpoint {:8.4f} {:8.4f} {:8.4f}",
                 n_modes, n_frames, amplitude, kpoint[0], kpoint[1], kpoint[2]);

  std::vector<double> masses(n_molecules);
  std::vector<occ::Vec3> positions(n_molecules);
  std::vector<size_t> mol_idxs;
  const auto &uc_mols = this->crystal().unit_cell_molecules();
  for (const auto &mol : uc_mols) {
    double m = mol.molar_mass();
    const int mol_idx = mol.unit_cell_molecule_idx();
    masses[mol_idx] = m;
    positions[mol_idx] = mol.center_of_mass();
    mol_idxs.push_back(mol_idx);
    n_atoms += mol.positions().cols();
  }

  for (size_t mode = 0; mode < n_modes; mode++) {
    double freq_cm = frequencies(mode);

    std::string filename =
        fmt::format("phonon_mode_k{:.2f}-{:.2f}-{:.2f}_{}_{:.2f}cm-1.xyz",
                    kpoint[0], kpoint[1], kpoint[2], mode, freq_cm);

    std::ofstream xyz_file(filename);
    if (!xyz_file.is_open()) {
      occ::log::error("Could not open file: {}", filename);
      continue;
    }

    occ::log::info("Generating trajectory for mode {} ({:.2f} cm-1)", mode,
                   freq_cm);

    occ::Vec mode_eigenvector = eigenvectors.col(mode);

    double norm = mode_eigenvector.norm();
    if (norm > 1e-10) {
      mode_eigenvector /= norm;
    }

    for (size_t frame = 0; frame < n_frames; frame++) {
      double t = static_cast<double>(frame) /
                 (static_cast<double>(n_frames - 1)) * period;
      double phase = std::cos(t);

      xyz_file << n_atoms << std::endl;
      xyz_file << fmt::format("# Phonon mode {} at {:.2f} cm-1 @ k-point "
                              "{:8.4f} {:8.4f} {:8.4f}, frame {}",
                              mode, freq_cm, kpoint[0], kpoint[1], kpoint[2],
                              frame)
               << std::endl;

      for (const size_t &mol : mol_idxs) {
        occ::Vec3 displacement;
        displacement[0] = mode_eigenvector[3 * mol + 0] * amplitude * phase;
        displacement[1] = mode_eigenvector[3 * mol + 1] * amplitude * phase;
        displacement[2] = mode_eigenvector[3 * mol + 2] * amplitude * phase;

        const auto &molecule = uc_mols[mol];
        const auto &atomic_pos = molecule.positions();
        const auto &elements = molecule.elements();

        for (size_t atom_idx = 0; atom_idx < atomic_pos.cols(); atom_idx++) {
          occ::Vec3 position = atomic_pos.col(atom_idx) + displacement;
          xyz_file << fmt::format("{} {:12.8f} {:12.8f} {:12.8f}",
                                  elements[atom_idx].symbol(), position[0],
                                  position[1], position[2])
                   << std::endl;
        }
      }
    }

    xyz_file.close();
    occ::log::info("Saved trajectory: {}", filename);
  }
  occ::log::info("");
}

} // namespace occ::elastic_fit
