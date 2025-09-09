#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>
#include <algorithm>
#include <fmt/os.h>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/io/crystal_json.h>
#include <occ/main/occ_elastic_fit.h>

#include <stdexcept>

using occ::crystal::Crystal;
using occ::main::EFSettings;
using occ::main::LinearSolverType;
using occ::main::LJ_AWrapper;

using occ::main::LJWrapper;
using occ::main::MorseWrapper;
using occ::main::PES;
using occ::main::PotentialType;

using occ::units::degrees;
using occ::units::EV_TO_KJ_PER_MOL;
using occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
using occ::units::PI;

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

inline void print_matrix_full(const occ::CMat &matrix, int precision = 6,
                              int width = 12) {
  for (int i = 0; i < matrix.rows(); ++i) {
    std::string row;
    for (int j = 0; j < matrix.cols(); ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j).real(), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix_full(const occ::Mat6 &matrix, int precision = 6,
                              int width = 12) {
  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int j = 0; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix_upper_triangle(const occ::Mat6 &matrix,
                                        int precision = 3, int width = 9) {

  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int k = 0; k < i; ++k) {
      row += fmt::format("{:{}}", "", width);
    }
    for (int j = i; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix(const occ::Mat6 &matrix,
                         bool upper_triangle_only = false, int precision = 3,
                         int width = 9) {
  if (upper_triangle_only) {
    print_matrix_upper_triangle(matrix, precision, width);
  } else {
    print_matrix_full(matrix, precision, width);
  }
}


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

inline void build_pes_from_json(nlohmann::json j, PES &pes, EFSettings settings,
                                std::vector<std::string> &gulp_strings) {
  const auto &pairs = j["all_pairs"];
  double gs = settings.gulp_scale;
  pes.set_scale(settings.scale_factor);
  PotentialType potential_type = settings.potential_type;
  int discarded_count = 0;
  double discarded_total_energy = 0.0;

  double max_energy = 0.0;
  if (settings.max_to_zero) {
    if (settings.include_positive) {
      occ::log::warn("Can't include positive and set max to zero.");
    }
    for (size_t mol_idx = 0; mol_idx < pairs.size(); mol_idx++) {
      const auto &mol_pairs = pairs[mol_idx];
      for (const auto &pair : mol_pairs) {
        const auto &energies_json = pair["energies"];
        double total_energy = energies_json["Total"];
        if (total_energy > max_energy) {
          max_energy = total_energy;
        }
      }
    }
    occ::log::info("Shifting all pair energies down by {:.4f} kJ/mol",
                   max_energy);
    pes.set_shift(max_energy);
  }

  ankerl::unordered_dense::set<std::string> dedup_gulp_strings;

  for (size_t mol_idx = 0; mol_idx < pairs.size(); mol_idx++) {
    const auto &mol_pairs = pairs[mol_idx];
    for (const auto &pair : mol_pairs) {
      const auto r_arr = pair["rvec"];
      occ::Vec3 rvec(r_arr[0], r_arr[1], r_arr[2]);
      occ::Vec3 unit_vec = rvec.normalized();
      const std::pair<int, int> &pair_indices = pair["pair_indices"];
      const auto [p1, p2] = pair_indices;
      const std::pair<int, int> &uc_pair_indices = pair["pair_uc_indices"];
      const auto [uc_p1, uc_p2] = uc_pair_indices;

      const auto pair_masses = pair["mass"];
      const auto pair_mass =
          std::pair(static_cast<double>(pair_masses[0]),
                    static_cast<double>(pair_masses[1])); // kg / mole
      double m = std::sqrt(pair_mass.first * pair_mass.second);

      const auto &energies_json = pair["energies"];
      double total_energy = energies_json["Total"];
      total_energy -= max_energy;
      const double r0 = pair["r"];
      const double rl = r0 - gs, ru = r0 + gs;
      if (total_energy > 0.0) {
        if (!settings.include_positive) {
          occ::log::debug("Skipping pair with positive total energy {:.4f}",
                          total_energy);
          discarded_count++;
          discarded_total_energy += total_energy;
          continue;
        }
        const double eps = -1.0 * total_energy;
        auto potential = std::make_unique<LJ_AWrapper>(eps, r0, rvec);
        occ::log::debug(
            "Added LJ_A potential: {:30} between pair {:4} {:4} ({:4} {:4})",
            potential->to_string(), p1, p2, uc_p1, uc_p2);
        potential->set_pair_mass(pair_mass);
        pes.add_potential(std::move(potential));
        continue;
      }

      switch (potential_type) {
      case PotentialType::MORSE: {
        double D0 = -1.0 * total_energy;
        double h = std::pow(10, 13);
        double conversion_factor = 1.6605388e-24 * std::pow(h, 2) * 6.0221418;
        double k = m * conversion_factor; // kj/mol/angstrom^2
        double alpha = sqrt(k / (2 * abs(D0)));

        auto potential = std::make_unique<MorseWrapper>(D0, r0, alpha, rvec);
        potential->set_pair_indices(pair_indices);
        potential->set_uc_pair_indices(uc_pair_indices);
        potential->set_pair_mass(pair_mass);
        occ::log::debug("Added Morse potential: {} between pair {} {} ({} {})",
                        potential->to_string(), p1, p2, uc_p1, uc_p2);
        pes.add_potential(std::move(potential));
        break;
      }
      case PotentialType::LJ: {
        double eps = -1.0 * total_energy;
        auto potential = std::make_unique<LJWrapper>(eps, r0, rvec);
        potential->set_pair_indices(pair_indices);
        potential->set_uc_pair_indices(uc_pair_indices);
        potential->set_pair_mass(pair_mass);
        occ::log::debug(
            "Added LJ potential: {:30} between pair {:4} {:4} ({:4} {:4})",
            potential->to_string(), p1, p2, uc_p1, uc_p2);

        auto [smaller, larger] = std::minmax(uc_p1, uc_p2);
        const std::string gulp_str =
            fmt::format("X{} core X{} core {:12.5f} {:12.5f} {:12.5f} {:12.5f}",
                        smaller + 1, larger + 1, eps, r0, rl, ru);
        dedup_gulp_strings.insert(gulp_str);
        pes.add_potential(std::move(potential));
        break;
      }
      case PotentialType::LJ_A: {
        throw std::runtime_error("Should not have happened.");
      }
      }
    }
  }
  for (const auto &str : dedup_gulp_strings) {
    gulp_strings.push_back(str);
  }

  if (discarded_count > 0) {
    occ::log::warn("Discarded {} pairs with positive interaction energies "
                   "(total: {:.3f} kJ/mol)",
                   discarded_count, discarded_total_energy / 2.0);
  }
}

inline PotentialType
determine_potential_type(const std::string &user_preference) {
  if (!user_preference.empty()) {
    if (user_preference == "morse")
      return PotentialType::MORSE;
    if (user_preference == "lj")
      return PotentialType::LJ;
  }
  occ::log::debug("Unrecognised user preference '{}' for potential type",
                  user_preference);
  occ::log::debug("Options are 'morse' or 'lj'"); // TODO: make generic
  occ::log::info("Using default potential type (Lennard-Jones)");
  return PotentialType::LJ;
}


inline LinearSolverType
determine_solver_type(const std::string &user_preference) {
  if (!user_preference.empty()) {
    if (user_preference == "lu")
      return occ::main::LinearSolverType::LU;
    if (user_preference == "svd")
      return occ::main::LinearSolverType::SVD;
    if (user_preference == "qr")
      return occ::main::LinearSolverType::QR;
    if (user_preference == "ldlt")
      return occ::main::LinearSolverType::LDLT;
  }
  occ::log::debug("Unrecognised solver type '{}', using SVD decomposition",
                  user_preference);
  occ::log::debug("Options are 'lu', 'svd', 'qr', 'ldlt'");
  return occ::main::LinearSolverType::SVD;
}

inline occ::IVec3 determine_mp_shrinking_factors(
    const std::vector<size_t> &shrinking_factors_raw) {
  size_t n_sf = shrinking_factors_raw.size();
  if (n_sf == 1) {
    size_t sf = shrinking_factors_raw[0];
    return occ::IVec3(sf, sf, sf);
  } else if (n_sf == 3) {
    return occ::IVec3(shrinking_factors_raw[0], shrinking_factors_raw[1],
                      shrinking_factors_raw[2]);
  } else {
    throw std::runtime_error(fmt::format(
        "Raw shrinking factor input had size {} (should be 1 or 3)", n_sf));
  }
}

inline occ::Vec3 determine_mp_shifts(const std::vector<double> &shifts_raw) {
  size_t n_shifts = shifts_raw.size();
  if (n_shifts == 1) {
    double shift = shifts_raw[0];
    return occ::Vec3(shift, shift, shift);
  } else if (n_shifts == 3) {
    return occ::Vec3(shifts_raw[0], shifts_raw[1], shifts_raw[2]);
  } else {
    throw std::runtime_error(fmt::format(
        "Raw shifts input had size {} (should be 1 or 3)", n_shifts));
  }
}

occ::Mat6 PES::compute_elastic_tensor(double volume,
                                      LinearSolverType solver_type,
                                      double svd_threshold) {

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
  double kBT = temp * occ::constants::boltzmann<double> * occ::constants::avogadro<double> / 1000;
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
      double f_Hz = occ::constants::speed_of_light<double> / ((1 / f_wvn) / 100);
      double Uf = f_Hz * occ::constants::planck<double> * occ::constants::avogadro<double> / 1000;
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
  occ::Vec freq_wavenumbers = freq_Hz / (occ::constants::speed_of_light<double> * 100.0) / (2.0 * PI);

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


inline void analyse_elat_results(const occ::main::EFSettings &settings) {
  std::string filename = settings.json_filename;
  occ::log::info("Reading elat results from: {}", filename);

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Could not open file: {}", filename));
  }

  nlohmann::json j;
  file >> j;

  if (j["result_type"] != "elat") {
    throw std::runtime_error("Invalid JSON: not an elat result file");
  }

  if (!j.contains("all_pairs")) {
    throw std::runtime_error("Need 'all_pairs' in JSON output.");
  }

  occ::log::info("Title: {}", j["title"].get<std::string>());
  occ::log::info("Model: {}", j["model"].get<std::string>());


  Crystal crystal = j["crystal"];

  std::vector<std::string> gulp_strings;
  gulp_strings.push_back("conp prop phon noden hessian");
  gulp_strings.push_back("");
  gulp_strings.push_back("cell");

  const auto &uc = crystal.unit_cell();
  const auto &lengths = uc.lengths();
  const auto &angles = uc.angles();
  std::string cry_str =
      fmt::format("{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}",
                  lengths[0], lengths[1], lengths[2], degrees(angles[0]),
                  degrees(angles[1]), degrees(angles[2]));
  gulp_strings.push_back(cry_str);
  gulp_strings.push_back("");
  gulp_strings.push_back("cart");
  const auto &uc_mols = crystal.unit_cell_molecules();
  for (const auto &mol : uc_mols) {
    const auto &com = mol.center_of_mass();
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("X{} core {:12.8f} {:12.8f} {:12.8f}",
                                       mol_idx + 1, com[0], com[1], com[2]);
    gulp_strings.push_back(gulp_str);
  }
  gulp_strings.push_back("");
  gulp_strings.push_back("element");
  for (const auto &mol : uc_mols) {
    double m = mol.molar_mass() * 1000;
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("mass X{} {:8.4f}", mol_idx + 1, m);
    gulp_strings.push_back(gulp_str);
  }
  gulp_strings.push_back("end");
  gulp_strings.push_back("");
  gulp_strings.push_back("space");
  gulp_strings.push_back("1");
  gulp_strings.push_back("");

  occ::main::EFSettings updated_settings = settings;

  updated_settings.potential_type =
      determine_potential_type(settings.potential_type_str);
  updated_settings.solver_type =
      determine_solver_type(settings.solver_type_str);
  updated_settings.shrinking_factors =
      determine_mp_shrinking_factors(settings.shrinking_factors_raw);
  updated_settings.shift = determine_mp_shifts(settings.shift_raw);

  std::string type_name;
  if (updated_settings.potential_type == PotentialType::MORSE) {
    type_name = "Morse";
    gulp_strings.push_back("morse inter kjmol");
  } else {
    type_name = "Lennard-Jones";
    gulp_strings.push_back("lennard epsilon kjmol");
  }
  occ::log::info("Using {} potential", type_name);

  PES pes(crystal);
  pes.set_temperature(settings.temperature);

  build_pes_from_json(j, pes, updated_settings, gulp_strings);

  gulp_strings.push_back("");
  gulp_strings.push_back("output drv file");

  if (!settings.gulp_file.empty()) {
    occ::log::info("Writing coarse-grained crystal to GULP input '{}'",
                   settings.gulp_file);
    occ::log::warn("The coarse-grained GULP input has not been thoroughly "
                   "tested. Use with caution.");
    std::ofstream gulp_file(settings.gulp_file);
    if (gulp_file.is_open()) {
      std::copy(gulp_strings.begin(), gulp_strings.end(),
                std::ostream_iterator<std::string>(gulp_file, "\n"));
    }
  }

  double elat = pes.lattice_energy(); // per mole of unit cells
  occ::Mat6 cij =
      pes.compute_elastic_tensor(crystal.volume(), updated_settings.solver_type,
                                 updated_settings.svd_threshold);

  if (settings.max_to_zero) {
    double og_elat = elat + pes.number_of_potentials() * pes.shift() / 2.0;
    occ::log::info("Unaltered lattice energy {:.3f} kJ/(mole unit cells)",
                   og_elat);
    occ::log::info("Shifted lattice energy {:.3f} kJ/(mole unit cells)", elat);
    occ::log::info("Shifted elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
    cij *= og_elat / elat;
    occ::log::info("Scaled+shifted elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
  } else {
    occ::log::info("Lattice energy {:.3f} kJ/(mole unit cells)", elat);
    occ::log::info("Elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
  }
  save_matrix(cij, settings.output_file);

  pes.phonons(updated_settings.shrinking_factors, updated_settings.shift,
              updated_settings.animate_phonons);
}

namespace occ::main {

CLI::App *add_elastic_fit_subcommand(CLI::App &app) {
  CLI::App *elastic_fit = app.add_subcommand(
      "elastic_fit", "fit elastic tensor from ELAT JSON results");
  auto config = std::make_shared<EFSettings>();

  elastic_fit

      ->add_option("json_file", config->json_filename, "ELAT JSON results file")
      ->required()
      ->check(CLI::ExistingFile);


  elastic_fit->add_option("-o,--out", config->output_file,
                          "Output filename for elastic tensor");

  elastic_fit->add_option("-s,--scale", config->scale_factor,
                          "Factor to scale alpha by.");

  elastic_fit->add_option("-p,--potential", config->potential_type_str,
                          "Potential type to fit to. Either 'morse' or 'lj'.");

  elastic_fit->add_option("-g,--gulp-file", config->gulp_file,
                          "Write coarse grained crystal as a GULP input file.");

  elastic_fit->add_option(
      "--gulp-scale", config->gulp_scale,
      "Fraction of pair distance to set min and max cutoff for GULP.");

  elastic_fit->add_flag("--include-positive", config->include_positive,
                        "Whether or not to include positive "
                        "dimer energies when fitting the elastic tensor.");

  elastic_fit->add_flag("--max-to-zero", config->max_to_zero,
                        "Whether or not to shift all pair energies "
                        "such that the maximum is zero.");

  elastic_fit->add_option("--solver", config->solver_type_str,
                          "Linear solver type for elastic tensor calculation. "
                          "Options: 'lu', 'svd' (default), 'qr', 'ldlt'.");

  elastic_fit->add_option("-t,--temperature", config->temperature,
                          "Temperature in Kelvin for Uvib calculation.");

  elastic_fit->add_option(
      "--svd-threshold", config->svd_threshold,
      "SVD threshold for pseudoinverse (when using SVD solver).");

  elastic_fit->add_flag("--animate-phonons", config->animate_phonons,
                        "Animate the phonons and write them to XYZ files.");

  elastic_fit
      ->add_option("--mp-shrinking-factors", config->shrinking_factors_raw,
                   "Shrinking factors for Monkhorst-Pack for phonons (either 1 "
                   "or 3 numbers).")
      ->expected(1, 3);

  elastic_fit
      ->add_option(
          "--mp-shift", config->shift_raw,
          "Origin shift for Monkhorst-Pack for phonons (either 1 or 3 numbers)")
      ->expected(1, 3);

  elastic_fit->callback([config]() { run_elastic_fit_subcommand(*config); });

  return elastic_fit;
}

void run_elastic_fit_subcommand(const EFSettings &settings) {
  try {
    analyse_elat_results(settings);
  } catch (const std::exception &e) {
    occ::log::error("Error analysing ELAT results: {}", e.what());
    exit(1);
  }
}

} // namespace occ::main
