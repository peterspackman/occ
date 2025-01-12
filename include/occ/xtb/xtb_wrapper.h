#pragma once
#include <array>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>

namespace occ::xtb {

class XTBCalculator {
public:
  enum class Method { GFN1, GFN2 };
  XTBCalculator(const occ::core::Molecule &mol);
  XTBCalculator(const occ::core::Dimer &dimer);
  XTBCalculator(const occ::core::Molecule &mol, Method method);
  XTBCalculator(const occ::core::Dimer &dimer, Method method);
  XTBCalculator(const occ::crystal::Crystal &crystal);
  XTBCalculator(const occ::crystal::Crystal &crystal, Method method);

  double single_point_energy();
  inline const auto &positions() const { return m_positions_bohr; }
  inline const auto &lattice_vectors() const { return m_lattice_vectors; }
  inline const auto &virial() const { return m_virial; }
  inline const auto &gradients() const { return m_gradients; }
  inline const auto &partial_charges() const { return m_partial_charges; }

  void set_charge(double c);
  void set_num_unpaired_electrons(int n);

  void set_accuracy(double accuracy);
  void set_max_iterations(int iterations);
  void set_temperature(double temp);
  void set_mixer_damping(double damping_factor);

  Vec charges() const;
  Mat bond_orders() const;
  inline int num_atoms() const { return m_atomic_numbers.rows(); }

  void update_structure(const Mat3N &positions);
  void update_structure(const Mat3N &positions, const Mat3 &lattice);
  void set_solvent(const std::string &solvent_name);
  void set_solvation_model(const std::string &);

  occ::crystal::Crystal to_crystal() const;
  occ::core::Molecule to_molecule() const;

private:
  void initialize_structure();
  void write_input_file(const std::string &);
  void read_json_contents(const std::string &);
  void read_engrad_contents(const std::string &);
  int gfn_method() const;

  Mat3N m_positions_bohr;
  Mat3N m_gradients;
  IVec m_atomic_numbers;
  Method m_method{Method::GFN2};
  double m_charge{0};
  int m_num_unpaired_electrons{0};
  double m_energy{0.0};
  Mat3 m_lattice_vectors;
  Mat3 m_virial;
  Vec m_partial_charges;
  std::array<bool, 3> m_periodic{false, false, false};
  std::string m_xtb_stdout;
  std::string m_xtb_stderr;
  double m_accuracy{0.01};
  int m_max_iterations{100};
  double m_temperature{0.0};
  double m_damping_factor{1.0};
  std::string m_solvent{""};
  std::string m_solvation_model{"cpcmx"};
  std::string m_xtb_executable_path{"xtb"};
};

} // namespace occ::xtb
