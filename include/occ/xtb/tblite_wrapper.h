#pragma once
#include <array>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>

extern "C" {
#include "tblite/calculator.h"
#include "tblite/error.h"
#include "tblite/structure.h"
#include "tblite/container.h"
#include "tblite/version.h"
}

namespace occ::xtb {

class TbliteCalculator {
  public:
    enum class Method { GFN1, GFN2 };
    TbliteCalculator(const occ::core::Molecule &mol);
    TbliteCalculator(const occ::core::Dimer &dimer);
    TbliteCalculator(const occ::core::Molecule &mol, Method method);
    TbliteCalculator(const occ::core::Dimer &dimer, Method method);
    TbliteCalculator(const occ::crystal::Crystal &crystal);
    TbliteCalculator(const occ::crystal::Crystal &crystal, Method method);

    double single_point_energy();
    inline const auto &positions() const { return m_positions_bohr; }
    inline const auto &lattice_vectors() const { return m_lattice_vectors; }
    inline const auto &virial() const { return m_virial; }
    inline const auto &gradients() const { return m_gradients; }

    void set_charge(double c);
    void set_num_unpaired_electrons(int n);

    void set_accuracy(double accuracy);
    void set_max_iterations(int iterations);
    void set_temperature(double temp);
    void set_mixer_damping(double damping_factor);

    bool set_solvent(const std::string &);

    Vec charges() const;
    Mat bond_orders() const;
    inline int num_atoms() const { return m_atomic_numbers.rows(); }

    void update_structure(const Mat3N &positions);
    void update_structure(const Mat3N &positions, const Mat3 &lattice);

    occ::crystal::Crystal to_crystal() const;
    occ::core::Molecule to_molecule() const;

    ~TbliteCalculator();

  private:
    void initialize_context();
    void initialize_method();
    void initialize_structure();

    Mat3N m_positions_bohr;
    Mat3N m_gradients;
    IVec m_atomic_numbers;
    Method m_method{Method::GFN2};
    double m_charge{0};
    int m_num_unpaired_electrons{0};
    Mat3 m_lattice_vectors;
    Mat3 m_virial;
    std::array<bool, 3> m_periodic{false, false, false};
    tblite_error m_tb_error{nullptr};
    tblite_context m_tb_ctx{nullptr};
    tblite_result m_tb_result{nullptr};
    tblite_structure m_tb_structure{nullptr};
    tblite_calculator m_tb_calc{nullptr};
    tblite_container m_solvent_container{nullptr};
};

std::string tblite_version();

} // namespace occ::xtb
