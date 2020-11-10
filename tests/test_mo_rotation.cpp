#include "linear_algebra.h"
#include "gto.h"
#include "catch.hpp"
#include "util.h"
#include "basisset.h"
#include <fmt/ostream.h>
#include "scf.h"
#include "hf.h"

using tonto::util::all_close;

TEST_CASE("Basic rotations", "[mo_rotation]")
{
    tonto::Mat3 rot = tonto::Mat3::Identity(3, 3);
    auto drot = tonto::gto::cartesian_gaussian_rotation_matrix<2>(rot);
    REQUIRE(all_close(drot, tonto::MatRM::Identity(6, 6)));

    auto frot = tonto::gto::cartesian_gaussian_rotation_matrix<3>(rot);
    REQUIRE(all_close(frot, tonto::MatRM::Identity(10, 10)));
}


TEST_CASE("Water 3-21G", "[basis]")
{
    libint2::initialize();
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    tonto::qm::BasisSet basis("3-21g", atoms);
    tonto::Mat3 rotation = Eigen::AngleAxisd(M_PI / 2, tonto::Vec3{0, 1, 0}).toRotationMatrix();
    fmt::print("Rotation by: {}\n", rotation);

    auto hf = tonto::hf::HartreeFock(atoms, basis);
    auto rot_basis = basis;

    rot_basis.rotate(rotation);
    auto rot_atoms = tonto::qm::rotated_atoms(atoms, rotation);
    auto hf_rot = tonto::hf::HartreeFock(rot_atoms, rot_basis);
    tonto::scf::SCF<tonto::hf::HartreeFock, tonto::qm::SpinorbitalKind::Restricted> scf(hf);
    double e = scf.compute_scf_energy();
    tonto::scf::SCF<tonto::hf::HartreeFock, tonto::qm::SpinorbitalKind::Restricted> scf_rot(hf_rot);
    double e_rot = scf.compute_scf_energy();

    REQUIRE(e == Approx(e_rot));
}


TEST_CASE("Water def2-tzvp MO rotation", "[basis]")
{
    libint2::initialize();
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    tonto::qm::BasisSet basis("def2-tzvp", atoms);
    tonto::Mat3 rotation = Eigen::AngleAxisd(M_PI / 8, tonto::Vec3{0, 0, 1}).toRotationMatrix();
    fmt::print("Rotation by: {}\n", rotation);

    auto hf = tonto::hf::HartreeFock(atoms, basis);
    auto rot_basis = basis;

    rot_basis.rotate(rotation);
    auto rot_atoms = tonto::qm::rotated_atoms(atoms, rotation);
    auto hf_rot = tonto::hf::HartreeFock(rot_atoms, rot_basis);
    tonto::scf::SCF<tonto::hf::HartreeFock, tonto::qm::SpinorbitalKind::Restricted> scf(hf);
    double e = scf.compute_scf_energy();
    tonto::MatRM mos = scf.C;
    tonto::MatRM D = scf.D;
    tonto::MatRM rot_mos = tonto::qm::rotate_molecular_orbitals(rot_basis, rotation, mos);
    tonto::MatRM rot_C_occ = rot_mos.leftCols(scf.n_occ);
    tonto::MatRM rot_D = rot_C_occ * rot_C_occ.transpose();
    double e_en = tonto::qm::expectation<tonto::qm::SpinorbitalKind::Restricted>(D, hf.compute_kinetic_matrix());
    double e_en_rot = tonto::qm::expectation<tonto::qm::SpinorbitalKind::Restricted>(rot_D, hf_rot.compute_kinetic_matrix());
    fmt::print("E_k    : {}\n", e_en);
    fmt::print("E_k_rot: {}\n", e_en_rot);
    REQUIRE(e_en == Approx(e_en_rot));
}
