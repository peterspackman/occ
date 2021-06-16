#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include "catch.hpp"
#include <occ/core/util.h>
#include <occ/qm/basisset.h>
#include <fmt/ostream.h>
#include <occ/qm/scf.h>
#include <occ/qm/hf.h>

using occ::util::all_close;
using occ::Mat;
using occ::Mat3;

TEST_CASE("Basic rotations", "[mo_rotation]")
{
    Mat3 rot = Mat3::Identity(3, 3);
    auto drot = occ::gto::cartesian_gaussian_rotation_matrix<2>(rot);
    REQUIRE(all_close(drot, Mat::Identity(6, 6)));

    auto frot = occ::gto::cartesian_gaussian_rotation_matrix<3>(rot);
    REQUIRE(all_close(frot, Mat::Identity(10, 10)));
}


TEST_CASE("Water 3-21G basis set rotation", "[basis]")
{
    libint2::initialize();
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    occ::qm::BasisSet basis("3-21g", atoms);
    Mat3 rotation = Eigen::AngleAxisd(M_PI / 2, occ::Vec3{0, 1, 0}).toRotationMatrix();
    fmt::print("Rotation by:\n{}\n", rotation);

    auto hf = occ::hf::HartreeFock(atoms, basis);
    auto rot_basis = basis;

    rot_basis.rotate(rotation);
    auto rot_atoms = occ::qm::rotated_atoms(atoms, rotation);
    auto hf_rot = occ::hf::HartreeFock(rot_atoms, rot_basis);
    occ::scf::SCF<occ::hf::HartreeFock, occ::qm::SpinorbitalKind::Restricted> scf(hf);
    double e = scf.compute_scf_energy();
    occ::scf::SCF<occ::hf::HartreeFock, occ::qm::SpinorbitalKind::Restricted> scf_rot(hf_rot);
    double e_rot = scf_rot.compute_scf_energy();

    REQUIRE(e == Approx(e_rot));
}


occ::Mat interatomic_distances(const std::vector<libint2::Atom> & atoms)
{
    size_t natoms = atoms.size();
    occ::Mat dists(natoms, natoms);
    for (size_t i = 0; i < natoms; i++)
    {
        dists(i, i) = 0;
        for(size_t j = i + 1; j < natoms; j++)
        {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            dists(i, j) = sqrt(dx*dx + dy*dy + dz*dz);
            dists(j, i) = dists(i, j);
        }
    }
    return dists;
}


TEST_CASE("Water def2-tzvp MO rotation", "[basis]")
{
    libint2::initialize();
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    occ::qm::BasisSet basis("def2-tzvp", atoms);
    basis.set_pure(false);
    Mat3 rotation = - Mat3::Identity();
    fmt::print("Rotation by:\n{}\n", rotation);
    fmt::print("Distances before rotation:\n{}\n", interatomic_distances(atoms));
    auto hf = occ::hf::HartreeFock(atoms, basis);

    auto rot_atoms = occ::qm::rotated_atoms(atoms, rotation);
    occ::qm::BasisSet rot_basis = basis;
    rot_basis.rotate(rotation);
    auto shell2atom = rot_basis.shell2atom(rot_atoms);

    fmt::print("Distances after rotation:\n{}\n", interatomic_distances(rot_atoms));
    auto hf_rot = occ::hf::HartreeFock(rot_atoms, rot_basis);
    REQUIRE(hf.nuclear_repulsion_energy() == Approx(hf_rot.nuclear_repulsion_energy()));
    occ::scf::SCF<occ::hf::HartreeFock, occ::qm::SpinorbitalKind::Restricted> scf(hf);
    double e = scf.compute_scf_energy();
    Mat mos = scf.C;
    Mat C_occ = mos.leftCols(scf.n_occ);
    Mat D = C_occ * C_occ.transpose();
    Mat rot_mos = occ::qm::rotate_molecular_orbitals(rot_basis, rotation, mos);
    Mat rot_C_occ = rot_mos.leftCols(scf.n_occ);
    Mat rot_D = rot_C_occ * rot_C_occ.transpose();
    double e_en = occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(D, hf.compute_nuclear_attraction_matrix());
    double e_en_rot = occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(rot_D, hf_rot.compute_nuclear_attraction_matrix());
    fmt::print("E_en      {}\n", e_en);
    fmt::print("E_en'     {}\n", e_en_rot);
    REQUIRE(e_en == Approx(e_en_rot));
}
