#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/partitioning.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/oniom.h>
#include <vector>

using occ::Mat;
using occ::Mat3;
using occ::qm::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::util::all_close;

// Basis

TEST_CASE("AOBasis set pure spherical") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    auto basis = occ::qm::AOBasis::load(atoms, "6-31G");
    basis.set_pure(true);
    for (const auto &sh : basis.shells()) {
        std::cout << sh << '\n';
    }
}

TEST_CASE("AOBasis load") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "6-31G");
    for (const auto &sh : basis.shells()) {
        std::cout << sh << '\n';
    }
}

// Density Fitting

TEST_CASE("Density Fitting H2O/6-31G J/K matrices") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.39839733}};
    auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
    basis.set_pure(false);
    auto hf = HartreeFock(basis);
    hf.set_density_fitting_basis("def2-universal-jkfit");

    occ::qm::MolecularOrbitals mo;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;

    mo.C = occ::Mat(2, 2);
    mo.C << 0.54884228, 1.21245192, 0.54884228, -1.21245192;

    mo.Cocc = mo.C.leftCols(1);
    mo.D = mo.Cocc * mo.Cocc.transpose();
    fmt::print("D:\n{}\n", mo.D);

    occ::Mat Fexact(2, 2);
    Fexact << 1.50976125, 0.7301775, 0.7301775, 1.50976125;

    occ::Mat Jexact(2, 2);
    Jexact << 1.34575531, 0.89426314, 0.89426314, 1.34575531;

    occ::Mat Kexact(2, 2);
    Kexact << 1.18164378, 1.05837468, 1.05837468, 1.18164378;

    occ::qm::JKPair jk_approx = hf.compute_JK(mo);
    occ::Mat F = 2 * (jk_approx.J - jk_approx.K);
    fmt::print("Fexact\n{}\n", Fexact);
    fmt::print("Fapprox\n{}\n", F);

    fmt::print("Jexact\n{}\n", Jexact);
    fmt::print("Japprox\n{}\n", jk_approx.J);

    fmt::print("Kexact\n{}\n", Kexact);
    fmt::print("Kapprox\n{}\n", 2 * jk_approx.K);
}

TEST_CASE("Electric Field evaluation H2/STO-3G") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.398397}};
    auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
    Mat D(2, 2);
    D.setConstant(0.301228);
    auto grid_pts = occ::Mat3N(3, 4);
    grid_pts << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0;
    HartreeFock hf(basis);

    occ::qm::MolecularOrbitals mo;
    mo.D = D;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;
    occ::Vec expected_esp = occ::Vec(4);
    expected_esp << -1.37628, -1.37628, -1.95486, -1.45387;

    auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
    fmt::print("Grid points\n{}\n", grid_pts);
    fmt::print("Nuclear E field values:\n{}\n", field_values);

    auto esp = hf.electronic_electric_potential_contribution(mo, grid_pts);
    fmt::print("ESP:\n{}\n", esp);
    REQUIRE(all_close(esp, expected_esp, 1e-5, 1e-5));
    occ::Mat expected_efield(field_values.rows(), field_values.cols());
    occ::Mat efield;

    expected_efield << -0.592642, 0.0, 0.0, 0.0, 0.0, -0.592642, 0.0, -0.652486,
        0.26967, 0.26967, -0.0880444, -0.116878;

    double delta = 1e-8;
    occ::Mat3N efield_fd(field_values.rows(), field_values.cols());
    for (size_t i = 0; i < 3; i++) {
        auto grid_pts_d = grid_pts;
        grid_pts_d.row(i).array() += delta;
        auto esp_d =
            hf.electronic_electric_potential_contribution(mo, grid_pts_d);
        efield_fd.row(i) = -(esp_d - esp) / delta;
    }
    REQUIRE(all_close(efield_fd, expected_efield, 1e-5, 1e-5));
    fmt::print("Electric field FD:\n{}\n", efield_fd);
}

// MO Rotation

TEST_CASE("Cartesian gaussian basic rotation matrices", "[mo_rotation]") {
    Mat3 rot = Mat3::Identity(3, 3);
    auto drot = occ::gto::cartesian_gaussian_rotation_matrices(2, rot)[2];
    REQUIRE(all_close(drot, Mat::Identity(6, 6)));

    auto frot = occ::gto::cartesian_gaussian_rotation_matrices(3, rot)[3];
    REQUIRE(all_close(frot, Mat::Identity(10, 10)));
}

TEST_CASE("Water 3-21G basis set rotation energy consistency", "[basis]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    auto basis = occ::qm::AOBasis::load(atoms, "3-21G");
    basis.set_pure(false);
    fmt::print("basis.size() {}\n", basis.size());
    Mat3 rotation =
        Eigen::AngleAxisd(M_PI / 2, occ::Vec3{0, 1, 0}).toRotationMatrix();
    fmt::print("Rotation by:\n{}\n", rotation);

    auto hf = HartreeFock(basis);
    occ::scf::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();

    occ::qm::AOBasis rot_basis = basis;
    rot_basis.rotate(rotation);
    auto rot_atoms = rot_basis.atoms();
    fmt::print("rot_basis.size() {}\n", rot_basis.size());
    auto hf_rot = HartreeFock(rot_basis);
    occ::scf::SCF<HartreeFock> scf_rot(hf_rot);
    double e_rot = scf_rot.compute_scf_energy();

    REQUIRE(e == Catch::Approx(e_rot));
}

occ::Mat interatomic_distances(const std::vector<occ::core::Atom> &atoms) {
    size_t natoms = atoms.size();
    occ::Mat dists(natoms, natoms);
    for (size_t i = 0; i < natoms; i++) {
        dists(i, i) = 0;
        for (size_t j = i + 1; j < natoms; j++) {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            dists(i, j) = sqrt(dx * dx + dy * dy + dz * dz);
            dists(j, i) = dists(i, j);
        }
    }
    return dists;
}

TEST_CASE("Water def2-tzvp MO rotation energy consistency", "[basis]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    auto basis = occ::qm::AOBasis::load(atoms, "def2-tzvp");
    basis.set_pure(true);
    Eigen::Quaterniond r(Eigen::AngleAxisd(
        0.423, Eigen::Vector3d(0.234, -0.642, 0.829).normalized()));
    Mat3 rotation = r.toRotationMatrix();

    fmt::print("Rotation by:\n{}\n", rotation);
    fmt::print("Distances before rotation:\n{}\n",
               interatomic_distances(atoms));
    auto hf = HartreeFock(basis);

    auto rot_basis = basis;
    rot_basis.rotate(rotation);
    auto rot_atoms = rot_basis.atoms();
    auto shell2atom = rot_basis.shell_to_atom();

    fmt::print("Distances after rotation:\n{}\n",
               interatomic_distances(rot_atoms));
    auto hf_rot = HartreeFock(rot_basis);
    REQUIRE(hf.nuclear_repulsion_energy() ==
            Catch::Approx(hf_rot.nuclear_repulsion_energy()));
    occ::scf::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mos = scf.mo;
    Mat C_occ = mos.C.leftCols(scf.n_occ);
    Mat D = C_occ * C_occ.transpose();

    mos.rotate(rot_basis, rotation);
    Mat rot_C_occ = mos.C.leftCols(scf.n_occ);
    Mat rot_D = rot_C_occ * rot_C_occ.transpose();

    double e_en = occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(
        D, hf.compute_nuclear_attraction_matrix());
    double e_en_rot =
        occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(
            rot_D, hf_rot.compute_nuclear_attraction_matrix());
    fmt::print("E_en      {}\n", e_en);
    fmt::print("E_en'     {}\n", e_en_rot);
    REQUIRE(e_en == Catch::Approx(e_en_rot));
}

// SCF

TEST_CASE("Water RHF SCF energy", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    SECTION("STO-3G") {
        auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
    }

    SECTION("3-21G") {
        auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
    }
}

TEST_CASE("Water UHF SCF energy", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    SECTION("STO-3G") {
        auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf, SpinorbitalKind::Unrestricted);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
    }

    SECTION("3-21G") {
        auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf, SpinorbitalKind::Unrestricted);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
    }
}

TEST_CASE("Water GHF SCF energy", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    SECTION("STO-3G") {
        auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf, SpinorbitalKind::General);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
    }

    SECTION("3-21G") {
        auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf, SpinorbitalKind::General);
        scf.convergence_settings.energy_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
    }
}

occ::Mat3N atomic_gradients(const occ::Mat &D, const occ::qm::MatTriple &grad,
                            const occ::qm::AOBasis &basis) {
    const auto &bf_to_atom = basis.bf_to_atom();
    occ::Mat3N result(3, basis.atoms().size());
    occ::Mat weighted_grad_x = D.cwiseProduct(grad.x);
    occ::Mat weighted_grad_y = D.cwiseProduct(grad.y);
    occ::Mat weighted_grad_z = D.cwiseProduct(grad.z);

    for (int bf1 = 0; bf1 < basis.nbf(); bf1++) {
        int atom1 = bf_to_atom[bf1];

        for (int bf2 = 0; bf2 < basis.nbf(); bf2++) {
            int atom2 = bf_to_atom[bf2];

            // Accumulate gradient contributions
            result(atom1, 0) += weighted_grad_x(bf1, bf2);
            result(atom1, 1) += weighted_grad_y(bf1, bf2);
            result(atom1, 2) += weighted_grad_z(bf1, bf2);

            if (atom1 != atom2) {
                result(atom2, 0) += weighted_grad_x(bf1, bf2);
                result(atom2, 1) += weighted_grad_y(bf1, bf2);
                result(atom2, 2) += weighted_grad_z(bf1, bf2);
            }
        }
    }
    return result;
}


TEST_CASE("Integral gradients", "[integrals]") {

    /*
    std::vector<occ::core::Atom> atoms{
        {1, 0.0, 0.0, 0.0},
	{1, 0.0, 0.0, 1.4 * occ::units::ANGSTROM_TO_BOHR}
    };
    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
    */
   
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");

    occ::Mat D(obs.nbf(), obs.nbf());
    D <<
    2.106529, -0.447611, 0.057951, 0.091761, -0.002396, -0.027622, -0.027249,
   -0.447611, 1.974382, -0.328030, -0.521593, 0.013615, -0.038544, -0.037120,
    0.057951, -0.328030, 0.877559, 0.221255, -0.002740, -0.203698, 0.711984,
    0.091761, -0.521593, 0.221255, 1.089979, 0.021845, 0.689851, 0.111189,
   -0.002396, 0.013615, -0.002740, 0.021845, 1.999469, -0.016473, -0.004447,
   -0.027622, -0.038544, -0.203698, 0.689851, -0.016473, 0.603384, -0.189923,
   -0.027249, -0.037120, 0.711984, 0.111189, -0.004447, -0.189923, 0.606432;

    
    //occ::Mat D(obs.nbf(), obs.nbf());
    //D.setConstant(0.77178414);
    occ::qm::MolecularOrbitals mo;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;
    mo.D = D * 0.5;
    std::cout << "D\n" << std::setprecision(4) << mo.D << '\n';

    occ::Mat Xref(obs.nbf(), obs.nbf()), Yref(obs.nbf(), obs.nbf()), Zref(obs.nbf(), obs.nbf());

    Xref <<
     -0.03898, -0.01702, 13.27512,  0.01153, -0.00015, -0.27938,  0.80778,
      0.00038,  0.02486,  4.61696, -0.00033, -0.00000, -0.67535,  2.04809,
    -15.50871, -7.76594,  0.06736,  0.00329, -0.00019, -2.04167,  0.14166,
      0.00312,  0.00478,  0.02816, -0.00443,  0.00035, -0.76461,  0.25917,
     -0.00004, -0.00007, -0.00084,  0.00035,  0.01086,  0.01816, -0.01219,
      0.31666,  0.94311,  1.04257,  0.92737, -0.02203,  0.31173,  0.91043,
     -0.92953, -2.75624, -1.52403, -0.30732,  0.01446, -1.08941, -0.86829;

    Yref <<
     -0.06087, -0.02660,  0.01153, 13.28648,  0.00109,  0.77487,  0.07826,
      0.00060,  0.03938, -0.00033,  4.61698, -0.00008,  1.96012,  0.22767,
      0.00312,  0.00478,  0.02816, -0.00443,  0.00035, -0.76461,  0.25917,
    -15.50563, -7.76067, -0.01976,  0.07139, -0.00086, -0.11750, -2.24876,
      0.00029,  0.00037,  0.00035, -0.00150,  0.01734, -0.05199, -0.00127,
     -0.89387, -2.62418,  0.90500, -1.22562,  0.06155, -0.81827, -0.76975,
     -0.09524, -0.26983, -0.28513,  1.31597,  0.00142,  0.48830, -0.06852;

    Zref <<
      0.00159,  0.00069, -0.00015,  0.00109, 13.33186, -0.01838, -0.00387,
     -0.00002, -0.00103, -0.00000, -0.00008,  4.61356, -0.04657, -0.01053,
     -0.00004, -0.00007, -0.00084,  0.00035,  0.01086,  0.01816, -0.01219,
      0.00029,  0.00037,  0.00035, -0.00150,  0.01734, -0.05199, -0.00127,
    -15.49334, -7.74518, -0.00447, -0.00753, -0.00071, -2.30545, -2.27486,
      0.02122,  0.06226, -0.02148,  0.06149,  1.36441,  0.01937,  0.01674,
      0.00458,  0.01327,  0.01392,  0.00148,  1.34587, -0.00939,  0.00377;

    occ::qm::IntegralEngine engine(obs);
    HartreeFock hf(obs);
    auto [J, K] = hf.compute_JK(mo);
    auto [grad, grad_k] = hf.compute_JK_gradient(mo);
    /*
    std::cout << "J:\n" << std::setprecision(4) << J << '\n';
    std::cout << "X:\n" << std::setprecision(4) << grad.x << '\n';
    std::cout << "Y:\n" << std::setprecision(4) << grad.y << '\n';
    std::cout << "Z:\n" << std::setprecision(4) << grad.z << '\n';
    */
    REQUIRE(all_close(grad.x, Xref, 1e-5, 1e-5));
    REQUIRE(all_close(grad.y, Yref, 1e-5, 1e-5));
    REQUIRE(all_close(grad.z, Zref, 1e-5, 1e-5));

    std::cout << "K:\n" << std::setprecision(4) << K << '\n';
    std::cout << "KX:\n" << std::setprecision(4) << grad_k.x << '\n';
    std::cout << "KY:\n" << std::setprecision(4) << grad_k.y << '\n';
    std::cout << "KZ:\n" << std::setprecision(4) << grad_k.z << '\n';

    //auto d = atomic_gradients(D, grad, obs);
    //fmt::print("Atom gradients:\n{}\n", d);

    /*
    grad = hf.compute_kinetic_gradient();
    fmt::print("kinetic\n");
    std::cout << "X:\n" << std::setprecision(2) << grad.x << '\n';
    std::cout << "Y:\n" << std::setprecision(2) << grad.y << '\n';
    std::cout << "Z:\n" << std::setprecision(2) << grad.z << '\n';

    d = atomic_gradients(D, grad, obs);
    fmt::print("Atom gradients:\n{}\n", d);
    */
}

TEST_CASE("Oniom ethane", "[oniom]") {
    using occ::core::Atom;
    using occ::scf::SCF;
    std::vector<Atom> atoms {
	{1, 2.239513249136882, -0.007369927999015981, 1.8661035638534056},
	{6, 1.4203174061693364, -0.042518815378938354, -0.039495255174213845},
	{1, 2.205120251808141, 1.5741410315846955, -1.075820515343538},
	{1, 2.1079883802313657, -1.7629245718671818, -0.9722635783317236},
	{6, -1.4203174061693364, 0.042518815378938354, 0.039495255174213845},
	{1, -2.205120251808141, -1.5748969216358768, 1.0746866802667663},
	{1, -2.108366325256956, 1.762357654328796, 0.9733974134084954},
	{1, -2.2393242766240866, 0.00831479056299239, -1.8661035638534056}
    };

    Atom artificial_h1 = atoms[1];
    artificial_h1.atomic_number = 1;
    
    Atom artificial_h2 = atoms[4];
    artificial_h2.atomic_number = 1;

    std::vector<Atom> methane1{
	atoms[0], atoms[1], atoms[2], atoms[3], artificial_h2
    };

    std::vector<Atom> methane2 {
	artificial_h1, atoms[4], atoms[5], atoms[6], atoms[7]
    };

    HartreeFock system_low(occ::qm::AOBasis::load(atoms, "STO-3G"));
    HartreeFock methane1_low(occ::qm::AOBasis::load(methane1, "STO-3G"));
    HartreeFock methane2_low(occ::qm::AOBasis::load(methane2, "STO-3G"));

    HartreeFock methane1_high(occ::qm::AOBasis::load(methane1, "def2-tzvp"));
    HartreeFock methane2_high(occ::qm::AOBasis::load(methane2, "def2-tzvp"));

    using Proc = SCF<HartreeFock>;
    occ::qm::Oniom<Proc, Proc> oniom{
	{SCF(methane1_high), SCF(methane2_high)},
	{SCF(methane1_low), SCF(methane2_low)},
	SCF(system_low)
    };

    fmt::print("Total energy: {}\n", oniom.compute_scf_energy());
}

TEST_CASE("Mulliken partition", "[partitioning]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    auto obs = occ::qm::AOBasis::load(atoms, "3-21G");

    auto hf = HartreeFock(obs);
    occ::scf::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();

    auto wfn = scf.wavefunction();

    auto charges = wfn.mulliken_charges();
    fmt::print("Charges:\n{}\n", charges);
    occ::Vec expected(3);
    expected << -0.724463, 0.363043, 0.361419;
    fmt::print("Expected:\n{}\n", expected);
    REQUIRE(all_close(expected, charges, 1e-5, 1e-5));

    auto energies = occ::qm::mulliken_partition(obs, wfn.mo, wfn.V);
    double total = occ::qm::expectation(wfn.mo.kind, wfn.mo.D, wfn.V);
    fmt::print("Partitioned energy\n{}\n", energies);
    REQUIRE(energies.sum() == Catch::Approx(total));
}
