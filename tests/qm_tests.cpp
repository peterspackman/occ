#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
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
using Catch::Matchers::WithinAbs;

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


TEST_CASE("Smearing functions", "[smearing]") {
    occ::qm::MolecularOrbitals mo;
    mo.energies = occ::Vec(43);

    mo.energies <<
        -20.7759836,   -1.75714997,  -0.6755331,   -0.47838917,  -0.34397101,
         0.39919219,    0.53017716,   0.84601736,   0.91026033,   0.93422352,
         0.93541066,    1.07197252,   1.1152775,    1.74421062,   1.78568455,
         1.87179641,    2.12778396,   2.22487896,   2.41855156,   2.63997607,
         2.69396251,    2.72429482,   2.94010404,   2.99844011,   3.27233171,
         3.29230622,    3.30213655,   3.40824971,   4.35313802,   4.62532886,
         5.78639639,    5.89978906,   6.07407985,   6.11565547,   6.16161542,
         6.80026627,    6.96537841,   7.04215449,   7.10247792,   7.17168709,
         7.63716226,    7.75364544,  45.00621777;



    SECTION("Fermi") {
        occ::qm::OrbitalSmearing smearing;
        smearing.sigma = 0.095;
        smearing.mu = -0.06;
        smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;

        occ::Vec res = smearing.calculate_fermi_occupations(mo);

        occ::Vec expected(43);
        expected <<
           1.00000000e+00, 9.99999983e-01, 9.98467461e-01, 9.87920549e-01,
           9.52082391e-01, 7.89497886e-03, 2.00042906e-03, 7.21259277e-05,
           3.66791050e-05, 2.85019162e-05, 2.81479763e-05, 6.68591892e-06,
           4.23830834e-06, 5.64954547e-09, 3.65102304e-09, 1.47486548e-09,
           9.96552046e-11, 3.58614765e-11, 4.66927912e-12, 4.53944824e-13,
           2.57159713e-13, 1.86869380e-13, 1.92735534e-14, 1.04298300e-14,
           5.83681688e-16, 4.73001131e-16, 4.26503520e-16, 1.39580290e-16,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00;

        REQUIRE(all_close(res, expected, 1e-5, 1e-5));

	// override occupations to be exactly the right thing
	mo.occupation = expected;
	smearing.entropy = smearing.calculate_entropy(mo);
	REQUIRE(smearing.ec_entropy() == Catch::Approx(-0.06301068258742577));
    }
}

TEST_CASE("H2 smearing", "[smearing]") {
    occ::Vec expected_energies(16), expected_correlations(16), separations(16);

    expected_energies <<
	-1.05401645, -1.10204847, -1.10241449, -1.09833917, -1.0429699 ,
        -0.96723086, -0.89171816, -0.82552549, -0.77490759, -0.74204562,
        -0.72385175, -0.71057038, -0.7076913 , -0.70707591, -0.70694503,
        -0.70691935;

    separations <<
	1.0, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5, 
	4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

    expected_correlations <<
	-0.02090322, -0.02916732, -0.03247401, -0.03609615, -0.0596489 ,
        -0.09272139, -0.13295573, -0.17469005, -0.21056677, -0.2356074 ,
        -0.25006385, -0.26093526, -0.2633199 , -0.2638024 , -0.26389139,
        -0.26390591;

    occ::Vec energies(16);

    for(int i = 0; i < separations.rows(); i++) {
	std::vector<occ::core::Atom> atoms{
	    {1, 0.0, 0.0, 0.0},
	    {1, separations(i), 0.0, 0.0}
	};
        auto obs = occ::qm::AOBasis::load(atoms, "cc-pvdz");
	obs.set_pure(true);
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock> scf(hf);

        scf.mo.smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;
        scf.mo.smearing.sigma = 0.095;

        energies(i) = scf.compute_scf_energy();

        REQUIRE_THAT(energies(i), WithinAbs(expected_energies(i), 1e-5));
        REQUIRE_THAT(scf.mo.smearing.ec_entropy(), 
		     WithinAbs(expected_correlations(i), 1e-5));
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
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");

    occ::Mat D(obs.nbf(), obs.nbf());
    D.setConstant(0.301228);

    occ::qm::IntegralEngine engine(obs);
    HartreeFock hf(obs);
    auto grad = hf.compute_nuclear_attraction_gradient();
    fmt::print("Nuclear\n");
    fmt::print("X:\n{}\n", grad.x);
    fmt::print("Y:\n{}\n", grad.y);
    fmt::print("Z:\n{}\n", grad.z);

    auto d = atomic_gradients(D, grad, obs);
    fmt::print("Atom gradients:\n{}\n", d);

    grad = hf.compute_kinetic_gradient();
    fmt::print("kinetic\n");
    fmt::print("X:\n{}\n", grad.x);
    fmt::print("Y:\n{}\n", grad.y);
    fmt::print("Z:\n{}\n", grad.z);
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
