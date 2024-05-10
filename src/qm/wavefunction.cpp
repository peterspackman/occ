#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/element.h>
#include <occ/core/gensqrtinv.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/conversion.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/io/wavefunction_json.h>
#include <occ/io/wavefunction_json.h>
#include <occ/qm/hf.h>
#include <occ/qm/merge.h>
#include <occ/qm/orb.h>
#include <occ/qm/partitioning.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>
#include <filesystem>
#include <occ/gto/density.h>

namespace fs = std::filesystem;

namespace occ::qm {

using occ::io::FchkReader;
using occ::qm::merge_atoms;
using occ::qm::merge_basis_sets;
using occ::qm::merge_molecular_orbitals;

Wavefunction::Wavefunction(const FchkReader &fchk)
    : num_electrons(fchk.num_electrons()),
      basis(fchk.basis_set()), nbf(basis.nbf()), atoms(fchk.atoms()) {
    energy.total = fchk.scf_energy();
    set_molecular_orbitals(fchk);
    compute_density_matrix();
}

Wavefunction::Wavefunction(const OrcaJSONReader &json)
    : num_electrons(json.num_electrons()),
      basis(json.basis_set()), nbf(basis.nbf()), atoms(json.atoms()) {
    energy.total = json.scf_energy();
    size_t rows, cols;

    mo.kind = json.spinorbital_kind();
    mo.n_alpha = json.num_alpha();
    mo.n_beta = json.num_beta();
    mo.n_ao = nbf;

    if (mo.kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from Orca json unsupported for General spinorbitals");
    } else if (mo.kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = json.alpha_mo_coefficients();
        block::b(mo.C) = json.beta_mo_coefficients();
        block::a(mo.energies) = json.alpha_mo_energies();
        block::b(mo.energies) = json.beta_mo_energies();
    } else {
        mo.C = json.alpha_mo_coefficients();
        mo.energies = json.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const MoldenReader &molden)
    : num_electrons(molden.num_electrons()), basis(molden.basis_set()),
      nbf(molden.nbf()), atoms(molden.atoms()) {
    size_t rows, cols;
    nbf = basis.nbf();
    mo.kind = molden.spinorbital_kind();
    mo.n_alpha = molden.num_alpha();
    mo.n_beta = molden.num_beta();
    mo.n_ao = nbf;

    if (mo.kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (mo.kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = molden.alpha_mo_coefficients();
        block::b(mo.C) = molden.beta_mo_coefficients();
        block::a(mo.energies) = molden.alpha_mo_energies();
        block::b(mo.energies) = molden.beta_mo_energies();
        block::a(mo.C) = molden.convert_mo_coefficients_from_molden_convention(
            basis, block::a(mo.C));
        block::b(mo.C) = molden.convert_mo_coefficients_from_molden_convention(
            basis, block::b(mo.C));
    } else {
        mo.C = molden.alpha_mo_coefficients();
        mo.C =
            molden.convert_mo_coefficients_from_molden_convention(basis, mo.C);
        mo.energies = molden.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b)
    : num_electrons(wfn_a.num_electrons + wfn_b.num_electrons),
      basis(merge_basis_sets(wfn_a.basis, wfn_b.basis)), nbf(basis.nbf()),
      atoms(merge_atoms(wfn_a.atoms, wfn_b.atoms)) {

    mo = MolecularOrbitals(wfn_a.mo, wfn_b.mo);
    update_occupied_orbitals();

    if (wfn_a.have_xdm_parameters && wfn_b.have_xdm_parameters) {
        occ::log::debug("Merging XDM parameters");
        have_xdm_parameters = true;
        occ::log::debug("Merging XDM polarizabilities");
        xdm_polarizabilities = Vec(wfn_a.xdm_polarizabilities.rows() +
                                   wfn_b.xdm_polarizabilities.rows());
        xdm_polarizabilities << wfn_a.xdm_polarizabilities,
            wfn_b.xdm_polarizabilities;

        occ::log::debug("Merging XDM moments");
        xdm_moments = Mat(wfn_a.xdm_moments.rows(),
                          wfn_a.xdm_moments.cols() + wfn_b.xdm_moments.cols());
        xdm_moments.leftCols(wfn_a.xdm_moments.cols()) = wfn_a.xdm_moments;
        xdm_moments.rightCols(wfn_b.xdm_moments.cols()) = wfn_b.xdm_moments;

        occ::log::debug("Merging XDM volumes");
        xdm_volumes = Vec(wfn_a.xdm_volumes.rows() + wfn_b.xdm_volumes.rows());
        xdm_volumes << wfn_a.xdm_volumes, wfn_b.xdm_volumes;
        occ::log::debug("Merging XDM free volumes");
        xdm_free_volumes =
            Vec(wfn_a.xdm_free_volumes.rows() + wfn_b.xdm_free_volumes.rows());
        xdm_free_volumes << wfn_a.xdm_free_volumes, wfn_b.xdm_free_volumes;
        occ::log::debug("Finished merging");
    }
}

void Wavefunction::update_occupied_orbitals() {
    mo.update_occupied_orbitals();
}

void Wavefunction::set_molecular_orbitals(const FchkReader &fchk) {
    size_t rows, cols;
    nbf = basis.nbf();
    mo.kind = fchk.spinorbital_kind();
    mo.n_alpha = fchk.num_alpha();
    mo.n_beta = fchk.num_beta();
    mo.n_ao = nbf;
    if (mo.kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (mo.kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = fchk.alpha_mo_coefficients();
        block::b(mo.C) = fchk.beta_mo_coefficients();
        block::a(mo.energies) = fchk.alpha_mo_energies();
        block::b(mo.energies) = fchk.beta_mo_energies();
    } else {
        mo.C = fchk.alpha_mo_coefficients();
        mo.energies = fchk.alpha_mo_energies();
    }
    mo = occ::io::conversion::orb::from_gaussian_order(basis, mo);
    update_occupied_orbitals();
}

void Wavefunction::compute_density_matrix() {
    mo.update_density_matrix();
}

void Wavefunction::symmetric_orthonormalize_molecular_orbitals(
    const Mat &overlap) {
    mo = mo.symmetrically_orthonormalized(overlap);
    // TODO remove this and make sure it's all performed in the mo method
    update_occupied_orbitals();
}

void Wavefunction::apply_transformation(const occ::Mat3 &rot,
                                        const occ::Vec3 &trans) {
    apply_rotation(rot);
    apply_translation(trans);
}
void Wavefunction::apply_translation(const occ::Vec3 &trans) {
    basis.translate(trans);
    occ::core::translate_atoms(atoms, trans);
}

void Wavefunction::apply_rotation(const occ::Mat3 &rot) {
    // check if identity matrix, otherwise do the rotation and
    // update the orbitals
    if (!rot.isIdentity(1e-6)) {
        mo.rotate(basis, rot);
        basis.rotate(rot);
        occ::core::rotate_atoms(atoms, rot);
        update_occupied_orbitals();
        compute_density_matrix();
    } else {
        occ::log::debug("Skipping rotation by identity matrix");
    }
}

occ::Mat3N Wavefunction::positions() const {
    occ::Mat3N pos(3, atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        pos(0, i) = atoms[i].x;
        pos(1, i) = atoms[i].y;
        pos(2, i) = atoms[i].z;
    }
    return pos;
}

occ::IVec Wavefunction::atomic_numbers() const {
    occ::IVec nums(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        nums(i) = atoms[i].atomic_number;
    }
    return nums;
}

void Wavefunction::save(FchkWriter &fchk) {
    bool have_ecps = basis.have_ecps();
    occ::timing::start(occ::timing::category::io);
    fchk.set_scalar("Number of atoms", atoms.size());
    fchk.set_scalar("Charge", charge());
    fchk.set_scalar("Multiplicity", multiplicity());
    fchk.set_scalar("SCF Energy", energy.total);
    fchk.set_scalar("Number of electrons", num_electrons);
    fchk.set_scalar("Number of alpha electrons", mo.n_alpha);
    fchk.set_scalar("Number of beta electrons", mo.n_beta);
    fchk.set_scalar("Number of basis functions", nbf);
    fchk.set_scalar("Number of independent functions", nbf);
    fchk.set_scalar("Number of point charges in /Mol/", 0);
    fchk.set_scalar("Number of translation vectors", 0);
    fchk.set_scalar("Force Field", 0);

    // nuclear charges
    occ::IVec nums = atomic_numbers();
    occ::Vec atomic_prop = nums.cast<double>();
    if (have_ecps) {
        // set nuclear charges to include ecp
        const auto &ecp_electrons = basis.ecp_electrons();
        for (int i = 0; i < atoms.size(); i++) {
            atomic_prop(i) -= ecp_electrons[i];
        }
    }
    fchk.set_vector("Atomic numbers", nums);
    fchk.set_vector("Nuclear charges", atomic_prop);
    fchk.set_vector("Current cartesian coordinates", positions());
    fchk.set_vector("Int Atom Types", occ::IVec::Zero(nums.rows()));
    fchk.set_vector("MM Charges", occ::Vec::Zero(nums.rows()));
    fchk.set_vector("Atom residue num", occ::IVec::Zero(nums.rows()));

    // atomic weights
    for (Eigen::Index i = 0; i < atomic_prop.rows(); i++)
        atomic_prop(i) =
            static_cast<double>(occ::core::Element(nums(i)).mass());
    fchk.set_vector("Integer atomic weights",
                    atomic_prop.array().round().cast<int>());
    fchk.set_vector("Real atomic weights", atomic_prop);

    auto mo_fchk = occ::io::conversion::orb::to_gaussian_order(basis, mo);
    Mat Dfchk;

    std::vector<double> density_lower_triangle, spin_density_lower_triangle;

    if (mo.kind == SpinorbitalKind::Unrestricted) {
        fchk.set_vector("Alpha Orbital Energies", block::a(mo_fchk.energies));
        fchk.set_vector("Alpha MO coefficients", block::a(mo_fchk.C));
        fchk.set_vector("Beta Orbital Energies", block::b(mo_fchk.energies));
        fchk.set_vector("Beta MO coefficients", block::b(mo_fchk.C));
        Mat occ_fchk =
            occ::qm::orb::occupied_unrestricted(mo_fchk.C, mo.n_alpha, mo.n_beta);
        Dfchk = occ::qm::orb::density_matrix_unrestricted(occ_fchk, mo.n_alpha,
                                                          mo.n_beta);

        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        spin_density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        auto da = block::a(Dfchk);
        auto db = block::b(Dfchk);
        for (Eigen::Index row = 0; row < nbf; row++) {
            for (Eigen::Index col = 0; col <= row; col++) {
                double va = da(row, col) * 2, vb = db(row, col) * 2;
                density_lower_triangle.push_back(va + vb);
                spin_density_lower_triangle.push_back(vb - va);
            }
        }
    } else {
        fchk.set_vector("Alpha Orbital Energies", mo_fchk.energies);
        fchk.set_vector("Alpha MO coefficients", mo_fchk.C);
        Mat occ_fchk = occ::qm::orb::occupied_restricted(mo_fchk.C, mo.n_alpha);
        Dfchk = occ::qm::orb::density_matrix_restricted(occ_fchk);
        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        for (Eigen::Index row = 0; row < nbf; row++) {
            for (Eigen::Index col = 0; col <= row; col++) {
                density_lower_triangle.push_back(Dfchk(row, col) * 2);
            }
        }
    }
    fchk.set_vector("Total SCF Density", density_lower_triangle);
    if (spin_density_lower_triangle.size() > 0)
        fchk.set_vector("Spin SCF Density", spin_density_lower_triangle);
    fchk.set_basis(basis);

    std::vector<int> shell2atom;
    shell2atom.reserve(basis.size());
    for (const auto &x : basis.shell_to_atom()) {
        shell2atom.push_back(x + 1);
    }
    fchk.set_vector("Shell to atom map", shell2atom);

    if (have_ecps) {
        occ::log::warn("Writing ECP information to fchk - this is very likely "
                       "unsupported and not working");
        // TODO finish ECP writing routines
        fchk.set_vector<int, double>("ECP-RNFroz", basis.ecp_electrons());
        std::vector<double> ecp_clp1;
        std::vector<double> ecp_clp2;
        std::vector<int> ecp_nlp;
        std::vector<double> ecp_zlp;
        std::vector<int> ecp_lmax(atoms.size(), 0);
        std::vector<int> ecp_kfirst;
        std::vector<int> ecp_klast;
        int ecp_max_length = 0;
        const auto &ecp_shell2atom = basis.ecp_shell_to_atom();
        int shell_index = 0;
        for (const auto &sh : basis.ecp_shells()) {
            int atom_idx = ecp_shell2atom[shell_index];
            ecp_max_length =
                std::max(static_cast<int>(sh.num_primitives()), ecp_max_length);
            ecp_lmax[atom_idx] =
                std::max(static_cast<int>(sh.l), ecp_lmax[atom_idx]);
            for (int i = 0; i < sh.num_primitives(); i++) {
                ecp_clp1.push_back(sh.contraction_coefficients(i, 0));
                ecp_zlp.push_back(sh.exponents(i));
                ecp_nlp.push_back(sh.ecp_r_exponents(i));
                ecp_clp2.push_back(0.0);
            }
            shell_index++;
        }
        fchk.set_scalar("ECP-MaxLECP", ecp_max_length);
        fchk.set_vector("ECP-LMax", ecp_lmax);
        fchk.set_vector("ECP-NLP", ecp_nlp);
        fchk.set_vector("ECP-CLP1", ecp_clp1);
        fchk.set_vector("ECP-CLP2", ecp_clp2);
        fchk.set_vector("ECP-ZLP", ecp_zlp);
    }

    // TODO fix this is wrong
    fchk.set_scalar("Virial ratio", -2 * energy.total / energy.kinetic);
    fchk.set_scalar("SCF ratio", energy.total);
    fchk.set_scalar("Total ratio", energy.total);

    occ::timing::stop(occ::timing::category::io);
}

Vec Wavefunction::electric_potential(const Mat3N &points) const {
    HartreeFock hf(basis);
    Vec esp_e = hf.electronic_electric_potential_contribution(mo, points);
    Vec esp_n = hf.nuclear_electric_potential_contribution(points);
    return esp_e + esp_n;
}

Mat3N Wavefunction::electric_field(const Mat3N &points) const {
    HartreeFock hf(basis);
    Mat3N efield_e = hf.electronic_electric_field_contribution(mo, points);
    Mat3N efield_n = hf.nuclear_electric_field_contribution(points);
    return efield_e + efield_n;
}

Vec Wavefunction::mulliken_charges() const {

    HartreeFock hf(basis);
    Mat overlap = hf.compute_overlap_matrix();

    Vec charges = -2 * occ::qm::mulliken_partition(basis, mo, overlap);

    const auto &ecp_electrons = basis.ecp_electrons();
    for (size_t i = 0; i < atoms.size(); i++) {
        charges(i) += atoms[i].atomic_number - ecp_electrons[i];
    }
    return charges;
}

Vec Wavefunction::electron_density(const Mat3N &pos) const {
    return occ::density::evaluate_density_on_grid<0>(*this, pos);
}

Mat3N Wavefunction::electron_density_gradient(const Mat3N &pos) const {
    return occ::density::evaluate_density_on_grid<1>(*this, pos).rightCols(3).transpose();
}

Vec Wavefunction::electron_density_mo(const Mat3N &pos, int mo_index) const {
    constexpr auto R = SpinorbitalKind::Restricted;

    switch(mo.kind) {
	case R: {
	    auto D = mo.density_matrix_single_mo(mo_index);
	    return occ::density::evaluate_density_on_grid<0, R>(basis, D, pos);
	}
	default:
	    throw std::runtime_error("Only restricted case for mo density implemented");
    }
}

Mat3N Wavefunction::electron_density_mo_gradient(const Mat3N &pos, int mo_index) const {
    constexpr auto R = SpinorbitalKind::Restricted;

    switch(mo.kind) {
	case R: {
	    auto D = mo.density_matrix_single_mo(mo_index);
	    return occ::density::evaluate_density_on_grid<1, R>(basis, D, pos).rightCols(3).transpose();
	}
	default:
	    throw std::runtime_error("Only restricted case for mo density implemented");
    }
}


Wavefunction Wavefunction::load(const std::string &filename) {
    fs::path path(filename);
    std::string ext = path.extension().string();
    if(ext == ".fchk") {
	FchkReader reader(filename);
	return Wavefunction(reader);
    }
    else if(ext ==  ".molden") {
	MoldenReader reader(filename);
	return Wavefunction(reader);
    }
    else if(io::valid_json_format_string(ext)) {
	io::JsonFormat fmt = io::json_format(ext);
	io::JsonWavefunctionReader reader(filename, fmt);
	return reader.wavefunction();
    }
    else throw std::runtime_error(
	fmt::format("Unknown wavefunction format: '{}', could not read in", ext)
    );
}

bool Wavefunction::is_likely_wavefunction_filename(const std::string &filename) {
    fs::path path(filename);
    std::string ext = path.extension().string();
    if(ext == ".fchk") return true;
    else if(ext ==  ".molden") return true;
    else if(io::valid_json_format_string(ext)) return true;
    return false;
}

bool Wavefunction::save(const std::string &filename) {
    fs::path path(filename);
    std::string ext = path.extension().string();
    if(io::valid_json_format_string(ext)) {
	occ::io::JsonWavefunctionWriter json_writer;
	json_writer.set_format(ext);
	json_writer.write(*this, path.string());
	occ::log::info("wavefunction stored in {}", path.string());
	return true;
    }
    else if(ext == "fchk" || ext == ".fchk") {
	occ::io::FchkWriter fchk_writer(path.string());
	save(fchk_writer);
	fchk_writer.write();
	occ::log::info("wavefunction stored in {}", path.string());
	return true;
    }
    occ::log::warn("Unknown wavefunction format: '{}', skipping writing", ext);
    return false;
}

} // namespace occ::qm
