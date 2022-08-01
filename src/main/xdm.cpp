#include <cmath>
#include <iostream>
#include <occ/qm/wavefunction.h>
#include <occ/io/fchkreader.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/io/moldenreader.h>
#include <filesystem>
#include <fmt/core.h>
#include <occ/gto/gto.h>
#include <occ/gto/density.h>
#include <occ/dft/grid.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/units.h>
#include <occ/dft/functional.h>
#include <occ/dft/dft.h>
#include <occ/core/timings.h>
#include <vector>

using occ::qm::Wavefunction;
using occ::Mat;
using occ::Vec;

static const std::array<double, 110> Thakkar_atomic_polarizability{
    4.50,   1.38,   164.04, 37.74,  20.43,  11.67,  7.26,   5.24,   3.70,
    2.66,   162.88, 71.22,  57.79,  37.17,  24.93,  19.37,  14.57,  11.09,
    291.10, 157.90, 142.30, 114.30, 97.30,  94.70,  75.50,  63.90,  57.70,
    51.10,  45.50,  38.35,  52.91,  40.80,  29.80,  26.24,  21.13,  16.80,
    316.20, 199.00, 153.00, 121.00, 106.00, 86.00,  77.00,  65.00,  58.00,
    32.00,  52.46,  47.55,  68.67,  57.30,  42.20,  38.10,  32.98,  27.06,
    396.00, 273.50, 210.00, 200.00, 190.00, 212.00, 203.00, 194.00, 187.00,
    159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30, 148.00, 109.00,
    88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73,  71.72,
    60.05,  48.60,  43.62,  40.73,  33.18,  315.20, 246.20, 217.00, 217.00,
    171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00,
    161.00, 123.00, 118.00, 0.00,   0.00,   0.00,   0.00,   0.00,   0.00,
    0.00,   0.00};



Wavefunction load_wavefunction(const std::string &filename) {
    namespace fs = std::filesystem;
    using occ::util::to_lower;
    std::string ext = fs::path(filename).extension().string();
    to_lower(ext);
    if (ext == ".fchk") {
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    }
    if (ext == ".molden" || ext == ".input") {
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        return Wavefunction(molden);
    }
    throw std::runtime_error(
        "Unknown file extension when reading wavefunction: " + ext);
}

void xfuncs(double x, double rhs, double &f, double &df) {
    // working
    double expo23 = std::exp(-2.0/3.0 *x);
    f = x * expo23 / (x - 2.0) - rhs;
    df = 2.0/3.0 * (2.0 * x - x*x - 3.0)/((x - 2.0) * (x - 2.0)) *expo23;
}

double br89_analytic(double rho, double Q, double norm) {
    double alpha1 = 1.5255251812009530;
    double alpha2 = 0.4576575543602858;
    double alpha3 = 0.4292036732051034;

    const double c[6] = {
	0.7566445420735584,
	-2.6363977871370960,
	5.4745159964232880,
	-12.657308127108290,
	4.1250584725121360,
	-30.42513395716384
    };

    const double b[6] = {
	0.4771976183772063,
	-1.7799813494556270,
	3.8433841862302150,
	-9.5912050880518490,
	2.1730180285916720,
	-30.425133851603660
    };

    const double d[6] = {
	0.00004435009886795587,
	0.58128653604457910,
	66.742764515940610,
	434.26780897229770,
	824.7765766052239000,
	1657.9652731582120
    };
    const double e[6] = {
	0.00003347285060926091,
	0.47917931023971350,
	62.392268338574240,
	463.14816427938120,
	785.2360350104029000,
	1657.962968223273000000
    };
    const double B = 2.085749716493756;

    const double third2 = 2.0 / 3.0;
    const double y = third2 * std::pow(M_PI * rho / norm, third2) * rho / Q;
    if(y <= 0) {
	const double g = -std::atan(alpha1 * y + alpha2) + alpha3;
	double p1y = 0.0;
	double p2y = 0.0;
	double yi = 1.0;
	for(int i = 0; i < 6; i++) {
	    p1y += c[i] * yi;
	    p2y += b[i] * yi;
	    yi *= y;
	}
	return g * p1y / p2y;
    }
    else {
	const double By = B * y;
	// inverse hyperbolic cosecant
	const double g = std::log(1.0 / By + std::sqrt(1.0 / (By * By) + 1)) + 2;
	double p1y = 0.0;
	double p2y = 0.0;
	double yi = 1.0;
	for(int i = 0; i < 6; i++) {
	    p1y += d[i] * yi;
	    p2y += e[i] * yi;
	    yi *= y;
	}
	return g * p1y / p2y;
    }
}

double becke_hole(double rho, double quad, double hnorm) {
    const bool use_analytic = true;
    double x = 0.0;
    if (use_analytic) {
	x = br89_analytic(rho, quad, hnorm);
    }
    else {
	double x1{0}, f{0}, df{0};
	double third2 = 2.0 / 3.0;

	const double rhs = third2 * std::pow(M_PI * rho / hnorm, third2) * rho/quad;
	double x0 = 2.0;
	double shift = (rhs > 0) ? 1.0 : -1.0;
	bool initialized = false;
	for(int i = 0; i < 16; i++) {
	    x = x0 + shift;
	    xfuncs(x, rhs, f, df);
	    if (f * rhs > 0.0) {
		initialized = true;
		break;
	    }
	    shift = 0.1 * shift;
	}
	if(!initialized)
	    throw std::runtime_error("newton algorithm failed to initialize");

	bool converged = false;
	for(int i = 0; i < 100; i++) {
	    xfuncs(x,rhs,f,df);
	    x1 = x - f / df;
	    if(std::abs(x1 - x) < 1e-10) {
		converged = true;
		break;
	    }
	    x = x1;
	}
	if(!converged) 
	    throw std::runtime_error("newton algorithm failed to converge");
	x = x1;
    }

    const double expo = std::exp(-x);
    const double prefac = rho / expo;
    const double alf = std::pow(8.0 * M_PI * prefac / hnorm, 1.0/3.0);
    return x / alf;
}

double xdm_polarizability(int n, double v, double vfree) {
    return  v * Thakkar_atomic_polarizability[n - 1] / vfree;
}

double xdm_dispersion_energy(const std::vector<occ::core::Atom> &atoms,
	const Mat &moments, const Vec &volume, const Vec &volume_free, double alpha1 = 1.0, double alpha2 = 1.0) {
    const size_t num_atoms = atoms.size();
    using occ::Vec3;
    Vec polarizabilities(num_atoms);
    for(int i = 0; i < num_atoms; i++) {
	polarizabilities(i) = xdm_polarizability(atoms[i].atomic_number, volume(i), volume_free(i));
    }
    fmt::print("Volume\n{}\n", volume);
    fmt::print("Volume Free\n{}\n", volume_free);
    fmt::print("Polarizibility: \n{}\n", polarizabilities);
    double edisp = 0.0;
    for(int i = 0; i < num_atoms; i++) {
	Vec3 pi = {atoms[i].x, atoms[i].y, atoms[i].z};
	double pol_i = polarizabilities(i);
	for(int j = i; j < num_atoms; j++) {
	    Vec3 pj = {atoms[j].x, atoms[j].y, atoms[j].z};
	    double pol_j = polarizabilities(j);
	    double factor = pol_i * pol_j / (moments(0, i) * pol_j + moments(0, j) * pol_i);
	    double rij = (pj - pi).norm();
	    double c6 = factor * moments(0, i) * moments(0, j);
	    double c8 = 1.5 * factor *(moments(0, i) * moments(1, j) + moments(1, i) * moments(0, j));
	    double c10 = 2.0 * factor *(moments(0, i) * moments(2, j) + moments(2, i)* moments(0, j))
		+ 4.2 * factor * moments(1, i) * moments(1, j);
	    double rc =  (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) +
			  std::sqrt(c10 / c8)) / 3;
	    double rvdw = alpha1 * rc + alpha2 * occ::units::ANGSTROM_TO_BOHR;

	    fmt::print("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n", i, j, rij, c6, c8, c10, rc, rvdw);
	    if(rij > 1e-15) {
		edisp -= c6 / (std::pow(rvdw, 6) + std::pow(rij, 6)) + c8 / (std::pow(rvdw, 8) + std::pow(rij, 8)) +
               c10 / (std::pow(rvdw, 10) + std::pow(rij, 10));

	    }
	}
    }
    return edisp;
}


namespace impl {

    void xdm_moment_kernel_restricted(
	Eigen::Ref<const Mat> r,
	Eigen::Ref<const Mat> rho,
	Eigen::Ref<const Vec> weights,
	Eigen::Ref<const Mat> rho_pro,
	Eigen::Ref<Vec> hirshfeld_charges,
	Eigen::Ref<Mat> volume,
	Eigen::Ref<Mat> moments,
	double &num_electrons,
	double &num_electrons_promol) {

	for(int j = 0; j < rho_pro.rows(); j++) {
	    double protot = rho_pro.row(j).sum();
	    if(protot < 1e-30) continue;
	    // now it holds the weight function
	    occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
	    hirshfeld_charges -= hirshfeld_weights.row(j).transpose() * 2 * rho(j, 0) * weights(j);
	    num_electrons_promol += protot * weights(j);
	    double lapl = rho(j, 4);
	    double tau = 2 * rho(j, 5);
	    double sigma = rho(j, 1) * rho(j, 1) + rho(j, 2) * rho(j, 2) + rho(j, 3) * rho(j, 3);
	    double dsigs = tau - 0.25 * sigma / std::max(rho(j, 0), 1e-30);
	    double q = (lapl - 2 * dsigs) / 6.0;

	    double bhole = becke_hole(rho(j, 0), q, 1.0);
	    const auto& rja = r.row(j).array();
	    occ::RowVec r_sub_b = (rja - bhole).unaryExpr([](double x) { return std::max(x, 0.0); }).transpose();
	    moments.row(0).array() += 2 * hirshfeld_weights.array() * (rja - r_sub_b.array()).pow(2) * rho(j, 0) * weights(j);
	    moments.row(1).array() += 2 * hirshfeld_weights.array() * (rja.pow(2) - r_sub_b.array().pow(2)).pow(2) * rho(j, 0) * weights(j);
	    moments.row(2).array() += 2 * hirshfeld_weights.array() * (rja.pow(3) - r_sub_b.array().pow(3)).pow(2) * rho(j, 0) * weights(j);
	    num_electrons += 2 * rho(j, 0) * weights(j);
	    volume.array() += (hirshfeld_weights.array() * r.row(j).array() * r.row(j).array() * r.row(j).array()).transpose().array() * 2 * rho(j, 0) * weights(j);
	}
    }

    void xdm_moment_kernel_unrestricted(
	Eigen::Ref<const Mat> r,
	Eigen::Ref<const Mat> rho,
	Eigen::Ref<const Vec> weights,
	Eigen::Ref<const Mat> rho_pro,
	Eigen::Ref<Vec> hirshfeld_charges,
	Eigen::Ref<Mat> volume,
	Eigen::Ref<Mat> moments,
	double &num_electrons,
	double &num_electrons_promol) {

	const auto &rho_a = occ::qm::block::a(rho);
	const auto &rho_b = occ::qm::block::b(rho);

	for(int j = 0; j < rho_pro.rows(); j++) {
	    double protot = rho_pro.row(j).sum();
	    if(protot < 1e-30) continue;
	    // now it holds the weight function
	    occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
	    hirshfeld_charges -= hirshfeld_weights.row(j).transpose() * 2 * rho(j, 0) * weights(j);
	    num_electrons_promol += protot * weights(j);
	    double lapl_a = rho_a(j, 4);
	    double tau_a = 2 * rho_a(j, 5);
	    double sigma_a = rho_a(j, 1) * rho_a(j, 1) + rho_a(j, 2) * rho_a(j, 2) + rho_a(j, 3) * rho_a(j, 3);
	    double dsigs_a = tau_a - 0.25 * sigma_a / std::max(rho_a(j, 0), 1e-30);
	    double q_a = (lapl_a - 2 * dsigs_a) / 6.0;
	    double bhole_a = becke_hole(rho_a(j, 0), q_a, 1.0);

	    double lapl_b = rho_b(j, 4);
	    double tau_b = 2 * rho_b(j, 5);
	    double sigma_b = rho_b(j, 1) * rho_b(j, 1) + rho_b(j, 2) * rho_b(j, 2) + rho_b(j, 3) * rho_b(j, 3);
	    double dsigs_b = tau_b - 0.25 * sigma_b / std::max(rho_b(j, 0), 1e-30);
	    double q_b = (lapl_b - 2 * dsigs_b) / 6.0;
	    double bhole_b = becke_hole(rho_b(j, 0), q_b, 1.0);

	    const auto& rja = r.row(j).array();
	    occ::RowVec r_sub_ba = (rja - bhole_a).unaryExpr([](double x) { return std::max(x, 0.0); }).transpose();
	    occ::RowVec r_sub_bb = (rja - bhole_b).unaryExpr([](double x) { return std::max(x, 0.0); }).transpose();

	    moments.row(0).array() += hirshfeld_weights.array() * weights(j) *
		((rja - r_sub_ba.array()).pow(2) * rho_a(j, 0)  + (rja - r_sub_bb.array()).pow(2) * rho_b(j, 0));

	    moments.row(1).array() += hirshfeld_weights.array() * (rja.pow(2) - r_sub_ba.array().pow(2)).pow(2) * rho_a(j, 0) * weights(j);
	    moments.row(1).array() += hirshfeld_weights.array() * (rja.pow(2) - r_sub_bb.array().pow(2)).pow(2) * rho_b(j, 0) * weights(j);

	    moments.row(2).array() += hirshfeld_weights.array() * (rja.pow(3) - r_sub_ba.array().pow(3)).pow(2) * rho_a(j, 0) * weights(j);
	    moments.row(2).array() += hirshfeld_weights.array() * (rja.pow(3) - r_sub_bb.array().pow(3)).pow(2) * rho_b(j, 0) * weights(j);

	    num_electrons += (rho_a(j, 0) +  rho_b(j, 0)) * weights(j);
	    volume.array() += (hirshfeld_weights.array() * r.row(j).array() * r.row(j).array() * r.row(j).array()).transpose().array() * (rho_a(j, 0) + rho_b(j, 0)) * weights(j);
	}
    }

}


class XDM {
public:
    struct Parameters {
	double a1{1.0};
	double a2{1.0}; // angstroms
    };

    XDM(const occ::qm::AOBasis &basis) : m_basis(basis), m_grid(basis) {
	for(int i = 0; i < basis.atoms().size(); i++) {
	    m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
	}
	size_t num_grid_points = std::accumulate(
	    m_atom_grids.begin(), m_atom_grids.end(), 0.0,
	    [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
	m_slater_basis = occ::slater::slaterbasis_for_atoms(m_basis.atoms());
    }

    double energy(const occ::qm::MolecularOrbitals &mo) {
	populate_moments(mo);
	fmt::print("moments\n{}\n", m_moments);
	
	double edisp = xdm_dispersion_energy(m_basis.atoms(), m_moments, m_volume, m_volume_free);

	return edisp;
    }

    inline const auto &moments() const { return m_moments; }
    inline const auto &hirshfeld_charges() const { return m_hirshfeld_charges; }
    inline const auto &atom_volume() const { return m_volume; }
    inline const auto &free_atom_volume() const { return m_volume_free; }

private:
    void populate_moments(const occ::dft::MolecularOrbitals &mo) {
	if(m_density_matrix.size() != 0 && occ::util::all_close(mo.D, m_density_matrix)) {
	    return;
	}
	m_density_matrix = mo.D;

	occ::gto::GTOValues gto_vals;
	const auto &atoms = m_basis.atoms();
	const size_t num_atoms = atoms.size();

	bool unrestricted = (mo.kind == occ::qm::SpinorbitalKind::Unrestricted);

	constexpr size_t BLOCKSIZE = 64;
	gto_vals.reserve(m_basis.nbf(), BLOCKSIZE, 2);
	Mat rho = Mat::Zero(BLOCKSIZE, occ::density::num_components(2));
	m_hirshfeld_charges = Vec::Zero(num_atoms);
	for(int i = 0; i < num_atoms; i++) {
	    m_hirshfeld_charges(i) = static_cast<double>(atoms[i].atomic_number);
	}
	double num_electrons{0.0};
	double num_electrons_promol{0.0};
	m_moments = Mat::Zero(3, num_atoms);
	m_volume = Vec::Zero(num_atoms);
	m_volume_free = Vec::Zero(num_atoms);

	constexpr double density_tolerance = 1e-10;

	for(const auto &atom_grid: m_atom_grids) {
	    const auto &atom_pts = atom_grid.points;
	    const auto &atom_weights = atom_grid.weights;
	    const size_t npt_total = atom_pts.cols();
	    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

	    for (size_t block = 0; block < num_blocks; block++) {
		Eigen::Index l = block * BLOCKSIZE;
		Eigen::Index u =
		    std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
		Eigen::Index npt = u - l;
		if (npt <= 0)
		    continue;
		Mat hirshfeld_weights = Mat::Zero(npt, num_atoms);
		Mat r = Mat::Zero(npt, num_atoms);
		const auto &pts_block = atom_pts.middleCols(l, npt);
		const auto &weights_block = atom_weights.segment(l, npt);
		occ::gto::evaluate_basis(m_basis, pts_block, gto_vals, 2);
		if(unrestricted) {
		    occ::density::evaluate_density<2, occ::qm::SpinorbitalKind::Unrestricted>(
			m_density_matrix * 2, gto_vals, rho);
		} else {
		    occ::density::evaluate_density<2, occ::qm::SpinorbitalKind::Restricted>(
			m_density_matrix, gto_vals, rho);
		}

		for(int i = 0; i < num_atoms; i++) {
		    auto el = occ::core::Element(i);
		    const auto &sb = m_slater_basis[i];
		    occ::Vec3 pos{atoms[i].x, atoms[i].y, atoms[i].z};
		    r.col(i) = (pts_block.colwise() - pos).colwise().norm();
		    const auto & ria = r.col(i).array();
		    // currently the hirsfheld weights array just holds the free
		    // atom density
		    hirshfeld_weights.col(i) = sb.rho(r.col(i));
		    m_volume_free(i) += (hirshfeld_weights.col(i).array()  * weights_block.array() * 
			ria * ria * ria).sum();
		}
		if(unrestricted) {
		    impl::xdm_moment_kernel_unrestricted(r, rho, weights_block, hirshfeld_weights, m_hirshfeld_charges, m_volume, m_moments, num_electrons, num_electrons_promol); 
		}
		else {
		    impl::xdm_moment_kernel_restricted(r, rho, weights_block, hirshfeld_weights, m_hirshfeld_charges, m_volume, m_moments, num_electrons, num_electrons_promol); 
		}

	    }
	}
    }

    occ::qm::AOBasis m_basis;
    occ::dft::MolecularGrid m_grid;
    std::vector<occ::dft::AtomGrid> m_atom_grids;
    std::vector<occ::slater::Basis> m_slater_basis;
    Mat m_density_matrix;
    Mat m_moments;
    Vec m_volume;
    Vec m_volume_free;
    Vec m_hirshfeld_charges;
};

struct XDMResult{
    double energy{0.0};

};


int main(int argc, char *argv[])
{
    auto wfn = load_wavefunction("water.fchk");
    fmt::print("wfn loaded: {}\n", "water.fchk");

    auto xdm = XDM(wfn.basis);
    double e = xdm.energy(wfn.mo);
    fmt::print("e = {}\n", e);
    return 0;
}

