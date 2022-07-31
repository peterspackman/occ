#include <cmath>
#include <iostream>
#include <occ/qm/wavefunction.h>
#include <occ/io/fchkreader.h>
#include <occ/core/element.h>
#include <occ/io/moldenreader.h>
#include <filesystem>
#include <fmt/core.h>
#include <occ/gto/gto.h>
#include <occ/gto/density.h>
#include <occ/dft/grid.h>
#include <occ/slater/slaterbasis.h>
#include <vector>

using occ::qm::Wavefunction;
using occ::Mat;
using occ::Vec;

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


double becke_hole(double rho, double quad, double hnorm) {
    double x{0}, x1{0}, f{0}, df{0};
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
    const double expo = std::exp(-x);
    const double prefac = rho / expo;
    const double alf = std::pow(8.0 * M_PI * prefac / hnorm, 1.0/3.0);
    return x / alf;
}

Mat read_matrix_from_file(const std::string &file, int rows, int cols) {

  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;
  Mat res(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}

int main(int argc, char *argv[])
{
    using occ::dft::AtomGrid;
    using occ::dft::MolecularGrid;
    auto wfn = load_wavefunction("water.fchk");
    fmt::print("wfn loaded: {}\n", "water.fchk");
    Mat postg_grid = read_matrix_from_file("postg_grid", 27160, 4);

    const size_t num_atoms = wfn.atoms.size();
    auto grid = MolecularGrid(wfn.basis);
    std::vector<AtomGrid> atom_grids;
    for(int i = 0; i < wfn.atoms.size(); i++) {
        atom_grids.push_back(grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        atom_grids.begin(), atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    fmt::print("finished calculating atom grids ({} points)\n", num_grid_points);

    occ::gto::GTOValues gto_vals;

    constexpr size_t BLOCKSIZE = 64;
    gto_vals.reserve(wfn.basis.nbf(), BLOCKSIZE, 2);
    Mat rho = Mat::Zero(BLOCKSIZE, occ::density::num_components(2));
    auto slaterbasis_data = occ::slater::load_slaterbasis("thakkar");
    std::vector<occ::slater::Basis> element_bases;
    Vec hirshfeld_charges = Vec::Zero(num_atoms);
    for(int i = 0; i < num_atoms; i++) {
	auto el = occ::core::Element(wfn.atoms[i].atomic_number);
	std::string sym = el.symbol();
	if(sym == "H") sym = "H_normal";
	element_bases.push_back(slaterbasis_data[sym]);
	hirshfeld_charges(i) = static_cast<double>(wfn.atoms[i].atomic_number);
    }
    double num_electrons{0.0};
    double num_electrons_promol{0.0};
    Mat mm = Mat::Zero(3, num_atoms);
    Vec volume = Vec::Zero(num_atoms);
    Vec volume_free = Vec::Zero(num_atoms);

    for(const auto &atom_grid: atom_grids) {
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
	    occ::gto::evaluate_basis(wfn.basis, pts_block, gto_vals, 2);
	    occ::density::evaluate_density<2, occ::qm::SpinorbitalKind::Restricted>(
		wfn.mo.D, gto_vals, rho);
	    for(int i = 0; i < num_atoms; i++) {
		auto el = occ::core::Element(i);
		const auto sb = element_bases[i];
		occ::Vec3 pos{wfn.atoms[i].x, wfn.atoms[i].y, wfn.atoms[i].z};
		r.col(i) = (pts_block.colwise() - pos).colwise().norm();
		const auto & ria = r.col(i).array();
		// currently the hirsfheld weights array just holds the free
		// atom density
		hirshfeld_weights.col(i) = sb.rho(r.col(i));
		volume_free(i) += (hirshfeld_weights.col(i).array()  * weights_block.array() * 
		    ria * ria * ria).sum();
	    }
	    for(int j = 0; j < hirshfeld_weights.rows(); j++) {
		double rtot = hirshfeld_weights.row(j).sum();
		if(rtot < 1e-30) continue;
		// now it holds the weight function
		hirshfeld_weights.row(j).array() /= rtot;
		hirshfeld_charges -= hirshfeld_weights.row(j).transpose() * 2 * rho(j, 0) * weights_block(j);
		num_electrons_promol += rtot * weights_block(j);
		if(rho(j, 0) > 1e-10) {
		    double lapl = rho(j, 4);
		    double tau = 2 * rho(j, 5);
		    double sigma = rho(j, 1) * rho(j, 1) + rho(j, 2) * rho(j, 2) + rho(j, 3) * rho(j, 3);
		    double dsigs = tau - 0.25 * sigma / std::max(rho(j, 0), 1e-30);
		    double q = (lapl - 2 * dsigs) / 6.0;
		    try {
			double bhole = becke_hole(rho(j, 0), q, 1.0);
			const auto& rja = r.row(j).array();
			//fmt::print("{} {} {} {}\n", rho(j, 0), q, 1.0, bhole);
			occ::RowVec r_sub_b = (rja - bhole).unaryExpr([](double x) { return std::max(x, 0.0); }).transpose();
			mm.row(0).array() += 2 * hirshfeld_weights.row(j).array() * (rja - r_sub_b.array()).pow(2) * rho(j, 0) * weights_block(j);
			mm.row(1).array() += 2 * hirshfeld_weights.row(j).array() * (rja.pow(2) - r_sub_b.array().pow(2)).pow(2) * rho(j, 0) * weights_block(j);
			mm.row(2).array() += 2 * hirshfeld_weights.row(j).array() * (rja.pow(3) - r_sub_b.array().pow(3)).pow(2) * rho(j, 0) * weights_block(j);
		    }
		    catch (const char *ex) {
			    fmt::print("Error: {}\n", ex);
			    return 1;
		    }
		}
		num_electrons += 2 * rho(j, 0) * weights_block(j);
		volume.array() += (hirshfeld_weights.row(j).array() * r.row(j).array() * r.row(j).array() * r.row(j).array()).transpose().array() * 2 * rho(j, 0) * weights_block(j);
	    }
	}
    }
    fmt::print("Num electrons          {:12.7f}\n", num_electrons);
    fmt::print("Num electrons (promol) {:12.7f}\n", num_electrons_promol);
    fmt::print("Hirshfeld charges\n{}\n", hirshfeld_charges);
    fmt::print("Moments^2 (row = l - 1, col = atom)\n{}\n", mm);
    fmt::print("Volume\n{}\n", volume);
    fmt::print("Volume Free\n{}\n", volume_free);

    return 0;
}

