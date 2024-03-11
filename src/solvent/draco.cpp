#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::solvent::draco {

inline double gfn_count(double k, double r, double r0) {
    return 1.0/(1.0 + std::exp(-k*(r0/r-1.0)));
}

Vec coordination_numbers(const std::vector<core::Atom> &atoms) {
    constexpr double ka = 10.0; // steepness of first counting func
    constexpr double kb = 20.0; // steepness of second counting func
    constexpr double r_shift = 2.0; // offset of second counting func
    constexpr double default_cutoff = 25.0;
    constexpr double directed_factor = 1.0;

    core::Molecule mol(atoms);
    Vec cov = mol.covalent_radii().array() * occ::units::ANGSTROM_TO_BOHR;
    Vec cn = Vec::Zero(atoms.size());

    for(int i = 0; i <= atoms.size(); i++) {
	const auto &ai = atoms[i];
	const auto &aj = atoms[i];
	for(int j = 0; j < i; j++) {
	    double r2 = atoms[i].square_distance(atoms[j].x, atoms[j].y, atoms[j].z);
	    if(r2 > default_cutoff * default_cutoff) continue;
	    double r = std::sqrt(r2);
	    double rc = cov(i) + cov(j);
	    double count = gfn_count(ka, r, rc) * gfn_count(kb, r, rc + r_shift);
	    cn(i) += count;
	    if(i != j) {
		cn(j) += count * directed_factor;
	    }
	}
    }
    return cn;
}

Vec smd_coulomb_radii(const Vec &charges, const std::vector<core::Atom> &atoms, const SMDSolventParameters &params) {
    nlohmann::json draco_params = load_draco_parameters();
    if(draco_params.empty()) throw std::runtime_error("No draco parameters set: did you set the OCC_DATA_PATH environment variable?");

    auto cn = coordination_numbers(atoms);
    return Vec();
}

}
