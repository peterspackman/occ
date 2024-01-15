#include <occ/qm/orbital_smearing.h>
#include <occ/qm/mo.h>
#include <occ/core/log.h>
#include <occ/core/optimize.h>
#include <unsupported/Eigen/SpecialFunctions>

namespace occ::qm {

Vec OrbitalSmearing::calculate_fermi_occupations(const MolecularOrbitals &mo) const {
    Vec result = Vec::Zero(mo.energies.rows());
    Vec de = (mo.energies.array() - mu).array() / sigma;
    for(int i = 0; i < result.rows(); i++) {
	if(de(i) < 40) {
	    result(i) = 1.0 / (std::exp(de(i)) + 1.0);
	}
    }
    return result;
}

Vec OrbitalSmearing::calculate_gaussian_occupations(const MolecularOrbitals &mo) const {
    return 0.5 * ((mo.energies.array() - mu).array() / sigma).erfc();
}

Vec OrbitalSmearing::calculate_linear_occupations(const MolecularOrbitals &mo) const {
    const double sigma2 = sigma * sigma;
    Vec result = Vec::Zero(mo.energies.rows());
    Vec de = mo.energies.array() - mu;
    for(int i = 0; i < result.rows(); i++) {
	const double d = de(i);
	if(d <= -sigma) {
	    result(i) = 1.0;
	}
	else if (d >= sigma) {
	    result(i) = 0.0;
	}
	else if (d < 1e-10) {
	    result(i) = 1.0 - (d + sigma) * (d + sigma) / 2 / sigma2;
	}
        else if (d > 0) {
	    result(i) = (d - sigma) * (d - sigma) / 2 / sigma2;
        }
    }
    return result;
}

void OrbitalSmearing::smear_orbitals(MolecularOrbitals &mo) {
    if(kind == Kind::None) return;
    using occ::core::opt::LineSearch;

    size_t n_occ = mo.n_alpha;
    fermi_level = mo.energies(n_occ - 1);
    occ::log::debug("Fermi level: {}, occ_sum: {}", fermi_level, mo.occupation.sum());

    auto cost_function = [&](double mu_value) {
        mu = mu_value;
	switch(kind) {
	    case Kind::Fermi:
		mo.occupation = calculate_fermi_occupations(mo);
		break;
	    case Kind::Gaussian:
		mo.occupation = calculate_gaussian_occupations(mo);
		break;
	    case Kind::Linear:
		mo.occupation = calculate_linear_occupations(mo);
		break;
	    default:
		break;
	}
	int num_electrons = mo.n_alpha + mo.n_beta;
	double sum_occ = (mo.occupation.array() * 2).sum();
	double diff = (num_electrons - sum_occ);
	return diff*diff;
    };


    LineSearch opt(cost_function);
    opt.set_left(fermi_level);
    opt.set_right(-fermi_level);
    opt.set_guess(fermi_level);
    double fmin = opt.f_xmin();

    occ::log::info("mu: {}, diff: {}", mu, fmin);
    occ::log::info("Occ\n{}\n", mo.occupation);
    occ::log::info("energies: {}\n", mo.energies);
    mo.update_occupied_orbitals_fractional();

    entropy = calculate_entropy(mo);
    occ::log::debug("entropy term: {}", entropy);

}

double OrbitalSmearing::calculate_entropy(const MolecularOrbitals &mo) const {
    double result = 0.0;
    if(kind == Kind::Fermi) {
	for(int i = 0; i < mo.occupation.rows(); i++) {
	    const double occ = mo.occupation(i);
	    if(occ >= 1.0 || occ <= 0.0) continue;
	    result -= occ * std::log(occ) + (1 - occ) * std::log(1 - occ);
	}
    }
    else {
	const double sigma2 = sigma * sigma;
	for(int i = 0; i < mo.energies.rows(); i++) {
	    const double e = mo.energies(i);
            const double x = (e - mu)/sigma;
	    result -= -std::exp(- x * x);
	}
	result /= 2 * std::sqrt(M_PI);
    }
    return result;
}

}
