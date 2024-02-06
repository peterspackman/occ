#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::qm {

struct MolecularOrbitals;

struct OrbitalSmearing {
    enum class Kind {
	None,
	Fermi,
	Gaussian,
	Linear
    };
    Kind kind{Kind::None};
    double mu{0.0};
    double fermi_level{0.0};
    double sigma{0.095};
    double entropy{0.0};
    void smear_orbitals(MolecularOrbitals &);

    double calculate_entropy(const MolecularOrbitals &) const;
    inline double ec_entropy() const { return -sigma * entropy; }

    Vec calculate_fermi_occupations(const MolecularOrbitals &) const;
    Vec calculate_gaussian_occupations(const MolecularOrbitals &) const;
    Vec calculate_linear_occupations(const MolecularOrbitals &) const;
};

}
