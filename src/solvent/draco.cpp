#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <cmath>
#include <occ/core/log.h>

namespace occ::solvent::draco {

namespace impl {
   // Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197)
   // values for metals decreased by 10%
   inline const std::array<double, 119> covalent{
      -1.0,
       0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25,
       1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10,
       1.07, 1.04, 1.00, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14, 1.17,
       1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23,
       1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57,
       1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37,
       1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31,
       1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49, 1.49,
       1.51, 1.51, 1.48, 1.50, 1.56, 1.58, 1.45, 1.41, 1.34, 1.29, 1.27, 1.21,
       1.16, 1.15, 1.09, 1.22, 1.36, 1.43, 1.46, 1.58, 1.48, 1.57
    };

   inline double get_covalent_radius_d3(int n) {
       return covalent[n] * 4 * occ::units::ANGSTROM_TO_BOHR / 3;
   }

}


inline double gfn_count(double k, double r, double r0) {
    return 1.0/(1.0 + std::exp(-k*(r0/r-1.0)));
}

Vec coordination_numbers(const std::vector<core::Atom> &atoms) {
    constexpr double ka = 10.0; // steepness of first counting func
    constexpr double kb = 20.0; // steepness of second counting func
    constexpr double r_shift = 2.0; // offset of second counting func
    constexpr double default_cutoff = 25.0;
    constexpr double cutoff2 = default_cutoff * default_cutoff;

    Vec cn = Vec::Zero(atoms.size());

    for(int i = 0; i < atoms.size(); i++) {
	const auto &ai = atoms[i];
	const double cov_i = impl::get_covalent_radius_d3(ai.atomic_number);
	for(int j = 0; j < i; j++) {
	    const auto &aj = atoms[j];
	    double r2 = ai.square_distance(aj);
	    if(r2 > cutoff2) continue;
	    const double cov_j = impl::get_covalent_radius_d3(aj.atomic_number);
	    double r = std::sqrt(r2);
	    double rc = cov_i + cov_j;
	    double count = gfn_count(ka, r, rc) * gfn_count(kb, r, rc + r_shift);
	    cn(i) += count;
	    if(i != j) {
		cn(j) += count;
	    }
	}
    }
    return cn;
}


/*
   subroutine calc_radii(mol, q, radtype, solvent, prefac, expo, o_shift, &
                  & radii_in, cn, k1, radii_out, atoms_to_change_radii, damp_small_rad,&
                  & dqdr, dcndr, drdr)
      use iso_fortran_env, only: output_unit
      !> Molecular structure data
      type(structure_type), intent(inout) :: mol
      !> Charge model charges
      real(wp), intent(in) :: q(mol%nat)

      !> Parameter
      real(wp), intent(in) :: prefac(:), expo(:), o_shift(:), k1(:)

      !> Coordination number
      real(wp), intent(in) :: cn(:)

      !> Derivatives
      real(wp), intent(in), optional, contiguous :: dqdr(:, :, :), dcndr(:, :, :)
      real(wp), intent(out), allocatable, optional :: drdr(:, :, :)

      !> Solvent
      character(len=*), intent(in) :: solvent
      !> Radii type
      character(len=*), intent(in) :: radtype
      !> Damping if radii too small?
      logical, intent(in) :: damp_small_rad
      !> Radii to be scaled
      integer, intent(in) :: atoms_to_change_radii(:)
      real(wp), intent(in) :: radii_in(mol%nat)
      real(wp), intent(out) :: radii_out(mol%nat)
      real(wp) :: asym, x3, x2, x1, a, b, c, d, k
      real(wp) :: eps, alpha_beta_scaling_h, alpha_beta_scaling_o
      real(wp) :: alpha, beta
      real(wp) :: outer

      real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)

      integer :: i

      logical :: grad

      character(len=*), parameter :: source = "calc_radii"

      alpha = get_alpha(solvent)
      beta = get_beta(solvent)
      eps = get_eps(trim(solvent))

      grad = present(dqdr) .and. present(dcndr) .and. present(drdr)
      if (grad) allocate (drdr(3, mol%nat, mol%nat))

      do i = 1, mol%nat
         if (any(atoms_to_change_radii == mol%num(mol%id(i)))) then
            a = prefac(mol%num(mol%id(i)))
            b = expo(mol%num(mol%id(i)))
            k = k1(mol%num(mol%id(i)))
            radii_out(i) = erf(a*(q(i) + k*q(i)*cn(i) - b)) + 1
            if (damp_small_rad .and. (radii_out(i) < min_rad(radtype))) &
                    & radii_out(i) = min_rad(radtype) !Set value to minimal radii tested
            if (grad) then
               outer = 2*exp(-a**2*(q(i) + k*q(i)*cn(i) - b)**2)/sqrt(pi)
               drdr(:, :, i) = a*(dqdr(:, :, i) + k*dqdr(:, :, i)*cn(i) + q(i)*k*dcndr(:, :, i))*outer
            end if
         else
            radii_out(i) = radii_in(i)/aatoau
         end if
      end do

      select case (radtype)
      case ('bondi')
         !Use the scaling on bondi radii
         do i = 1, mol%nat
            if (any(atoms_to_change_radii == mol%num(mol%id(i)))) then
               radii_out(i) = radii_out(i)*(radii_in(i)/aatoau)
               if (grad) then
                  drdr(:, :, i) = drdr(:, :, i)*(radii_in(i)/aatoau)
               end if
            end if
         end do

      case ('cpcm', 'cosmo')
         !Use the scaling on cpcm radii
         do i = 1, mol%nat
            if (any(atoms_to_change_radii == mol%num(mol%id(i)))) then
               !> Change only radii of Oxygene (H-bond acceptor)
               if (mol%num(mol%id(i)) == 8 .AND. get_alpha(solvent) < 0.43) then
                  radii_out(i) = radii_out(i) + (o_shift(mol%num(mol%id(i)))*(0.43 - get_alpha(solvent)))
               end if
               radii_out(i) = radii_out(i)*(radii_in(i)/aatoau)
               if (grad) then
                  drdr(:, :, i) = drdr(:, :, i)*(radii_in(i)/aatoau)
               end if
            end if
         end do

      case ('smd')
         !Use the scaling on SMD radii
         do i = 1, mol%nat
            if (any(atoms_to_change_radii == mol%num(mol%id(i)))) then
               !> Change only radii of Oxygene (H-bond acceptor)
               if (mol%num(mol%id(i)) == 8 .AND. get_alpha(solvent) < 0.43) then
                  radii_out(i) = radii_out(i) + (o_shift(mol%num(mol%id(i)))*(0.43 - get_alpha(solvent)))
               end if
               radii_out(i) = radii_out(i)*(radii_in(i)/aatoau)
               if (grad) then
                  drdr(:, :, i) = drdr(:, :, i)*(radii_in(i)/aatoau)
               end if
            end if
         end do
      end select

   end subroutine calc_radii

end module draco_calc
*/

Vec smd_coulomb_radii(const Vec &charges, const std::vector<core::Atom> &atoms, const SMDSolventParameters &params) {
    nlohmann::json draco_params = load_draco_parameters();
    if(draco_params.empty()) throw std::runtime_error("No draco parameters set: did you set the OCC_DATA_PATH environment variable?");

    std::vector<double> kfacs, prefactors, exponents, radii;
    double o_shift{0.0};
    draco_params["vdw"]["smd"].get_to(radii);
    if(params.is_water) {
	draco_params["eeq_smd"]["k_water"].get_to(kfacs);
	draco_params["eeq_smd"]["exponents_water"].get_to(exponents);
	draco_params["eeq_smd"]["prefactors_water"].get_to(prefactors);
    }
    else {
	draco_params["eeq_smd"]["k"].get_to(kfacs);
	draco_params["eeq_smd"]["exponents"].get_to(exponents);
	draco_params["eeq_smd"]["prefactors"].get_to(prefactors);
	draco_params["eeq_smd"]["o_shift"].get_to(o_shift);
    }

    auto cn = coordination_numbers(atoms);

    Vec result(atoms.size());
    Vec unscaled(atoms.size());

    for(int i = 0; i < atoms.size(); i++) {
	const int num = atoms[i].atomic_number;
	const int idx = num - 1;
	double a = prefactors[idx];
	double b = exponents[idx];
	double k = kfacs[idx];
	double rad = radii[idx];
	result(i) = std::erf(a * (charges(i) + k * charges(i) * cn(i) - b)) + 1;
	if(num == 8 && params.acidity < 0.43) {
	    result(i) += o_shift * (0.43 - params.acidity);
	}
	unscaled(i) = rad * occ::units::ANGSTROM_TO_BOHR;
	result(i) = result(i) * rad * occ::units::ANGSTROM_TO_BOHR;
    }

    occ::log::debug("DRACO scaled radii results:");
    occ::log::debug("{:>4s} {:>4s} {:>12s} {:>12s} {:>12s} {:>12s}", "idx", "sym", "charge", "cn", "scaled", "smd");
    for(int i = 0; i < atoms.size(); i++) {
	occ::log::debug("{:4d} {:>4s} {: 12.5f} {: 12.5f} {: 12.5f} {: 12.5f}",
		       i, core::Element(atoms[i].atomic_number).symbol(),
		       charges(i), cn(i),
		       result(i) * occ::units::BOHR_TO_ANGSTROM, unscaled(i) * occ::units::BOHR_TO_ANGSTROM);
    }
    return result;
}

}
