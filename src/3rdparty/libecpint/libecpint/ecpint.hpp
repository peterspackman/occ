/* 
 *      Copyright (c) 2020 Robert Shaw
 *		This file is a part of Libecpint.
 *
 *      Permission is hereby granted, free of charge, to any person obtaining
 *      a copy of this software and associated documentation files (the
 *      "Software"), to deal in the Software without restriction, including
 *      without limitation the rights to use, copy, modify, merge, publish,
 *      distribute, sublicense, and/or sell copies of the Software, and to
 *      permit persons to whom the Software is furnished to do so, subject to
 *      the following conditions:
 *
 *      The above copyright notice and this permission notice shall be
 *      included in all copies or substantial portions of the Software.
 *
 *      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *      EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *      NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *      LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *      OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *      WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef ECPINT_HEAD
#define ECPINT_HEAD

#include <vector>
#include <array>
#include "multiarr.hpp"
#include "gaussquad.hpp"
#include "ecp.hpp"
#include "bessel.hpp"
#include "radial.hpp"
#include "angular.hpp"
#include "gshell.hpp"

#include "config.hpp"

namespace libecpint {

/// returns the index of the basis function with y,z angular momentum components l,m
#define N_INDEX(l, m) (((l+m)*(l+m+1))/2 + m)

	/** 
	* \class ECPIntegral
	* \brief Calculates ECP integrals. 
	* 
	* Given an ECP basis, and orbital bases, this will calculate the ECP integrals over all ECP centers. 
	* 
	* REFERENCES: 
	* (Shaw2017) R. A. Shaw, J. G. Hill, J. Chem. Phys. 147 (2017), 074108
	* (Flores06) R. Flores-Moreno et al., J. Comput. Chem. 27 (2006), 1009
	* (MM81) L. E. McMurchie and E. R. Davidson, J. Comp. Phys. 44 (1981), 289 - 301
	*/
	class ECPIntegral
	{
	private:
		RadialIntegral radInts; ///< The interface to the radial integral calculation
		AngularIntegral angInts; ///< The angular integrals, which can be reused over all ECP centers

    	static constexpr double tolerance = 1e-12;
	
		/// Worker functions for calculating binomial expansion coefficients
		double calcC(int a, int m, double A) const;
		
		/// Array of function pointers to generated integral evaluators in qgen
		static void(*QGEN[LIBECPINT_MAX_L+1][LIBECPINT_MAX_L+1][LIBECPINT_MAX_L+1])(
		    const ECP&, const GaussianShell&, const GaussianShell&,
        const FiveIndex<double>&, const FiveIndex<double>&,
        const TwoIndex<double>&, const TwoIndex<double>&,
        double, double,
        const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&,
        ThreeIndex<double>&);

	public:
		int skipped, zero, nonzero;
		
		/** 
		  * Constructs the coefficients in the binomial expansion (see REF. Shaw2017)
		  * @param C - reference to a FiveIndex array to store the results in
		  * @param L - maximum angular momentum to go up to in expansion
		  * @param A - xyz coordinates for the center to calculate over
		  */ 
		void makeC(FiveIndex<double> &C, int L, const double *A) const;
		
		/**
		  * Creates an ECP integrator, initialising the radial and angular parts for subsequent calculations.
		  * @param maxLB - the maximum angular momentum in the orbital basis
		  * @param maxLU - the maximum angular momentum in the ECP basis
		  * @param deriv - the maximum order of derivative to be calculated (TODO: derivs currently being implemented)
		  * @param tol - the tolerance for convergence of radial integrals (defaults to 1e-15)
		  * @param small - the maximum number of quadrature points for the small radial integration grid (default 256, minimum recommended)
		  * @param large - the maximum number of quadrature points for the large radial integration grid (default 1024, minimum recommended)
		  */ 
		ECPIntegral(int maxLB, int maxLU, int deriv=0,
					double thresh = 1e-15, unsigned smallGrid = 256, unsigned bigGrid = 1024);
	
		/**
		  * Calculates the type 1 integrals for the given ECP center over the given shell pair, using quadrature
		  * @param U - reference to ECP
		  * @param shellA - the first basis shell (rows in values)
		  * @param shellB - the second basis shell (cols in values)
		  * @param data - wrapper for data about shell pair
		  * @param CA - binomial expansion coefficients for shellA, made with makeC
		  * @param CB - binomial expansion coefficients for shellB, made with makeC
		  * @param parameters - pre-calculated parameters for the radial integral
		  * @param values - array in which results are returned
		  */
		void type1(const ECP& U, const GaussianShell &shellA, const GaussianShell &shellB,
               const ShellPairData &data, const FiveIndex<double> &CA, const FiveIndex<double> &CB,
               const RadialIntegral::Parameters & parameters, TwoIndex<double> &values) const;
		
		/**
		  * Calculates the type 2 integrals for the given ECP center over the given shell pair
		  * @param l - angular momentum shell of ECP to calculate over
		  * @param U - reference to ECP
		  * @param shellA - the first basis shell (rows in values)
		  * @param shellB - the second basis shell (cols in values)
		  * @param data - wrapper for data about shell pair
		  * @param CA - binomial expansion coefficients for shellA, made with makeC
		  * @param CB - binomial expansion coefficients for shellB, made with makeC
		  * @param parameters - pre-calculated parameters for the radial integral
		  * @param values - array in which results are returned
		  */
		void type2(int l,
               const ECP& U, const GaussianShell &shellA, const GaussianShell &shellB,
               const ShellPairData &data, const FiveIndex<double> &CA, const FiveIndex<double> &CB,
               const RadialIntegral::Parameters & parameters, ThreeIndex<double> &values) const;
		
		void estimate_type2(
        const ECP& U, const GaussianShell &shellA, const GaussianShell &shellB,
        const ShellPairData &data, double* results) const;
	
		/**
		  * Computes the overall ECP integrals over the given ECP center and shell pair. This is the lower level API, where you want finer control
		  * over the calculation.
		  * Results are returned with rows corresponding to shellA and cols to shellB, with the Cartesian functions in alpha order e.g.
		  * {xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz} l = 3
		  * 
		  * @param U - reference to the ECP to calculate the integral over
		  * @param shellA - the first basis shell (rows in values) 
		  * @param shellB - the second basis shell (cols in values)
		  * @param values - reference to TwoIndex array where the results will be stored
		  */ 
		void compute_shell_pair(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        TwoIndex<double> &values, int shiftA = 0, int shiftB = 0) const;
		
		/**
	 	  * Computes the overall ECP integral first derivatives over the given ECP center, C, and shell pair (A | B) 
		  * The results are placed in order [Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz] and are calculated so that each component
		  * can always be added to the relevant total derivative. E.g. if A = B, then the contribution to the total derivative
		  * for that coordinate on the x axis will be Ax + Bx. 
		  * The order for each derivative matrices matches that specified in compute_shell_pair
	 	  * 
	 	  * @param U - reference to the ECP
	 	  * @param shellA - the first basis shell (rows in values) 
	 	  * @param shellB - the second basis shell (cols in values)
	 	  * @param results - reference to array of 9 TwoIndex arrays where the results will be stored
	 	  */ 
		void compute_shell_pair_derivative(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        std::array<TwoIndex<double>, 9> &results) const;
		
		/**
 		  * Computes the overall ECP integral second derivatives over the given ECP center, C, and shell pair (A | B) 
		  * The results are placed in order [AA, AB, AC, BB, BC, CC] with components [xx, xy, xz, yy, yz, zz] for AA, BB, and CC,
		  * and [xx, xy, xz, yx, yy, yz, zx, zy, zz] for AB, AC, and BC. As for the first derivatives, the components are calculated
		  * such that they can usually be added to the relevant total derivative. However, this is more complicated than for first derivatives,
		  * especially in the instance where A=B. It's recommended to look at the compute_second_derivatives interface in api.cpp for how
		  * to handle this. 
		  * The order for each derivative matrices matches that specified in compute_shell_pair
 		  * 
 		  * @param U - reference to the ECP
 		  * @param shellA - the first basis shell (rows in values) 
 		  * @param shellB - the second basis shell (cols in values)
 		  * @param results - reference to array of 45 TwoIndex arrays where the results will be stored
 		  */ 
		void compute_shell_pair_second_derivative(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        std::array<TwoIndex<double>, 45> &results) const;
		
		/** 
		  * Worker function to calculate the derivative of the integral <A | C | B> with respect to A. 
		  * This is given as <d_q A(l_q) | C | B> = l_q*<A(l_q-1) | C | B> - 2*mu*<A(l_q + 1) | C | B>
		  * where l_q is the angular momentum component of A in the q coordinate, and mu is the exponent of A.
		  * 
		  * @param U - reference to the ECP	
 		  * @param shellA - the first basis shell (rows in values) 
 		  * @param shellB - the second basis shell (cols in values)
 		  * @param results - reference to array of 3 TwoIndex arrays for the [x, y, z] derivatives
		  */
		void left_shell_derivative(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        std::array<TwoIndex<double>, 3> &results) const;
		
		/** 
		  * Worker function to calculate the second derivatives of the integral <A | C | B> with respect to AA. 
		  * This is given as <d_p d_q A(l_p, l_q) | C | B> = l_p*l_q*<A(l_p-1, l_q-1) | C | B> - 2*mu*l_p*<A(l_p-1, l_q+1) | C | B>
		  * - 2*mu*l_q*<A(l_p+1, l_q-1) | C | B> + 4*mu^2*<A(l_p+1, l_q+1) | C | B >
		  * 
		  * @param U - reference to the ECP
		  * @param shellA - the first basis shell (rows in values) 
		  * @param shellB - the second basis shell (cols in values)
		  * @param results - reference to array of 6 TwoIndex arrays for the [xx, xy, xz, yy, yz, zz] derivatives
		  */
		void left_shell_second_derivative(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        std::array<TwoIndex<double>, 6> &results) const;
		
		/** 
		  * Worker function to calculate the second derivatives of the integral <A | C | B> with respect to AB. 
		  * This is given as <d_p A(l_p) | C | d_q B(l_q)> = l_p*l_q*<A(l_p-1) | C | B(l_q-1)> - 2*mu_B*l_p*<A(l_p-1) | C | B(l_q+1)>
		  * - 2*mu_A*l_q*<A(l_p+1) | C | B(l_q-1)> + 4*mu_A*mu_B*<A(l_p+1) | C | B(l_q+1) > 
		  * 
		  * @param U - reference to the ECP
		  * @param shellA - the first basis shell (rows in values) 
		  * @param shellB - the second basis shell (cols in values)
		  * @param results - reference to array of 9 TwoIndex arrays for the [xx, xy, xz, yx, yy, yz, zx, zy, zz] derivatives
		  */
		void mixed_second_derivative(
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        std::array<TwoIndex<double>, 9> &results) const;
		
	};

}
#endif
