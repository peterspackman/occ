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

#ifndef RADIAL_HEAD
#define RADIAL_HEAD

#include <vector>
#include "multiarr.hpp"
#include "gaussquad.hpp"
#include "ecp.hpp"
#include "bessel.hpp"
#include "gshell.hpp"

namespace libecpint {

	constexpr double MIN_EXP = 0.002;
	/** 
	* \class RadialIntegral
	* \brief Abstracts the calculation of radial integrals for ECP integration.
	* 
	* This should not be used directly, and is owned by ECPIntegral.
	* It provides the interface to the adaptive quadrature algorithms used to calculate the type 1 and 2 radial integrals,
	* and the unrolled recursive scheme for type 2 radial integrals. 
	*/
	class RadialIntegral
	{
	private:
		/// The larger integration grid for type 1 integrals, and for when the smaller grid fails for type 2 integrals 
		GCQuadrature bigGrid;
		/// The smaller integration grid, default for the type 2 integrals
		GCQuadrature smallGrid;
		/// Even smaller grid for primitive integrals
		GCQuadrature primGrid; 
		/// Modified spherical Bessel function of the first kind
		BesselFunction bessie;

		/// Tolerance for change below which an integral is considered converged
		double tolerance;
	
		/// This integrand simply returns the pretabulated integrand values stored in p given an index ix
		static double integrand(double r, const double *p, int ix);

		/**
		* Builds a matrix of Bessel at the given points up to the given maximum angular momentum. 
		* @param r - vector of points to evaluate at
		* @param nr - number of points in r (for convenience)
		* @param maxL - the maximum angular momentum needed
		* @param values - TwoIndex<double> to store the values in
		* @param weight - factor to weight r by (defaults to 1)
		*/
		void buildBessel(const std::vector<double> &r, int nr, int maxL, TwoIndex<double> &values, double weight = 1.0) const;
	
		double calcKij(double Na, double Nb, double zeta_a, double zeta_b, double R2) const;
	
		/**
		* Tabulate r^{N+2} times the ECP for all quadrature points. 
		* @param U - the ECP to be pretabulated
		* @param l - the angular momentum shell of the ECP to be used
		* @param N - the power of r to weight the ECP by
		* @param grid - the quadrature grid to be used
		* @param Utab - the array to put the values into.
		*/
		void buildU(const ECP &U, const int l, const int N, const GCQuadrature &grid, double *Utab) const;
	
		/**
		* Tabulate the F function values for the default mode of calculating type 2 integrals.
		* @param shell - the shell of orbital basis functions to tabulate over
		* @param lstart - the lowest angular momentum needed
		* @param lend - the maximum angular momentum needed
		* @param r - quadrature grid points
		* @param nr - the number of grid points in r
		* @param start - the grid point to start at
		* @param end - the grid point to stop at
		* @param F - the matrix to put the values in
		*/
		void buildF(
        const GaussianShell &shell, double A, int lstart, int lend,
        const std::vector<double> &r, int nr, int start, int end,
        TwoIndex<double> &F) const;
	
		/**
		* Performs the integration given the pretabulated integrand values. 
		* @param maxL - the maximum angular momentum needed
		* @param gridSize - the number of quadrature points
		* @param intValues - the TwoIndex<double> of pretabulated integrand values for each angular momentum needed
		* @param grid - the quadrature grid
		* @param values - the vector to put the resulting integrals into
		* @param start - index of first point to integrate over
		* @param end - index of last point to integrate over
		* @param offset - the angular momentum to start at (defaults to 0)
		* @param skip - the steps of angular momentum to go up in (defaults to 1)
		*/
		int integrate(int maxL, int gridSize, const TwoIndex<double> &intValues, GCQuadrature &grid, std::vector<double> &values,
                int start, int end, int offset = 0, int skip = 1) const;
		
		/** 
		  * Computes the base integrals needed for the recursive type 2 integration. See ref. Shaw2017 for details. 
		  * @param N_min - the lowest index of base integral needed (usually 2)
		  * @param N_max - the highest index of base integral needed
		  * @param p - the sum of shellA, shellB, and ECP exponents (p = a + b + n)
		  * @param o_root_p = 1/sqrt(p)
		  * @param P1 = (a * A + b * B) / p where A, B are the magnitudes of distance of shellA and shellB from ECP
		  * @param P2  = (b * B - a * A) / p
		  * @param P1_2 = P1^2
		  * @param P2_2 = P2^2
		  * @param X1 = exp(p * P1^2 - a*A^2*b*B^2) / (16.0 * a * A * b * B) 
		  * @param X2 = exp(p * P2^2 - a*A^2*b*B^2) / (16.0 * a * A * b * B)
		  * @param oP1 = 1 / P1 (P1 can't be zero as a, A, b, B, p > 0)
		  * @param oP2 = 1 / P2 (set to zero if P2 = 0)
		  * @param values - array of size N_max - N_min + 1 in which results are returned
		  */
		void compute_base_integrals(int N_min, int N_max, double p, double o_root_p, double P1,
			 						double P2, double P1_2, double P2_2, double X1, double X2,
									double oP1, double oP2, double* values) const;
									
		/**
		  * Performs a single radial quadrature over a small grid
		  * @param N - power of r in integrand
		  * @param l1 - angular momentum of first Bessel function
		  * @param l2 - angular momentum of second Bessel function
		  * @param n - exponent of ECP
		  * @param a - exponent of primitive in shellA
		  * @param b - exponent of primitive in shellB
		  * @param A - magnitude of distance of shellA from ECP
		  * @param B - magnitude of distance of shellB from ECP
		  * @return a pair, where the first is the integral value, and the second is true if integration converged
		  */
		std::pair<double, bool> integrate_small(
		    int N, int l1, int l2, double n, double a, double b, double A, double B) const;
		
	public:
		/// Default constructor creates an empty object
		RadialIntegral();
	
		/**
		* Initialises the object, in turn intialising the quadrature grids and BesselFunction
		* @param maxL - the maximum angular momentum of integral needed
		* @param tol - the tolerance for convergence of integrals (defaults to 1e-15)
		* @param small - the maximum number of quadrature points for the small integration grid (default 256, minimum recommended)
		* @param large - the maximum number of quadrature points for the large integration grid (default 1024, minimum recommended)
		*/
		void init(int maxL, double tol = 1e-15, int small = 256, int large = 1024);

    /// struct to store all parameters needed in both type 1 and 2 integrations
		struct Parameters {
      /// Matrices of parameters needed in both type 1 and 2 integrations
      TwoIndex<double> p, P, P2, K;
		};
		/**
		* Given two GaussianShells, builds the parameters needed by both kind of integral. 
		* @param shellA - the first GaussianShell
		* @param shellB - the second GaussianShell
		* @param data - the data container for the shell pair
		* @returns the parameters needed in both type 1 and 2 integrations
		*/
		Parameters buildParameters(
		    const GaussianShell &shellA, const GaussianShell &shellB, const ShellPairData &data) const;
	
		/**
		* Calculates all type 1 radial integrals over two Gaussian shells up to the given maximum angular momentum.
		* @param maxL - the maximum angular momentum
		* @param N - the power of r that the integrand is weighted by
		* @param offset - the starting angular momentum
		* @param U - the ECP to be integrated over
		* @param shellA - the first GaussianShell
		* @param shellB - the second GaussianShell
		* @param data - the data container for the shell pair
		* @param parameters - pre-calculated parameters for the radial integral
		* @param values - the matrix to return the integrals in
		*/
		void type1(int maxL, int N, int offset,
               const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
               const ShellPairData &data, const Parameters & parameters, TwoIndex<double> &values) const;
	
		/**
		* Calculates all type 2 radial integrals over two Gaussian shells for the given ECP angular momentum l using quadrature
		* @param lam - the ECP shell angular momentum to be calculated over
		* @param l1start - the angular momentum to start on for the first shell
		* @param l1end - the angular momentum to stop at for the first shell
		* @param l2start - the angular momentum to start on for the second shell
		* @param l2end - the angular momentum to stop at for the second shell
		* @param N - the power of r that the integrand is weighted by
		* @param U - the ECP to be integrated over
		* @param shellA - the first GaussianShell
		* @param shellB - the second GaussianShell
		* @param data - the data container for the shell pair
		* @param parameters - pre-calculated parameters for the radial integral
		* @param values - the matrix to return the integrals in
		*/
		void type2(int lam, int l1start, int l1end, int l2start, int l2end, int N,
               const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
               const ShellPairData &data, const Parameters & parameters, TwoIndex<double> &values) const;

		/**
		* Calculates all the requested type 2 radial integrals using predominantly a recursive algorithm. 
		* In the triples, l1 must be less than or equal to l2. Symmetry means that for l1 > l2, {N, l1, l2} can be calculated as
		* {N, l2, l1} but with shellA and shellB (And therefore also A and B) swapped. 
		* 
		* @param triples - vector of triples of form {N, l1, l2} of all required radial integrals
		* @param nbase - the maximum number of base integrals that will be needed (so only have to compute once)
		* @param lam - the ECP shell angular momentum to be calculated over
		* @param U - the ECP to be integrated over
		* @param shellA - the first GaussianShell
		* @param shellB - the second GaussianShell
		* @param A - the magnitude of the distance of shellA from the ECP
		* @param B - the magnitude of the distance of shellB from the ECP
		* @param radials - the array to return the integrals in, indexed as (N, l1, l2)
		*/
		void type2(
        const std::vector<Triple> &triples, int nbase, int lam,
        const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
        double A, double B, ThreeIndex<double> &radials) const;
	
		/**
		  * Estimates the value of the requested type 2 radial integral for prescreening, as described in ref. Shaw2017.
		  * The modal point is estimated by ignoring the ratios of bessel function derivatives - this gives an overestimate
		  * so is okay for screening, but would not be good for approximating the integral itself.
		  *
		  * @param N - power of r in integrand
		  * @param l1 - angular momentum of first Bessel function
		  * @param l2 - angular momentum of second Bessel function
		  * @param n - exponent of ECP
		  * @param a - exponent of primitive in shellA
		  * @param b - exponent of primitive in shellB
		  * @param A - magnitude of distance of shellA from ECP
		  * @param B - magnitude of distance of shellB from ECP
		  * @return estimated value (upper bound) of the type 2 integral 
		  */
		double estimate_type2(int N, int l1, int l2, double n, double a, double b, double A, double B) const;
	};

}

#endif
