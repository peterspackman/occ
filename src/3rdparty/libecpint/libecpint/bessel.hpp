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

#ifndef BESSEL_FUNCTION_HEAD
#define BESSEL_FUNCTION_HEAD

#include <vector>

namespace libecpint {

	constexpr double SMALL = 1.0E-7; ///< Numerical tolerance of z below which K_n(z) = { 1 (n=0), 0 otherwise }
	constexpr int TAYLOR_CUT = 5; ///< Order of local Taylor series to be used in Bessel function expansion

	/**
	 *  \class BesselFunction
	 *  \brief Computes a modified spherical Bessel function of the first kind.
	 *  
	 *  Uses pretabulation to calculate the Bessel function up to a given maximum
	 *  angular momentum. Values are interpolated using local Taylor series.
	 *
	 *  REFERENCES:
	 *  R. Flores-Moreno et al., J. Comput. Chem. 27 (2006), 1009
	 *  L.E. McMurchie, E. Davidson, J. Comput. Phys. 44 (1981), 289 
	 */
	class BesselFunction 
	{
	private:
		int lMax; ///< Maximum angular momentum
		int N; ///< Number of abscissae
		int order; ///< Order to which the Bessel series is expanded
		double scale; ///< N/16.0
	
		std::vector<std::vector<double>> K; ///< Bessel function values
		std::vector<std::vector<std::vector<double>>> dK; ///< Bessel function derivatives
		std::vector<double> C; ///< Coefficients of derivatives of the Bessel function
	
		/**
		* Pretabulates the Bessel function (and derivs) to a given accuracy.
		* @param accuracy - the tolerance at which a value is considered converged
		* @return zero if successful, -1 if not converged
		*/
		int tabulate(double accuracy);
	
	public:
		/// Default constructor. Creates a blank object.
		BesselFunction();
		
		/// Specified constructor. Will call init with given arguments.
		BesselFunction(int lMax, int N, int order, double accuracy);
		
	
		/**
		* Initialises and pretabulates the BesselFunction up to the given angular momentum. 
		* @param lMax - the maximum angular momentum needed
		* @param N - the maximum number of points to be used in pretabulation, suggested 1600
		* @param order - the order at which the expansion is cut off, suggested 200
		* @param accuracy - the tolerance below which a value is considered converged
		*/
		void init(int lMax, int N, int order, double accuracy);
	
		/**
		* Calculates the Bessel function values at a given point up to a given angular momentum
		* @param z - point at which to evaluate
		* @param maxL - maximum angular momentum needed; must be <= lMax for object
		* @param values - reference to vector in which to put the values for l = 0 to maxL
		*/
		void calculate(double z, int maxL, std::vector<double> &values) const;
		
		/**
		* Calculates the Bessel function value at a given point for a single angular momentum
		* @param z - point at which to evaluate
		* @param L - angular momentum needed; must be <= lMax for object
		*/
		double calculate(double z, int L) const;
		
		/**
		* Calculates an upper bound to the Bessel function value at a given point for a given angular momentum
		* @param z - point at which to evaluate
		* @param L - angular momentum needed; must be <= lMax for object
		*/
		double upper_bound(double z, int L) const;
	};

}
#endif
