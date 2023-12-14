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

#ifndef GC_QUAD_HEAD
#define GC_QUAD_HEAD

#include <functional>
#include <vector>

namespace libecpint {

	/// Different choices of integration algorithm, see references
	enum GCTYPE {
		ONEPOINT, ///< Described in Perez92
		TWOPOINT  ///< Described in Perez93
	};

	/** 
	* \class GCQuadrature
	* \brief Performs adaptive Gauss-Chebyshev quadrature of the second kind for any given function.
	* 
	* Stores the weights and abscissae for the quadrature, and provides two different methods to integrate on [-1, 1] 
	* Also contains means to transform the region of integration to [0, infinity) and [rmin, rmax]
	*
	* REFERENCES:
	* (Perez92) J.M. Perez-Jorda et al., Comput. Phys. Comm. 70 (1992), 271-284
	* (Perez93) J.M. Perez-Jorda et al., Comput. Phys. Comm. 77 (1993), 46-56
	* (Krack98) M. Krack, A.M. Koster, J. Chem. Phys. 108 (1998), 3226 - 3234
	* (Flores06) R. Flores-Moreno et al., J. Comput. Chem. 27 (2006), 1009-1019
	*/
	class GCQuadrature {
	private:
		int maxN; ///< Maximum number of points to use in quadrature
		int M; 	///< Index of midpoint
	
		std::vector<double> x; ///< Weights
		std::vector<double> w; ///< Abscissae
 
		double I; ///< Integral value
	
		GCTYPE t; ///< Algorithm type to be used
	
		/// Worker function for integration routines, should not be called directly.	
		double sumTerms(const std::function<double(double, const double*, int)> &f,
                  const double *p, int limit, int start, int end, int shift, int skip) const;

	public:
		
		/// Default constructor, creates empty object
		GCQuadrature();
		
		/// Copy constructor, carbon copies all members
		GCQuadrature(const GCQuadrature &other);
	
		/**
		* Intialises the integration grid to the given number of points, and integration type. 
		* ONEPOINT will choose N = 2^n - 1 closest to the given number of points, whilst
		* TWOPOINT will choose N= 3*2^n - 1 in the same way.
		*
		* @param points - maximum number of quadrature points to be used
		* @param t - the algorithm to be used (ONEPOINT / TWOPOINT)
		*/
		void initGrid(int points, GCTYPE t);
	
		/**
		* Integrates the given function (over [-1, 1] by default) to within the given tolerance. 
		* @param f - the function to be integrated
		* @param params - array of parameters for the function to be integrated
		* @param tolerance - change below which convergenced is considered to be achieved
		* @param start - the index of the first point used in the integration
		* @param end - the index of the last point used in the integration
		* @returns the integral (first) and true if integration converged, false otherwise (second)
		*/
		std::pair<double, bool> integrate(
		    std::function<double(double, const double*, int)> &f,
		    const double *params, double tolerance, int start, int end) const;
	
		/**
		* Transforms the region of integration to [0, inf) using the logarithmic transformation of Krack98
		*/
		void transformZeroInf();
		
		/**
		* Transforms region of integration to [rmin, rmax] using the linear transformation from Flores06, assuming 
		* a Gaussian envelope. rmin/rmax are the distances from the centre of the envelope such that the integrand is effectively zero.
		* @param z - the exponent of the Gaussian envelope
		* @param p - the centre of the Gaussian envelope
		*/
		void transformRMinMax(double z, double p);  // Transfromation from [-1, 1] to [rmin, rmax] from Flores06
		void untransformRMinMax(double z, double p);
	
		/// @return the maximum number of quadrature points
		int getN() const { return maxN; }
	
		/// @return a reference to the abscissae
    std::vector<double>& getX() { return x; }
    const std::vector<double>& getX() const { return x; }
	};
}

#endif
