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

#ifndef API_HEAD
#define API_HEAD

#include <vector>
#include <array>
#include <string>
#include <memory>
#include "ecpint.hpp"
#include "multiarr.hpp"

namespace libecpint {

/// returns the index for the ij-th second derivative for a system with N atoms
#define H_START(i, j, N) (9*j + 3*(3*N-1)*i - (9*i*(i+1))/2 - 3)
	
	constexpr double TWO_C_TOLERANCE = 1E-12;
	
	/** \struct ECPIntegrator
	  * \brief API object that stores and handles all data for computing ECP integrals and their derivatives.
	  * 
	  * This is a higher level interface than directly using the ECPIntegral objects to calculate
	  * individual integrals. To use it, you follow these steps:
	  *  1) specify the gaussian basis 
	  *  2) specify the ecp basis (either as a stream or from the library)
	  *  3) initialise the integrator, specifying derivative order
	  *  4) compute the desired quantities
	  *
	  *  The order of results follows the order of the atoms/basis functions as you
	  *  specify them in steps 1 and 2. If you update the positions of your atoms at any point
	  *  you must update them through this interface. Results are obtained as a pointer to a 
	  *  stream of data. 
	  */
	struct ECPIntegrator {
		std::vector<GaussianShell> shells; ///< a container of the Gaussian basis shells
		ECPBasis ecps; ///< a container for the ECPs
		std::shared_ptr<ECPIntegral> ecpint; ///< pointer to the ECP integral engine
		int maxLB; ///< the maximum angular momentum in the gaussian basis, determined by set_gaussian_basis
		int deriv; ///< maximum derivative order to be calculated (defaults to 0)
		int ncart; ///< total number of cartesian gaussians in the gaussian basis, determined by set_gaussian_basis
		int natoms; ///< total number of distinct atoms, determined during init
		double min_alpha; ///< the minimum exponent in the gaussian basis
		
		bool ecp_is_set; ///< true if the ecp basis has been set, false by default
		bool basis_is_set; ///< true if the gaussian basis has been set, false by default
		
		/** Container for the calculated ECP integrals, in canonical Cartesian order
		  * i.e. for L=2 : {x^2, xy, xz, y^2, yz, z^2}
		  */
		TwoIndex<double> integrals; 
		
		/** Container for the ECP 1st derivatives, matrices are in same order as integrals,
		  * order in the vector is {Ax, Ay, Az, Bx, By, Bz, ... } where atom order {A, B, C, ...}
		  * is the same as provided when calling set_gaussian_basis. Total length is 3*natoms
		  */
		std::vector<TwoIndex<double>> first_derivs;
		
		/** Container for the ECP 2nd derivatives, matrices are in same order as integrals,
		  * order in the vector is {AA, AB, AC, ..., BB, BC, ..., CC, ...} where atom order {A, B, C, ...}
		  * is the same as provided when calling set_gaussian_basis. The coordinate order within this is
		  * {xx, xy, xz, yy, yz, zz} for AA, BB, CC, etc. and
		  * {xx, xy, xz, yx, yy, yz, zx, zy, zz} for AB, AC, BC, etc.
		  * Total length is therefore: 3*natoms*(3*natoms+1)/2
		  */
		std::vector<TwoIndex<double>> second_derivs; 
		
		/** Default constructor, sets default values
		  */
		ECPIntegrator() { ecp_is_set = basis_is_set = false; maxLB = ncart = 0; }
		
		/**  Constructs a basis set of GaussianShell objects from a stream of coordinates, exponents, coefficients,
		  *  and angular momenta. Determines maxLB and ncart, and sets basis_is_set to true. The order of the atoms 
		  *  in coords determines the order of atoms in computed derivatives.
		  * 
		  *  @param nshells - the number of angular momentum shells in the basis
		  *  @param coords  - a stream of cartesian coordinates (in bohr) in xyz order for each shell (size should be 3*nshells)
		  *  @param exponents - a stream of primitive exponents for each shell (size should be sum of shell_lengths)
		  *  @param coefs - a stream of coefficients corresponding to each exponent in exponents
		  *  @param ams - the angular momentum of each shell (size should be nshells)
		  *	 @param shell_lengths - the number of primitives in each shell (size should be nshells)
		  */
		void set_gaussian_basis(int nshells, const double* coords, const double* exponents, const double* coefs, const int* ams, const int* shell_lengths);
		
		/**  Constructs an ECPBasis with ECP objects for each ECP from streams of data. Determines maxLU. 
		  *  The order of the atoms doesn't matter, and ids will be matched to the order from set_gaussian_basis.
		  * 
		  *  @param necps - the number of ECPs
		  *  @param coords  - a stream of cartesian coordinates (in bohr) in xyz order for each ECP (size should be 3*necps)
		  *  @param exponents - a stream of primitive exponents for each ECP (size should be sum of shell_lengths)
		  *  @param coefs - a stream of coefficients corresponding to each exponent in exponents
		  *  @param ams - the angular momentum of each primitive in exponents/coefs
		  *  @param ns  - the order of r multiplying each primitive - we follow the convention where 2 is the default            
		  *	 @param shell_lengths - the number of primitives in each ECP (size should be necps)
		  */
		void set_ecp_basis(int necps, const double* coords, const double* exponents, const double* coefs, const int* ams, const int* ns, const int* shell_lengths);
		
#ifdef HAS_PUGIXML
		/**  Constructs an ECPBasis with ECP objects for each ECP, from the built-in ECP library.
		  *  The order of the atoms doesn't matter, and ids will be matched to the order from set_gaussian_basis.
		  *  It will search for the file  "share_dir + / + name + .xml"     
		  * 
		  *  @param necps - the number of ECPs
		  *  @param coords  - a stream of cartesian coordinates (in bohr) in xyz order for each ECP (size should be 3*necps)
		  *  @param charges - the atomic numbers of each ECP atom (in same order as coords, size necps)
		  *  @param names - the name of each ECP, in same order as charges, e.g. "ecp10mdf" (size necps)
		  *  @param share_dir - the location of the share directory with the ecp library (typically "PATH/share/libecpint/xml")
		  */
		void set_ecp_basis_from_library(int necps, const double* coords, const int* charges, const std::vector<std::string> & names, const std::string & share_dir);
#endif
		
		/**  Updates the positions of the GaussianShells.
		  *  The order of the coordinates must match that when originally specified in set_gaussian_basis.
		  * 
		  *  @param nshells - the number of angular momentum shells in the basis - must match nshells from set_gaussian_basis
		  *  @param coords  - a stream of cartesian coordinates (in bohr) in xyz order for each shell (size should be 3*nshells)
		  */
		void update_gaussian_basis_coords(int nshells, const double* coords);
		
		/**  Updates the positions of the ECPs
		  *  The order of the coordinates must match that when originally specified in set_ecp_basis/set_ecp_basis_from_library
		  * 
		  *  @param necps - the number of ECPs
		  *  @param coords - a stream of cartesian coordinates (in bohr) in xyz order for each ECP (size should be 3*necps)
		  */
		void update_ecp_basis_coords(int necps, const double* coords);
		
		/** Initialises the ECPIntegral object, and determines the atom ids for each GaussianShell and ECP. 
		  * This must be called AFTER the ECP/Gaussian bases are set, but BEFORE calling any of the compute functions.
		  *
		  * @param deriv_ - the maximum derivative order to be computed; affects whether compute_first/second_derivs can be called; default 0
		  */
		void init(int deriv_ = 0);
		
		/** Computes the ECP integrals across all shell pairs, returning the results into the integrals matrix. 
		  * The order of the shells is canonical cartesian order, and matches the order in which the shells were specified 
		  * in set_gaussian_basis. 
		  */
		void compute_integrals();
		
		/** Computes the first derivative of the ECP integrals with respect to each atomic coordinate, placing the results
		  * in first_derivs. The atom order matches that specified in set_gaussian_basis, in xyz order.
		  */ 
		void compute_first_derivs();
		
		/** Computes the second derivative of the ECP integrals with respect to each pair of atomic coordinates,  
		  * but taking into account symmetry of second derivatives. The atom order matches that specified in set_gaussian_basis.
		  * See the docs for second_derivs for detailed description of the order. 
		  */ 
		void compute_second_derivs();
		
		/** @return a shared pointer to the underlying data for integrals 
		  * The packing is such that M(i, j) = i*ncart + j.
		  */
		std::shared_ptr<std::vector<double>> get_integrals() { return std::make_shared<std::vector<double>>(integrals.data); }
		
		/** @return a vector (size 3*natoms) of shared pointers to the data for first_derivs
		  * the packing is the same as for get_integrals, and the order is Ax,Ay,Az,Bx,By,Bz, etc.
		  */ 
		std::vector<std::shared_ptr<std::vector<double>>> get_first_derivs() {
			std::vector<std::shared_ptr<std::vector<double>>> results;
			for (auto& v : first_derivs) results.push_back(std::make_shared<std::vector<double>>(v.data));
			return results;
		}
		
		/** @return a vector (size 3*natoms*(3*natoms+1)/2) of shared pointers to the data for second_derivs
	 	 * the packing is the same as for get_integrals, and the order is that specified in the docs for second_derivs
	  	 */ 
		std::vector<std::shared_ptr<std::vector<double>>> get_second_derivs() {
			std::vector<std::shared_ptr<std::vector<double>>> results;
			for (auto& v : second_derivs) results.push_back(std::make_shared<std::vector<double>>(v.data));
			return results;
		}
	};
	
	double shell_bound(int la, double alpha, double A2, double eta);
	
}

#endif
