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

#ifndef ANGULAR_HEAD
#define ANGULAR_HEAD

#include <vector>
#include "multiarr.hpp"

namespace libecpint {
	
	/** 
	* \class AngularIntegral
	* \brief Calculates and stores the angular integrals needed for ECP integration.
	* 
	* This should not usually be created directly, it is instead owned by an ECPIntegral object,
	* so that integrals can be performed over multiple ECP centers without duplicating work.
	*/
	class AngularIntegral 
	{
	private: 
		int LB; ///< Maximum angular momentum of orbital basis
		int LE; ///< Maximum angular momentum of ECP basis

		int wDim; ///< Limits for the w-integral indices
		int maxL; ///< Limits for the angular momentum indices
	
		FiveIndex<double> W; ///< Stores the type 1 angular integrals
		SevenIndex<double> omega; ///< Stores the type 2 angular integrals
	
		/// Worker functions for calculating terms in the USP to spherical transformation coefficients
		double calcG(int l, int m) const;
		double calcH1(int i, int j, int l, int m) const;
		double calcH2(int i, int j, int k, int m) const;
	
		/**
		* Calculates the polynomial integrals, int (x^i y^j z^k dOm) where dOm is the solid angle differential
		* @param maxI - the maximum power, i, to determine
		* @return ThreeIndex of polynomials P(i, j, k), where strictly i >= j >= k
		*/
		ThreeIndex<double> Pijk(int maxI) const; 
	
		/**
		* Builds the type 1 angular integrals
		* @param U - the USP to spherical transformation coefficients
		*/
		void makeW(const FiveIndex<double> &U);
		/** 
		* Builds the type 2 angular integrals
		* @param U - the USP to spherical transformation coefficients
		*/
		void makeOmega(const FiveIndex<double> &U);
	
	public:
	
		/**
		* Calculates all possible USP to spherical transformation coefficients for a given angular momentum
		* @param lam - the angular momentum
		* @param mu - the subshell
		* @return ThreeIndex of the values U_lam,mu(k, l, m)
		*/
		ThreeIndex<double> uklm(int lam, int mu) const;
	
		/**
		* Builds the USP to spherical transformation coefficients for use in calculating the type 1 and 2 integrals
		* @return FiveIndex of the coefficients U(lam, lam+mu, k, l, m)
		*/
		FiveIndex<double> makeU() const;
	
		/// Default constructor creates empty object
		AngularIntegral(); 
		
		/**
		 *  Specified constructor calls init with given arguments
		 * @param LB - the maximum angular momentum of the orbital basis
		 * @param LE - the maximum angular momentum of the ECP basis
		 */
		AngularIntegral(int LB, int LE); 
		
		/**
		* Initialises the object, must be called before anything else if default constructor was used.
		* @param LB - the maximum angular momentum of the orbital basis
		* @param LE - the maximum angular momentum of the ECP basis
		*/
		void init(int LB, int LE);
		
		/**
		* Computes the type 1 and 2 angular integrals
		*/
		void compute();
	
		/// TODO: Clears the W and omega arrays
		void clear();
	
		/**
		* Returns the type 1 angular integral W(k, l, m, lam, mu)
		* @param k - x index
		* @param l - y index
		* @param m - z index
		* @param lam - angular momentum
		* @param mu - subshell
		* @return value of type 1 angular integral
		*/
		double getIntegral(int k, int l, int m, int lam, int mu) const; 
		
		/**
		* Returns the type 2 angular integral Omega(k, l, m, lam, mu, rho, sigma)
		* @param k - x index
		* @param l - y index
		* @param m - z index
		* @param lam - angular momentum of current ECP shell
		* @param mu - subshell of lam
		* @param rho - angular momentum of current basis shell
		* @param sigma - subshell of rho
		* @return value of type 2 angular integral
		*/
		double getIntegral(int k, int l, int m, int lam, int mu, int rho, int sigma) const;
		
		int* getOmegaMults() { return omega.mults; }
    const int* getOmegaMults() const { return omega.mults; }
		int* getOmegaDims() { return omega.dims; }
    const int* getOmegaDims() const { return omega.dims; }
		std::vector<double>& getOmegaData() { return omega.data; }
    const std::vector<double>& getOmegaData() const { return omega.data; }

		
		/// is W(k, l, m, lam, mu) zero to within a given tolerance?
		bool isZero(int k, int l, int m, int lam, int mu, double tolerance) const;
		
		/// is Omega(k, l, m, lam, mu, rho, sigma) zero to within a given tolerance?
		bool isZero(int k, int l, int m, int lam, int mu, int rho, int sigma, double tolerance) const;	
	};

}
#endif
