/* 
*      Copyright (c) 2020 Robert Shaw
*		  This file is a part of Libecpint.
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

#ifndef QGEN_HEAD
#define QGEN_HEAD

/** 
* \file qgen.hpp
* \brief Generated header of generated integral functions
*/

#include "multiarr.hpp"
#include "radial.hpp"
#include "angular.hpp"
#include "gshell.hpp"
#include "ecp.hpp"

namespace libecpint {
	namespace qgen {

		/** 
		*  For integral functions that have not been unrolled, this loops over the angular and radial integrals to provide the overall ECP integrals,
		*  returning in an array indexed as (shellA, shellB, mu), for mu = -lam .. lam
		*
		*  @param lam - angular momentum of ECP shell
		*  @param LA - angular momentum of basis shellA
		*  @param LB - angular momentum of basis shellB
		*  @param radials - array of radial integrals, indexed as (N, lA, lB)
		*  @param CA - binomial expansion coefficients of shellA
		*  @param CB - binomial expansion coefficients of shellB
		*  @param SA - spherical harmonics of shellA
		*  @param SB - spherical harmonics of shellB
		*  @param angints - object containing the angular integrals
		*  @param values - the array in which the results are returned
		*/
		void rolled_up(int lam, int LA, int LB, const ThreeIndex<double>& radials, const FiveIndex<double>& CA, const FiveIndex<double>& CB, const TwoIndex<double>& SA, const TwoIndex<double>& SB, const AngularIntegral& angints, ThreeIndex<double>& values);

		/** 
		*  As per rolled_up, but for the special case where shell A is located on the ECP (see Shaw2017 supp. info.)
		*  per symmetry, the shell B on ECP case is the same with A and B variables reversed.  
		*
		*  @param lam - angular momentum of ECP shell
		*  @param LA - angular momentum of basis shellA
		*  @param LB - angular momentum of basis shellB
		*  @param radials - array of radial integrals, indexed as (N, lA, lB)
		*  @param CB - binomial expansion coefficients of shellB
		*  @param SB - spherical harmonics of shellB
		*  @param angints - object containing the angular integrals
		*  @param values - the array in which the results are returned
		*/
		void rolled_up_special(int lam, int LA, int LB, const ThreeIndex<double>& radials, const FiveIndex<double>& CB, const TwoIndex<double>& SB, const AngularIntegral& angints, ThreeIndex<double>& values);
		
	void Q0_0_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_0_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_0_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_0_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_0_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_0_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_1_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_1_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_2_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_2_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_2_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_3_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_3_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_3_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_3_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_4_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_4_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_4_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_4_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_4_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q0_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q1_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q2_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q3_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q4_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_0(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_1(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_2(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_3(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_4(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);
	void Q5_5_5(const ECP&, const GaussianShell&, const GaussianShell&, const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&, double, double, const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&);

}
}
#endif
