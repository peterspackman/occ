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

#include "qgen.hpp"
#include "ecpint.hpp"

namespace libecpint {

	void(*ECPIntegral::QGEN[LIBECPINT_MAX_L+1][LIBECPINT_MAX_L+1][LIBECPINT_MAX_L+1])(
      const ECP&, const GaussianShell&, const GaussianShell&,
      const FiveIndex<double>&, const FiveIndex<double>&, const TwoIndex<double>&, const TwoIndex<double>&,
      double, double,
	    const RadialIntegral&, const AngularIntegral&, const RadialIntegral::Parameters&, ThreeIndex<double>&) {

		{ 			{qgen::Q0_0_0, qgen::Q0_0_1, qgen::Q0_0_2, qgen::Q0_0_3, qgen::Q0_0_4, qgen::Q0_0_5},
			{qgen::Q0_1_0, qgen::Q0_1_1, qgen::Q0_1_2, qgen::Q0_1_3, qgen::Q0_1_4, qgen::Q0_1_5},
			{qgen::Q0_2_0, qgen::Q0_2_1, qgen::Q0_2_2, qgen::Q0_2_3, qgen::Q0_2_4, qgen::Q0_2_5},
			{qgen::Q0_3_0, qgen::Q0_3_1, qgen::Q0_3_2, qgen::Q0_3_3, qgen::Q0_3_4, qgen::Q0_3_5},
			{qgen::Q0_4_0, qgen::Q0_4_1, qgen::Q0_4_2, qgen::Q0_4_3, qgen::Q0_4_4, qgen::Q0_4_5},
			{qgen::Q0_5_0, qgen::Q0_5_1, qgen::Q0_5_2, qgen::Q0_5_3, qgen::Q0_5_4, qgen::Q0_5_5}
		},
		{ 			{qgen::Q0_1_0, qgen::Q0_1_1, qgen::Q0_1_2, qgen::Q0_1_3, qgen::Q0_1_4, qgen::Q0_1_5},
			{qgen::Q1_1_0, qgen::Q1_1_1, qgen::Q1_1_2, qgen::Q1_1_3, qgen::Q1_1_4, qgen::Q1_1_5},
			{qgen::Q1_2_0, qgen::Q1_2_1, qgen::Q1_2_2, qgen::Q1_2_3, qgen::Q1_2_4, qgen::Q1_2_5},
			{qgen::Q1_3_0, qgen::Q1_3_1, qgen::Q1_3_2, qgen::Q1_3_3, qgen::Q1_3_4, qgen::Q1_3_5},
			{qgen::Q1_4_0, qgen::Q1_4_1, qgen::Q1_4_2, qgen::Q1_4_3, qgen::Q1_4_4, qgen::Q1_4_5},
			{qgen::Q1_5_0, qgen::Q1_5_1, qgen::Q1_5_2, qgen::Q1_5_3, qgen::Q1_5_4, qgen::Q1_5_5}
		},
		{ 			{qgen::Q0_2_0, qgen::Q0_2_1, qgen::Q0_2_2, qgen::Q0_2_3, qgen::Q0_2_4, qgen::Q0_2_5},
			{qgen::Q1_2_0, qgen::Q1_2_1, qgen::Q1_2_2, qgen::Q1_2_3, qgen::Q1_2_4, qgen::Q1_2_5},
			{qgen::Q2_2_0, qgen::Q2_2_1, qgen::Q2_2_2, qgen::Q2_2_3, qgen::Q2_2_4, qgen::Q2_2_5},
			{qgen::Q2_3_0, qgen::Q2_3_1, qgen::Q2_3_2, qgen::Q2_3_3, qgen::Q2_3_4, qgen::Q2_3_5},
			{qgen::Q2_4_0, qgen::Q2_4_1, qgen::Q2_4_2, qgen::Q2_4_3, qgen::Q2_4_4, qgen::Q2_4_5},
			{qgen::Q2_5_0, qgen::Q2_5_1, qgen::Q2_5_2, qgen::Q2_5_3, qgen::Q2_5_4, qgen::Q2_5_5}
		},
		{ 			{qgen::Q0_3_0, qgen::Q0_3_1, qgen::Q0_3_2, qgen::Q0_3_3, qgen::Q0_3_4, qgen::Q0_3_5},
			{qgen::Q1_3_0, qgen::Q1_3_1, qgen::Q1_3_2, qgen::Q1_3_3, qgen::Q1_3_4, qgen::Q1_3_5},
			{qgen::Q2_3_0, qgen::Q2_3_1, qgen::Q2_3_2, qgen::Q2_3_3, qgen::Q2_3_4, qgen::Q2_3_5},
			{qgen::Q3_3_0, qgen::Q3_3_1, qgen::Q3_3_2, qgen::Q3_3_3, qgen::Q3_3_4, qgen::Q3_3_5},
			{qgen::Q3_4_0, qgen::Q3_4_1, qgen::Q3_4_2, qgen::Q3_4_3, qgen::Q3_4_4, qgen::Q3_4_5},
			{qgen::Q3_5_0, qgen::Q3_5_1, qgen::Q3_5_2, qgen::Q3_5_3, qgen::Q3_5_4, qgen::Q3_5_5}
		},
		{ 			{qgen::Q0_4_0, qgen::Q0_4_1, qgen::Q0_4_2, qgen::Q0_4_3, qgen::Q0_4_4, qgen::Q0_4_5},
			{qgen::Q1_4_0, qgen::Q1_4_1, qgen::Q1_4_2, qgen::Q1_4_3, qgen::Q1_4_4, qgen::Q1_4_5},
			{qgen::Q2_4_0, qgen::Q2_4_1, qgen::Q2_4_2, qgen::Q2_4_3, qgen::Q2_4_4, qgen::Q2_4_5},
			{qgen::Q3_4_0, qgen::Q3_4_1, qgen::Q3_4_2, qgen::Q3_4_3, qgen::Q3_4_4, qgen::Q3_4_5},
			{qgen::Q4_4_0, qgen::Q4_4_1, qgen::Q4_4_2, qgen::Q4_4_3, qgen::Q4_4_4, qgen::Q4_4_5},
			{qgen::Q4_5_0, qgen::Q4_5_1, qgen::Q4_5_2, qgen::Q4_5_3, qgen::Q4_5_4, qgen::Q4_5_5}
		},
		{ 			{qgen::Q0_5_0, qgen::Q0_5_1, qgen::Q0_5_2, qgen::Q0_5_3, qgen::Q0_5_4, qgen::Q0_5_5},
			{qgen::Q1_5_0, qgen::Q1_5_1, qgen::Q1_5_2, qgen::Q1_5_3, qgen::Q1_5_4, qgen::Q1_5_5},
			{qgen::Q2_5_0, qgen::Q2_5_1, qgen::Q2_5_2, qgen::Q2_5_3, qgen::Q2_5_4, qgen::Q2_5_5},
			{qgen::Q3_5_0, qgen::Q3_5_1, qgen::Q3_5_2, qgen::Q3_5_3, qgen::Q3_5_4, qgen::Q3_5_5},
			{qgen::Q4_5_0, qgen::Q4_5_1, qgen::Q4_5_2, qgen::Q4_5_3, qgen::Q4_5_4, qgen::Q4_5_5},
			{qgen::Q5_5_0, qgen::Q5_5_1, qgen::Q5_5_2, qgen::Q5_5_3, qgen::Q5_5_4, qgen::Q5_5_5}
		}
	};
}
