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

#include "gshell.hpp"

namespace libecpint {

	GaussianShell::GaussianShell(double *A, const int _l) : centerVec(A), l(_l), local_ptr(false), min_exp(100.0) {}
	GaussianShell::GaussianShell(const std::array<double, 3> & A, const int _l) : l(_l) {
		centerVec = localCenter;
		local_ptr = true;
		localCenter[0] = A[0];
		localCenter[1] = A[1];
		localCenter[2] = A[2];
		min_exp = 100.0;
	}

	void GaussianShell::addPrim(const double e, const double c) {
		exps.push_back(e);
		coeffs.push_back(c);
		min_exp = e < min_exp ? e : min_exp;
	}

}
