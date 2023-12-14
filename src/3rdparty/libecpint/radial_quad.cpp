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

#include "radial.hpp"
#include "mathutil.hpp"
#include <iostream>
#include <cmath>

namespace libecpint {

	RadialIntegral::RadialIntegral() {}

	void RadialIntegral::init(int maxL, double tol, int small, int large) {
		bigGrid.initGrid(large, ONEPOINT);
		primGrid.initGrid(128, ONEPOINT); 
		smallGrid.initGrid(small, TWOPOINT);
		smallGrid.transformZeroInf();
	
		bessie.init(maxL, 1600, 200, tol);
	
		tolerance = tol;
	}

	void RadialIntegral::buildBessel(
	    const std::vector<double> &r, const int nr, const int maxL, TwoIndex<double> &values, const double weight) const {
		std::vector<double> besselValues(maxL+1, 0.0);
		if (std::abs(weight) < 1e-15) {
			for (int i = 0; i < nr; i++) {
				values(0, i) = 1.0;
				for (int l = 1; l <= maxL; l++) values(l, i) = 0.0;
			}
		} else {
			for (int i = 0; i < nr; i++) {
				bessie.calculate(weight * r[i], maxL, besselValues);
				for (int l = 0; l <= maxL; l++) values(l, i) = besselValues[l];
			}
		}
	}

	double RadialIntegral::calcKij(
	    const double Na, const double Nb, const double zeta_a, const double zeta_b, const double R2) const {
		double muij = zeta_a * zeta_b / (zeta_a + zeta_b);
		return Na * Nb * std::exp(-muij * R2);
	}

	// Assumes that p is the pretabulated integrand at the abscissae
	double RadialIntegral::integrand(const double r, const double *p, const int ix) {
		return p[ix];
	}

	RadialIntegral::Parameters RadialIntegral::buildParameters(
	    const GaussianShell &shellA, const GaussianShell &shellB, const ShellPairData &data) const {
		int npA = shellA.nprimitive();
		int npB = shellB.nprimitive();

		// Initialise result object
		Parameters result;
		auto & p = result.p;
		auto & P = result.P;
		auto & P2 = result.P2;
		auto & K = result.K;

		p.assign(npA, npB, 0.0);
		P.assign(npA, npB, 0.0);
		P2.assign(npA, npB, 0.0);
		K.assign(npA, npB, 0.0);

		double Pvec[3];
		double zetaA, zetaB;
		for (int a = 0; a < npA; a++) {
			zetaA = shellA.exp(a);
		
			for (int b = 0; b < npB; b++) {
				zetaB = shellB.exp(b);
			
				p(a, b) = zetaA + zetaB;
				for (int n = 0; n < 3; n++) 
					Pvec[n] = (zetaA * data.A[n] + zetaB * data.B[n])/p(a, b);
			
				P2(a, b) = Pvec[0]*Pvec[0] + Pvec[1]*Pvec[1] + Pvec[2]*Pvec[2];
				P(a, b) = std::sqrt(P2(a, b));
				K(a, b) = calcKij(1.0, 1.0, zetaA, zetaB, data.RAB2);
			
			}
		}
		return result;
	}

	void RadialIntegral::buildU(
	    const ECP &U, const int l, const int N, const GCQuadrature &grid, double *Utab) const {
		int gridSize = grid.getN();
    const std::vector<double> &gridPoints = grid.getX();
	
		// Tabulate weighted ECP values
		double r;
		for (int i = 0; i < gridSize; i++) {
			r = gridPoints[i];
			Utab[i] = FAST_POW[N+2](r) * U.evaluate(r, l);
		}
	}

	int RadialIntegral::integrate(
      const int maxL, const int gridSize, const TwoIndex<double> &intValues, GCQuadrature &grid,
      std::vector<double> &values, const int start, const int end, const int offset, const int skip) const {
		std::function<double(double, const double*, int)> intgd = integrand;
		values.assign(maxL+1, 0.0);
		int test;
		double params[gridSize];
		for (int i = 0; i < start; i++) params[i] = 0.0;
		for (int i = end+1; i < gridSize; i++) params[i] = 0.0;
		for (int l = offset; l <= maxL; l+=skip) {
			for (int i = start; i <= end; i++) params[i] = intValues(l, i);
			const auto integral_and_test =
			    grid.integrate(intgd, params, tolerance, start, end);
			values[l] = integral_and_test.first;
			test = integral_and_test.second;
			if (test == 0) break;
		}
		return test;
	}

	void RadialIntegral::type1(
      const int maxL, const int N, const int offset,
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      const ShellPairData &data, const Parameters & parameters, TwoIndex<double> &values) const {
		int npA = shellA.nprimitive();
		int npB = shellB.nprimitive();
	
		int gridSize = bigGrid.getN();

		const auto & p = parameters.p;
		const auto & P = parameters.P;
		const auto & P2 = parameters.P2;
		const auto & K = parameters.K;

		// Now pretabulate integrand
		TwoIndex<double> intValues(maxL+1, gridSize, 0.0);
		// and bessel function
		TwoIndex<double> besselValues(maxL+1, gridSize, 0.0);
		// Calculate type1 integrals
		double da, db, za, zb, val;
		double A = data.Am;
		double B = data.Bm;
		std::vector<double> tempValues;
		values.assign(maxL+1, 2*maxL + 1, 0.0);
	
		// Tabulate integrand
		double x, phi, Px, Py;
		for (int a = 0; a < npA; a++) {
			da = shellA.coef(a);
			za = shellA.exp(a);
		
			for (int b = 0; b < npB; b++) {
				db = shellB.coef(b);
				zb = shellB.exp(b);
			
				// Reset grid starting points
				GCQuadrature newGrid = bigGrid;
				newGrid.transformRMinMax(p(a, b), (za * A + zb * B)/p(a, b));
				std::vector<double> &gridPoints = newGrid.getX();
				auto start = 0;
				auto end = gridSize - 1;
			
				// Build U and bessel tabs
				double Utab[gridSize];
				buildU(U, U.getL(), N, newGrid, Utab);
				buildBessel(gridPoints, gridSize, maxL, besselValues, 2.0*p(a,b)*P(a,b));
			
				// Start building intvalues, and prescreen
				bool foundStart = false, tooSmall = false;
				for (int i = 0; i < gridSize; i++) {
					for (int l = offset; l <= maxL; l+=2) {
						intValues(l, i) = Utab[i] * besselValues(l, i); 
						tooSmall = tooSmall || (intValues(l, i) < tolerance);
					}
					if (!tooSmall && !foundStart) {
						foundStart = true; 
						start = i;
					}
					if (tooSmall && foundStart) {
						end = i-1;
						break;
					}
				}
			
				for (int i = start; i <= end; i++) {
					val = -p(a, b) * (gridPoints[i]*(gridPoints[i] - 2*P(a, b)) + P2(a, b));
					val = std::exp(val);
					for (int l = offset; l <= maxL; l+=2)
						intValues(l, i) *= val;
				}

				int test = integrate(maxL, gridSize, intValues, newGrid, tempValues, start, end, offset, 2);
				if (test == 0) std::cerr << "Failed to converge" << std::endl;
				
				// Calculate real spherical harmonic
				x = std::abs(P(a, b)) < 1e-12 ? 0.0 : (za * data.A[2] + zb * data.B[2]) / (p(a, b) * P(a, b));
				Py = (za * data.A[1] + zb * data.B[1]) / p(a, b);
				Px = (za * data.A[0] + zb * data.B[0]) / p(a, b);
				phi = std::atan2(Py, Px);

				TwoIndex<double> harmonics = realSphericalHarmonics(maxL, x, phi);
				for (int l = offset; l <= maxL; l+=2) {
					for (int mu = -l; mu <= l; mu++)
						values(l, l+mu) += da * db * harmonics(l, l+mu) * K(a, b) * tempValues[l];
				}
			}
		}
		//std::cout << "\n\n";
	}

	// F_a(lam, r) = sum_{i in a} d_i K_{lam}(2 zeta_a A r)*std::exp(-zeta_a(r - A)^2)
	void RadialIntegral::buildF(
      const GaussianShell &shell, const double A, const int lstart, const int lend,
      const std::vector<double> &r, const int nr, const int start, const int end,
      TwoIndex<double> &F) const {
		int np = shell.nprimitive();
		
		double weight, zeta, c;
		TwoIndex<double> besselValues(lend+1, nr, 0.0);
	
		F.assign(lend + 1, nr, 0.0);
		for (int a = 0; a < np; a++) {
			zeta = shell.exp(a);
			c = shell.coef(a);
			weight = 2.0 * zeta * A;
		
			buildBessel(r, nr, lend, besselValues, weight);
		
			for (int i = start; i <= end; i++) {
				weight = r[i] - A;
				weight = c * std::exp(-zeta * weight * weight);
			
				for (int l = lstart; l <= lend; l++) 
					F(l, i) += weight * besselValues(l, i); 
			}
		}
	}
	
	double RadialIntegral::estimate_type2(
      const int N, const int l1, const int l2, const double n,
      const double a, const double b, const double A, const double B) const {
		double kA = 2.0*a*A;
		double kB = 2.0*b*B;
		double c0 = std::max(N - l1 - l2, 0);
		double c1_min = kA + kB;
		double p = a + b + n;

		double P = c1_min + std::sqrt(c1_min*c1_min + 8.0*p*c0);
		P /= (4.0*p);

		double zA = P - A; 
		double zB = P - B;
		double besselValue1 = bessie.upper_bound(kA * P, l1);
		double besselValue2 = bessie.upper_bound(kB * P, l2);
		double Fres = FAST_POW[N](P) * std::exp(-n * P * P - a * zA * zA - b * zB * zB) * besselValue1 * besselValue2;
		return (0.5 * std::sqrt(M_PI/p) * Fres * (1.0 + std::erf(std::sqrt(p)*P)));
	}

	void RadialIntegral::type2(
      const int l, const int l1start, int l1end, const int l2start, int l2end,
      const int N, const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      const ShellPairData &data, const Parameters & parameters, TwoIndex<double> &values) const {
	
		std::function<double(double, const double*, int)> intgd = integrand;

		int npA = shellA.nprimitive();
		int npB = shellB.nprimitive();
	
		double A = data.Am;
		double B = data.Bm;

		const auto & p = parameters.p;
		const auto & P = parameters.P;
		const auto & P2 = parameters.P2;
		const auto & K = parameters.K;

		// Start with the small grid
		// Pretabulate U
		int gridSize = smallGrid.getN();
		const std::vector<double> &gridPoints = smallGrid.getX();
	
		// Reset grid starting points
		const auto start = 0;
		const auto end = gridSize-1;
	
		double Utab[gridSize];
		buildU(U, l, N, smallGrid, Utab);
		values.assign(l1end+1, l2end+1, 0.0);
	
		// Build the F matrices
		if (A < 1e-15) l1end = 0; 
		if (B < 1e-15) l2end = 0; 
		TwoIndex<double> Fa;
		TwoIndex<double> Fb;
		buildF(shellA, data.Am, l1start, l1end, gridPoints, gridSize, start, end, Fa);
		buildF(shellB, data.Bm, l2start, l2end, gridPoints, gridSize, start, end, Fb);
	
		// Build the integrals
		bool foundStart, tooSmall;
		std::vector<int> tests((l1end +1) * (l2end+1));
		double params[gridSize]; 
		bool failed = false;
		int ix = 0;
		for (int l1 = 0; l1 <= l1end; l1++) {
			int l2start = (l1 + N) % 2;
			for (int l2 = l2start; l2 <= l2end; l2+=2) {
				
				for (int i = 0; i < gridSize; i++) params[i] = Utab[i] * Fa(l1, i) * Fb(l2, i);
				const auto this_integral_and_test = smallGrid.integrate(intgd, params, tolerance, start, end);
				tests[ix] = this_integral_and_test.second;
				failed = failed || (tests[ix] == 0);
				values(l1, l2) = tests[ix] == 0 ? 0.0 : this_integral_and_test.first;
				ix++;
			}
		}
	
		if (failed) {
			// Not converged, switch to big grid
			double zeta_a, zeta_b, c_a, c_b;
				
			gridSize = bigGrid.getN();
			Fa.assign(l1end+1, gridSize, 0.0);
			Fb.assign(l2end+1, gridSize, 0.0);
		
			for (int a = 0; a < npA; a++) {
				c_a = shellA.coef(a);
				zeta_a = shellA.exp(a);
			
				for (int b = 0; b < npB; b++) {
					c_b = shellB.coef(b);
					zeta_b = shellB.exp(b);
				
					GCQuadrature newGrid = bigGrid;
					newGrid.transformRMinMax(p(a, b), (zeta_a * A + zeta_b * B)/p(a, b));
					std::vector<double> &gridPoints2 = newGrid.getX();
					const auto start = 0;
					const auto end = gridSize - 1;
			
					// Build U and bessel tabs
					double Utab2[gridSize];
					buildU(U, l, N, newGrid, Utab2);
					buildBessel(gridPoints2, gridSize, l1end, Fa, 2.0*zeta_a*A);
					buildBessel(gridPoints2, gridSize, l2end, Fb, 2.0*zeta_b*B);
				
					double Xvals[gridSize];
					double ria, rib;
					for (int i = 0; i < gridSize; i++) {
						ria = gridPoints2[i] - A;
						rib = gridPoints2[i] - B;
						Xvals[i] = std::exp(-zeta_a*ria*ria -zeta_b*rib*rib) * Utab2[i];
					}
				
					double params2[gridSize]; 
					ix = 0;
					for (int l1 = 0; l1 <= l1end; l1++) {
						int l2start = (l1 + N) % 2; 
						
						for (int l2 = l2start; l2 <= l2end; l2+=2) {
						
							if (tests[ix] == 0) {
								for (int i = 0; i < gridSize; i++)
									params2[i] = Xvals[i] * Fa(l1, i) * Fb(l2, i);
								const auto integral_and_test =
								    newGrid.integrate(intgd, params2, tolerance, start, end);
								if (!integral_and_test.second) std::cerr << "Failed at second attempt" << std::endl;
								values(l1, l2) += c_a * c_b * integral_and_test.first;
							}
							ix++; 
						
						}
					}
				
				}
			}
		
		}
	}

}
