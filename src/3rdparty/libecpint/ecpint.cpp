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

#include "ecpint.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include "mathutil.hpp"
#include "qgen.hpp"
#include <cassert>

namespace libecpint {

	ECPIntegral::ECPIntegral(int maxLB, int maxLU, int deriv,
							 double thresh, unsigned smallGrid, unsigned bigGrid) {
		// Make sure library can perform requested integrals
		assert(maxLB+deriv <= LIBECPINT_MAX_L); 
		assert(maxLU <= LIBECPINT_MAX_L);
		
		// Initialise singletons
		initFactorials();
		zero = nonzero = skipped = 0;
		
		// Initialise angular and radial integrators
		angInts.init(maxLB + deriv, maxLU);
		angInts.compute();
#ifdef DEBUG
		std::cout << "Initializing ECP radial integrator: thresh = " << thresh << "  grids: "
				<< smallGrid << "/" << bigGrid << std::endl;
#endif
		radInts.init(2*(maxLB+deriv) + maxLU, thresh, smallGrid, bigGrid);
	};

	double ECPIntegral::calcC(const int a, const int m, const double A) const {
		double value = 1.0 - 2*((a-m) % 2);
		value *= std::pow(A, a-m);
		value *= FAC[a]/(FAC[m] * FAC[a-m]);
		return value;
	}

	void ECPIntegral::makeC(FiveIndex<double> &C, const int L, const double *A) const {
		int z; double Ck, Cl;
		int na = 0;
		for (int x = L; x >= 0; x--) {
			for (int y = L-x; y >= 0; y--) {
				z = L - x - y;
			
				for (int k = 0; k<= x; k++) {
					Ck = calcC(x, k, A[0]);
					for (int l = 0; l <= y; l++) {
						Cl = calcC(y, l, A[1]);
						for (int m = 0; m <= z; m++) C(0, na, k, l, m) = Ck * Cl * calcC(z, m, A[2]);
					}
				}
				na++;
			}
		}
	}

	void ECPIntegral::type1(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      const ShellPairData &data, const FiveIndex<double> &CA, const FiveIndex<double> &CB,
      const RadialIntegral::Parameters & parameters, TwoIndex<double> &values) const {

		int LA = data.LA; int LB = data.LB;
		int maxLBasis = data.maxLBasis;
	
		// Build radial integrals
		int L = LA + LB;
		TwoIndex<double> temp;
		ThreeIndex<double> radials(L+1, L+1, 2*L+1);
		for (int ix = 0; ix <= L; ix++) {
			radInts.type1(ix, ix, ix % 2, U, shellA, shellB, data, parameters, temp);
			for(int l = 0; l <= ix; l++) {
				for (int m = -l; m <= l; m++) radials(ix, l, l+m) = temp(l, l+m);
			}
		}
	
		// Unpack positions
		double Ax = data.A[0]; double Ay = data.A[1]; double Az = data.A[2];
		double Bx = data.B[0]; double By = data.B[1]; double Bz = data.B[2];
	
		// Calculate chi_ab for all ab in shells
		int z1, z2, lparity, mparity, msign, ix, k, l, m;
		double C;
		int na = 0, nb = 0;
		for (int x1 = LA; x1 >= 0; x1--) {
			for (int y1 = LA-x1; y1 >= 0; y1--) {
				z1 = LA - x1 - y1;
				nb = 0;
			
				for (int x2 = LB; x2 >= 0; x2--) {
					for (int y2 = LB-x2; y2 >= 0; y2--) {
						z2 = LB - x2 - y2;
					
						for (int k1 = 0; k1 <= x1; k1++) {
							for (int k2 = 0; k2 <= x2; k2++) {
								k = k1 + k2;
							
								for (int l1 = 0; l1 <= y1; l1++) {
									for (int l2 = 0; l2 <= y2; l2++) {
										l = l1 + l2;
									
										for (int m1 = 0; m1 <= z1; m1++) {
											for (int m2 = 0; m2 <= z2; m2++){
												m = m1 + m2;
												C = CA(0, na, k1, l1, m1) * CB(0, nb, k2, l2, m2);
												if ( fabs(C) > 1e-14 ) {
													// Build radial integrals
													ix = k + l + m;
													
													// Certain terms can be neglected as the angular integrals will always be zero
													// See Flores06 appendix for details. 
													lparity = ix % 2; 
													msign = 1 - 2*(l%2);
													mparity = (lparity + m) % 2;
												
													for (int lam = lparity; lam <= ix; lam+=2) {
														for (int mu = mparity; mu <= lam; mu+=2)
															values(na, nb) += C * angInts.getIntegral(k, l, m, lam, msign*mu) * radials(ix, lam, lam+msign*mu);
													}
								
												}
											}
										}
									}
								}
							}
						}
					
						values(na, nb) *= 4.0 * M_PI;
						nb++;
					}
				}
			
				na++;
			}
		}
	
	}

	void ECPIntegral::type2(
      const int lam, const ECP& U, const GaussianShell &shellA, const GaussianShell &shellB,
      const ShellPairData &data, const FiveIndex<double> &CA, const FiveIndex<double> &CB,
      const RadialIntegral::Parameters & parameters, ThreeIndex<double> &values) const {
	
		// Unpack some data for convenience
		int LA = data.LA;
		int LB = data.LB;
		int L = LA + LB;	
		int maxLBasis = data.maxLBasis;
	
		double Am = data.Am; double Bm = data.Bm;

		if (data.A_on_ecp && data.B_on_ecp) {
			
			// Both on ECP, simplest case - see Shaw2017 supplementary material
			double prefactor = 4.0 * M_PI; 
			int npA = shellA.nprimitive();
			int npB = shellB.nprimitive();
			int npC = U.getN(); 
		
			double zA, zB, zC, dA, dB, dC, p; 
			int nC, z1, z2;
			
			int na = 0;
			for (int x1 = LA; x1 >= 0; x1--) {
				for (int r1 = LA-x1; r1 >= 0; r1--) {
					z1 = LA - x1 - r1; 
			
					int nb = 0;
					for (int x2 = LB; x2 >= 0; x2--) {
						for (int y2 = LB - x2; y2 >= 0; y2--) {
							z2 = LB - x2 - y2; 
						
							double value = 0.0;
							for (int c = 0; c < npC; c++) {
                const GaussianECP& g = U.getGaussian(c);
								if (g.l == lam) {
									zC = g.a;
									dC = g.d;
									nC = g.n; 
 
									for (int a = 0; a < npA; a++) {
										zA = shellA.exp(a);
										dA = shellA.coef(a);
									
										for (int b = 0; b < npB; b++) {
											zB = shellB.exp(b);
											dB = shellB.coef(b); 
										
											p = zA + zB + zC;
										
											double o_root_p = 1.0 / sqrt(p);
											int N = 2 + LA + LB + nC;
											value += 0.5*dA*dB*dC*GAMMA[N]*FAST_POW[N+1](o_root_p);
										}
									}
								}
							}
							
							for (int mu = -lam; mu <= lam; mu++) {
						
								double angular = prefactor * angInts.getIntegral(x1, r1, z1, lam, mu, 0, 0) * angInts.getIntegral(x2, y2, z2, lam, mu, 0, 0); 
								values(na, nb, lam+mu) = angular * value; 
							}
							nb++; 
						}
					}
				
					na++;
				}
			}
			
		} else {
			
			// At least one of the shells is not on the ECP, so spherical harmonics will be required
			
			double xA = Am > 0 ? data.A[2] / Am : 0.0;
			double xB = Bm > 0 ? data.B[2] / Bm : 0.0;
			double phiA = atan2(data.A[1], data.A[0]);
			double phiB = atan2(data.B[1], data.B[0]);
			TwoIndex<double> SA = realSphericalHarmonics(lam+LA, xA, phiA);
			TwoIndex<double> SB = realSphericalHarmonics(lam+LB, xB, phiB);
		
			if (data.A_on_ecp) {
				// Radial integrals need to be calculated by a different recursive scheme, or by quadrature
				ThreeIndex<double> radials(L+1, lam + LA + 1, lam + LB + 1); 
				TwoIndex<double> temp;
				std::fill(values.data.begin(), values.data.end(), 0.0);

				for (int N = 0; N < L+1; N++) {
					radInts.type2(lam, 0, lam + LA, 0, lam + LB, N, U, shellA, shellB, data, parameters, temp);
					for (int l1 = 0; l1 < lam + LA + 1; l1++)
						for (int l2 = 0; l2 < lam + LB + 1; l2++)
							radials(N, l1, l2) = temp(l1, l2);
				}
				
				// a significant number of terms can be neglected a priori - see Shaw2017 supplementary material. 
				qgen::rolled_up_special(lam, LA, LB, radials, CB, SB, angInts, values);
				
			} else if (data.B_on_ecp){
				// Same as above with A and B reversed
				ThreeIndex<double> radials(L+1, lam + LB + 1, lam + LA + 1); 
				ThreeIndex<double> tmpValues(values.dims[1], values.dims[0], values.dims[2]);
				std::fill(tmpValues.data.begin(), tmpValues.data.end(), 0.0);
				TwoIndex<double> temp;

				for (int N = 0; N < L+1; N++) {
					radInts.type2(lam, 0, lam + LA, 0, lam + LB, N, U, shellA, shellB, data, parameters, temp);
					for (int l1 = 0; l1 < lam + LB + 1; l1++)
						for (int l2 = 0; l2 < lam + LA + 1; l2++)
							radials(N, l1, l2) = temp(l2, l1);
				}
				
				// a significant number of terms can be neglected a priori - see Shaw2017 supplementary material. 
				qgen::rolled_up_special(lam, LB, LA, radials, CA, SA, angInts, tmpValues);
				// transcribe back into values
				for (int na = 0; na < values.dims[0]; na++)
					for (int nb = 0; nb < values.dims[1]; nb++)
						for (int nc = 0; nc < values.dims[2]; nc++)
							values(na, nb, nc) = tmpValues(nb, na, nc);
			} else {
				
				// Neither is on the ECP, the full recursive scheme with generated integrals can be used
				// Need LA <= LB, but symmetry means we can just swap the arguments if LB > LA. 
				if (LA <= LB) 
					QGEN[LA][LB][lam](U, shellA, shellB, CA, CB, SA, SB, Am, Bm, radInts, angInts, parameters, values);
				else {
					ThreeIndex<double> temp_values(data.ncartB, data.ncartA, 2*U.getL() + 1); 
					QGEN[LB][LA][lam](U, shellB, shellA, CB, CA, SB, SA, Bm, Am, radInts, angInts, parameters, temp_values);
					for (int na = 0; na < data.ncartA; na++)
						for (int nb = 0; nb < data.ncartB; nb++)
							for (int nu = 0; nu < 2*U.getL() + 1; nu++)
								values(na, nb, nu) = temp_values(nb, na, nu); 
				}
					
			}			
		}
	}

	void ECPIntegral::estimate_type2(
      const ECP& U, const GaussianShell &shellA, const GaussianShell &shellB,
      const ShellPairData &data, double* results) const {
		double sigma_a, sigma_b, min_eta, n2, an, bn, a_bound, b_bound, ab_bound;
		double atilde, btilde, ztilde, Tk, Tk_0, xp;
		
		double Na_0 = 0.5 * data.LA / M_EULER;
		double Nb_0 = 0.5 * data.LB / M_EULER;
		
		for (int l = 0; l <= U.getL(); l++) {
			min_eta = U.min_exp_l[l];
			n2 = min_eta * min_eta;
			an = shellA.min_exp + min_eta;
			bn = shellB.min_exp + min_eta;
			if (data.A2 < 1e-6) sigma_a = 0.5 * an / shellA.min_exp;
			else sigma_a = 0.5 * data.LA * an * an / (shellA.min_exp * (n2*data.A2 + data.LA * an));
			if (data.B2 < 1e-6) sigma_b = 0.5 * bn / shellB.min_exp;
			else sigma_b = 0.5 * data.LB * bn * bn / (shellB.min_exp * (n2*data.B2 + data.LB * bn));
			
			atilde = (1.0 - sigma_a) * shellA.min_exp;
			btilde = (1.0 - sigma_b) * shellB.min_exp;
			
			a_bound = 0.0;
			for (int i = 0; i < shellA.exps.size(); i++)
				a_bound += FAST_POW[data.LA](std::sqrt(Na_0 / (shellA.exps[i] * sigma_a))) * std::abs(shellA.coeffs[i]); 
			
			b_bound = 0.0;
			for (int i = 0; i < shellB.exps.size(); i++)
				b_bound += FAST_POW[data.LB](std::sqrt(Nb_0 / (shellB.exps[i] * sigma_b))) * std::abs(shellB.coeffs[i]);
			
			double Tk_0 = 2.0 * atilde * btilde * data.Am * data.Bm; 
			ab_bound = 0.0;
			xp = atilde*atilde*data.A2 + btilde*btilde*data.B2;
			for (int k = U.l_starts[l]; k < U.l_starts[l+1]; k++) {
        const GaussianECP& g = U.getGaussian(k);
				ztilde = atilde + btilde + g.a;
				Tk = Tk_0 / ztilde;
				Tk = Tk > 1 ? 0.5 * std::exp(Tk) / Tk : SINH_1;
				ab_bound += std::abs(g.d) * FAST_POW[3](std::sqrt(M_PI/g.a)) * std::exp(xp / ztilde) * Tk;
			}
			ab_bound *= std::exp(-atilde*data.A2 -btilde*data.B2);
			results[l] = (2*l+1)*(2*l+1)* a_bound * b_bound * ab_bound;
		}
	}

	void ECPIntegral::compute_shell_pair(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      TwoIndex<double> &values, const int shiftA, const int shiftB) const {
	
		ShellPairData data;
		
		// Shift A and B to be relative to U
		const double* C = U.center();
		data.A[0] = shellA.center()[0] - C[0]; 
		data.A[1] = shellA.center()[1] - C[1];
		data.A[2] = shellA.center()[2] - C[2]; 
		data.B[0] = shellB.center()[0] - C[0]; 
		data.B[1] = shellB.center()[1] - C[1];
		data.B[2] = shellB.center()[2] - C[2]; 
	
	  	// Construct data that will be reused everywhere, and takes account of derivative shifts
		data.LA = shellA.am() + shiftA; 
		data.LB = shellB.am() + shiftB;
		data.maxLBasis = data.LA > data.LB ? data.LA : data.LB;
		data.ncartA = (data.LA+1)*(data.LA+2)/2;
		data.ncartB = (data.LB+1)*(data.LB+2)/2;
	
		data.A2 = data.A[0]*data.A[0] + data.A[1]*data.A[1] + data.A[2]*data.A[2];
		data.Am = sqrt(data.A2);
		data.A_on_ecp = (data.Am < 1e-6); 
		data.B2 = data.B[0]*data.B[0] + data.B[1]*data.B[1] + data.B[2]*data.B[2];
		data.Bm = sqrt(data.B2);
		data.B_on_ecp = (data.Bm < 1e-6);
		double RAB[3] = {data.A[0] - data.B[0], data.A[1] - data.B[1], data.A[2] - data.B[2]};
		data.RAB2 = RAB[0]*RAB[0] + RAB[1]*RAB[1] + RAB[2]*RAB[2];
		data.RABm = sqrt(data.RAB2);
		
		// Prepare the radial integrator
		const auto radIntParameters = radInts.buildParameters(shellA, shellB, data);
	
		// Construct coefficients 
		FiveIndex<double> CA(1, data.ncartA, data.LA+1, data.LA+1, data.LA+1);
		FiveIndex<double> CB(1, data.ncartB, data.LB+1, data.LB+1, data.LB+1);
		makeC(CA, data.LA, data.A);
		makeC(CB, data.LB, data.B);
		
		double screens[U.getL() + 1];
		estimate_type2(U, shellA, shellB, data, screens);
	
		// Calculate type1 integrals, if necessary
		values.assign(data.ncartA, data.ncartB, 0.0);
		if (!U.noType1() && screens[U.getL()] > tolerance)
			type1(U, shellA, shellB, data, CA, CB, radIntParameters, values);
		
		std::vector<int> l_list; 
		for (int l = 0; l < U.getL(); l++) 
			if (screens[l] > tolerance) l_list.push_back(l); 
		
		// Now all the type2 integrals
		ThreeIndex<double> t2vals(data.ncartA, data.ncartB, 2*U.getL() + 1);
		for (int l : l_list) {
			t2vals.fill(0.0);
			type2(l, U, shellA, shellB, data, CA, CB, radIntParameters, t2vals);

			for (int m = -l; m <= l; m++) {
				for(int na = 0; na < data.ncartA; na++) {
					for (int nb = 0; nb < data.ncartB; nb++) {
						values(na, nb) += t2vals(na, nb, l+m);
					}
				}
			}
		}
	}
	
	void ECPIntegral::left_shell_derivative(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      std::array<TwoIndex<double>, 3> &results) const {
		int LA = shellA.am();
		int LB = shellB.am();
		
		int ncartB = (LB+1) * (LB+2) / 2;
		int ncartA = (LA+1) * (LA+2) / 2;
		int ncartA_minus = LA * (LA+1) / 2;
		TwoIndex<double> Q_minus, Q_plus; 
		
		for (auto& r : results) r.assign(ncartA, ncartB, 0.0); 
		
		if (LA != 0)
			compute_shell_pair(U, shellA, shellB, Q_minus, -1, 0); 
		
		// hack in the exponents to the coefficients
		GaussianShell tempA = shellA.copy();
		for (int i = 0; i < tempA.nprimitive(); i++) 
			tempA.coeffs[i] *= tempA.exps[i];
		compute_shell_pair(U, tempA, shellB, Q_plus, 1, 0); 
		
		// Now compile the derivatives
		if (LA != 0) {
			int nA = 0;
			int nA_minus, nA_plus;
			for (int k=LA; k >= 0; k--) {
				for (int l=LA-k; l>=0; l--) {
					int m = LA - k - l;
						
					for (int nB = 0; nB < ncartB; nB++) {
						nA_plus = N_INDEX(l, m);
						nA_minus = std::min(nA_plus, Q_minus.dims[0]-1);
						results[0](nA, nB) = -k*Q_minus(nA_minus, nB) + 2.0*Q_plus(nA_plus, nB);
						
						nA_minus = l > 0 ? N_INDEX(l-1, m) : 0;
						nA_plus  = N_INDEX(l+1, m);
						results[1](nA, nB) = -l*Q_minus(nA_minus, nB) + 2.0*Q_plus(nA_plus, nB);
						
						nA_minus = m > 0 ? N_INDEX(l, m-1) : 0;
						nA_plus  = N_INDEX(l, m+1);
						results[2](nA, nB) = -m*Q_minus(nA_minus, nB) + 2.0*Q_plus(nA_plus, nB);
					}
					nA += 1;
				}
			}
		} else {
			for (int nB = 0; nB < ncartB; nB++) {
				results[0](0, nB) = 2.0*Q_plus(0, nB);
				results[1](0, nB) = 2.0*Q_plus(1, nB);
				results[2](0, nB) = 2.0*Q_plus(2, nB);
			}
		}
	}
	
	void ECPIntegral::left_shell_second_derivative(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      std::array<TwoIndex<double>, 6> &results) const {
		int LA = shellA.am();
		int LB = shellB.am();
		
		int ncartB = (LB+1) * (LB+2) / 2;
		int ncartA = (LA+1) * (LA+2) / 2;
		int ncartA_minus = std::max(1, (LA-1) * (LA) / 2);
		TwoIndex<double> Q_minus, Q_plus, Q_0;
		
		for (auto& r : results) r.assign(ncartA, ncartB, 0.0); 
		
		if (LA > 1)
			compute_shell_pair(U, shellA, shellB, Q_minus, -2, 0); 
		else
			Q_minus.assign(ncartA_minus, ncartB, 0.0);
		
		// hack in the exponents to the coefficients
		GaussianShell tempA = shellA.copy();
		for (int i = 0; i < tempA.nprimitive(); i++) 
			tempA.coeffs[i] *= tempA.exps[i];
		compute_shell_pair(U, tempA, shellB, Q_0, 0, 0); 
		
		// and for the l+2
		for (int i = 0; i < tempA.nprimitive(); i++) 
			tempA.coeffs[i] *= tempA.exps[i];
		compute_shell_pair(U, tempA, shellB, Q_plus, 2, 0); 

		// Now compile the derivatives
		int nA = 0;
		int nA_mm, nA_pp, nA_mp, nA_pm;
		for (int k=LA; k >= 0; k--) {
			for (int l=LA-k; l>=0; l--) {
				int m = LA - k - l;
					
				for (int nB = 0; nB < ncartB; nB++) {
					nA_mp = nA_pp = N_INDEX(l, m); //dxx
					nA_mm = std::min(nA_mp, Q_minus.dims[0]-1);
					results[0](nA, nB) = k*(k-1)*Q_minus(nA_mm, nB) - 2.0*(2*k+1)*Q_0(nA_mp, nB)
										+4.0*Q_plus(nA_pp, nB);
					
					nA_pm = l > 0 ? N_INDEX(l-1, m) : 0;
					nA_mm = k > 0 ? nA_pm : 0; //dxy
					nA_pp = N_INDEX(l+1, m); 
					nA_mp  = k > 0 ? nA_pp : 0;
					results[1](nA, nB) = k*l*Q_minus(nA_mm, nB) - 2.0*k*Q_0(nA_mp, nB)
										- 2.0*l*Q_0(nA_pm, nB) + 4.0*Q_plus(nA_pp, nB);

					nA_pm = m > 0 ? N_INDEX(l, m-1) : 0;
					nA_mm = k > 0 ? nA_pm : 0; //dxz
					nA_pp = N_INDEX(l, m+1);
					nA_mp  = k > 0 ? nA_pp : 0;
					results[2](nA, nB) = k*m*Q_minus(nA_mm, nB) - 2.0*k*Q_0(nA_mp, nB)
										- 2.0*m*Q_0(nA_pm, nB) + 4.0*Q_plus(nA_pp, nB);

					nA_mm = l > 1 ? N_INDEX(l-2, m) : 0; //dyy
					nA_mp = N_INDEX(l, m);
					nA_pp  = N_INDEX(l+2,m);
					results[3](nA, nB) = l*(l-1)*Q_minus(nA_mm, nB) - 2.0*(2*l+1)*Q_0(nA_mp, nB)
										+4.0*Q_plus(nA_pp, nB);

					nA_mm = l*m > 0 ? N_INDEX(l-1, m-1) : 0; //dyz
					nA_mp = l > 0 ? N_INDEX(l-1, m+1) : 0; 
					nA_pm = m > 0 ? N_INDEX(l+1, m-1) : 0; 
					nA_pp  = N_INDEX(l+1, m+1);
					results[4](nA, nB) = l*m*Q_minus(nA_mm, nB) - 2.0*l*Q_0(nA_mp, nB)
										- 2.0*m*Q_0(nA_pm, nB) + 4.0*Q_plus(nA_pp, nB);

					nA_mm =  m > 1 ? N_INDEX(l, m-2) : 0; //dzz
					nA_mp = N_INDEX(l, m);
					nA_pp  = N_INDEX(l,m+2);
					results[5](nA, nB) = m*(m-1)*Q_minus(nA_mm, nB) - 2.0*(2*m+1)*Q_0(nA_mp, nB)
										+4.0*Q_plus(nA_pp, nB);

				}
				nA += 1;
			}
		}
	}
	
	void ECPIntegral::mixed_second_derivative(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      std::array<TwoIndex<double>, 9> &results) const {
		int LA = shellA.am();
		int LB = shellB.am();
		
		int ncartB = (LB+1) * (LB+2) / 2;
		int ncartA = (LA+1) * (LA+2) / 2;
		int ncartB_minus = std::max(1, (LB) * (LB+1) / 2);
		int ncartA_minus = std::max(1, (LA) * (LA+1) / 2);
		int ncartB_plus = (LB+2) * (LB+3) / 2;
		int ncartA_plus = (LA+2) * (LA+3) / 2;
		TwoIndex<double> Q_mm, Q_mp, Q_pm, Q_pp;
		
		for (auto& r : results) r.assign(ncartA, ncartB, 0.0); 
		
		GaussianShell tempA = shellA.copy();
		for (int i = 0; i < tempA.nprimitive(); i++) 
			tempA.coeffs[i] *= tempA.exps[i];
		GaussianShell tempB = shellB.copy();
		for (int i = 0; i < tempB.nprimitive(); i++) 
			tempB.coeffs[i] *= tempB.exps[i];
		
		if (LA > 0) {
			if (LB > 0) {
				compute_shell_pair(U, shellA, shellB, Q_mm, -1, -1); 
				compute_shell_pair(U, tempA, shellB, Q_pm, 1, -1);
			} else {
				Q_mm.assign(ncartA_minus, ncartB_minus, 0.0);
				Q_pm.assign(ncartA_plus, ncartB_minus, 0.0);
			}
			compute_shell_pair(U, shellA, tempB, Q_mp, -1, 1);
		} else if (LB > 0) {
			compute_shell_pair(U, tempA, shellB, Q_pm, 1, -1);
			Q_mm.assign(ncartA_minus, ncartB_minus, 0.0);
			Q_mp.assign(ncartA_minus, ncartB_plus, 0.0);
		} else {
			Q_mm.assign(ncartA_minus, ncartB_minus, 0.0);
			Q_mp.assign(ncartA_minus, ncartB_plus, 0.0);
			Q_pm.assign(ncartA_plus, ncartB_minus, 0.0);
		}
		compute_shell_pair(U, tempA, tempB, Q_pp, 1, 1); 

		// Now compile the derivatives
		int nA = 0;
		int nB = 0;
		int nA_m[3], nA_p[3], nB_m[3], nB_p[3], AL[3], BL[3];
		for (int ka=LA; ka >= 0; ka--) {
			for (int la=LA-ka; la>=0; la--) {
				int ma = LA - ka - la;
				AL[0]=ka; AL[1]=la; AL[2]=ma;
				nA_p[0] = N_INDEX(la, ma);
				nA_m[0] = std::min(nA_p[0], Q_mm.dims[0]-1);
				nA_m[1] = la > 0 ? N_INDEX(la-1, ma) : 0; 
				nA_m[2] = ma > 0 ? N_INDEX(la, ma-1) : 0;
				nA_p[1] = N_INDEX(la+1,ma);
				nA_p[2] = N_INDEX(la, ma+1);
				
				nB = 0;
				for (int kb=LB; kb >= 0; kb--) {
					for (int lb=LB-kb; lb>=0; lb--) {
						int mb = LB - kb - lb;
						nB_p[0] = N_INDEX(lb, mb);
						nB_m[0] = std::min(nB_p[0], Q_mm.dims[1]-1);
						nB_m[1] = lb > 0 ? N_INDEX(lb-1, mb) : 0; 
						nB_m[2] = mb > 0 ? N_INDEX(lb, mb-1) : 0;
						nB_p[1] = N_INDEX(lb+1,mb);
						nB_p[2] = N_INDEX(lb, mb+1);
						BL[0]=kb; BL[1]=lb; BL[2]=mb;

						for (int p = 0; p < 3; p++) {
							for (int q = 0; q < 3; q++) {
								results[3*p+q](nA, nB) = AL[p]*BL[q]*Q_mm(nA_m[p], nB_m[q]) - 2.0*BL[q]*Q_pm(nA_p[p], nB_m[q])
									- 2.0*AL[p]*Q_mp(nA_m[p], nB_p[q]) + 4.0*Q_pp(nA_p[p], nB_p[q]);
							}
						}
						
						nB += 1;
					}
				}
				nA += 1;
			}
		}
	}
	
	void ECPIntegral::compute_shell_pair_derivative(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      std::array<TwoIndex<double>, 9> &results) const {
		// First we check centres
		double A[3], B[3], C[3];
		for (int i = 0; i < 3; i++) {
			A[i] = shellA.center()[i];
			B[i] = shellB.center()[i];
			C[i] = U.center()[i];
		}
		
		double dAC = std::abs(A[0] - C[0]) + std::abs(A[1] - C[1]) + std::abs(A[2] - C[2]);
		double dBC = std::abs(B[0] - C[0]) + std::abs(B[1] - C[1]) + std::abs(B[2] - C[2]);
		
		// Calculate shell derivatives
		std::array<TwoIndex<double>, 3> QA, QB;
		if (dAC > 1e-6) 
			left_shell_derivative(U, shellA, shellB, QA);
		if (dBC > 1e-6)
			left_shell_derivative(U, shellB, shellA, QB);
		
		// initialise results matrices
		int ncartA = (shellA.am()+1) * (shellA.am()+2) / 2;
		int ncartB = (shellB.am()+1) * (shellB.am()+2) / 2;
		
		// Now construct the nuclear derivs
		if (dAC > 1e-6) {
			results[0] = QA[0];
			results[1] = QA[1];
			results[2] = QA[2];
			if (dBC > 1e-6) {
				results[3] = QB[0].transpose();
				results[4] = QB[1].transpose();
				results[5] = QB[2].transpose();
				for (int i = 6; i < 9; i++) results[i].assign(ncartA, ncartB, 0.0);
				for (int nA = 0; nA < ncartA; nA++) {
					for (int nB = 0; nB < ncartB; nB++){
						results[6](nA, nB) = -1.0 * (results[0](nA, nB) + results[3](nA, nB));
						results[7](nA, nB) = -1.0 * (results[1](nA, nB) + results[4](nA, nB));
						results[8](nA, nB) = -1.0 * (results[2](nA, nB) + results[5](nA, nB));
					}
				}
			} else {
			   results[3] = results[0]; results[3].multiply(-1.0);
			   results[4] = results[1]; results[4].multiply(-1.0);
			   results[5] = results[2]; results[5].multiply(-1.0);
			   for (int i = 6; i < 9; i++) results[i].assign(ncartA, ncartB, 0.0);
			}
		} else if (dBC > 1e-6) {
			results[3] = QB[0].transpose();
			results[4] = QB[1].transpose();
			results[5] = QB[2].transpose();
		   	results[0] = results[3]; results[0].multiply(-1.0);
		   	results[1] = results[4]; results[1].multiply(-1.0);
		   	results[2] = results[5]; results[2].multiply(-1.0);
			for (int i = 6; i < 9; i++) results[i].assign(ncartA, ncartB, 0.0);
		} else {
			// else everything is zero
			for (auto& r : results) r.assign(ncartA, ncartB, 0.0);
		}
	}

	void ECPIntegral::compute_shell_pair_second_derivative(
      const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB,
      std::array<TwoIndex<double>, 45> &results) const {
		// First we check centres
		double A[3], B[3], C[3];
		for (int i = 0; i < 3; i++) {
			A[i] = shellA.center()[i];
			B[i] = shellB.center()[i];
			C[i] = U.center()[i];
		}
		
		double dAC = std::abs(A[0] - C[0]) + std::abs(A[1] - C[1]) + std::abs(A[2] - C[2]);
		double dBC = std::abs(B[0] - C[0]) + std::abs(B[1] - C[1]) + std::abs(B[2] - C[2]);
		
		// Calculate shell derivatives
		std::array<TwoIndex<double>, 6> QAA, QBB;
		std::array<TwoIndex<double>, 9> QAB;

		if (dAC > 1e-6) {
			left_shell_second_derivative(U, shellA, shellB, QAA);
			if (dBC > 1e-6) {
				left_shell_second_derivative(U, shellB, shellA, QBB);
				mixed_second_derivative(U, shellA, shellB, QAB);
			}
		} else if (dBC > 1e-6) {
			left_shell_second_derivative(U, shellB, shellA, QBB);
		}
		
		// initialise results matrices
		int ncartA = (shellA.am()+1) * (shellA.am()+2) / 2;
		int ncartB = (shellB.am()+1) * (shellB.am()+2) / 2;
		for (auto& r : results) r.assign(ncartA, ncartB, 0.0);
		
		// Now construct the nuclear derivs
		int jaas[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
		int jbbs[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
		int jaa, jbb;
		if (dAC > 1e-6) {
			//AA (xx, xy, xz, yy, yz, zz)
			for (int i = 0; i < 6; i++) results[i] = QAA[i]; 
			
			if (dBC > 1e-6) {	
				// AB (xx, xy, xz, yx, yy, yz, zx, zy, zz)
				for (int i = 6; i < 15; i++) results[i] = QAB[i-6];
				 //BB (xx, xy, xz, yy, yz, zz) 
				for (int i = 24; i < 30; i++) results[i] = QBB[i-24].transpose();

				for (int nA = 0; nA < ncartA; nA++) {
					for (int nB = 0; nB < ncartB; nB++){
						for (int j = 0; j < 9; j++) {
							jaa = jaas[j];
							jbb = jbbs[j];
							
							// AC (xx, xy, xz, yx, yy, yz, zx, zy, zz)
							results[15+j](nA, nB) = -1.0*(QAA[jaa](nA, nB) + QAB[j](nA, nB));
							
							// BC (xx, xy, xz, yx, yy, yz, zx, zy, zz)
							results[30+j](nA, nB) = -1.0*(QBB[jaa](nB, nA) + QAB[jbb](nA, nB));
							
							// CC (xx, xy, xz, yy, yz, zz)
							results[39+jaa](nA, nB) = -results[30+j](nA, nB) -results[15+j](nA, nB); 
						}
					}
				}
			} else {
				// AB (xx, xy, xz, yx, yy, yz, zx, zy, zz)
				for (int i = 6; i < 15; i++) {
					results[i] = QAA[jaas[i-6]];
					results[i].multiply(-1.0);
				}
				 //BB (xx, xy, xz, yy, yz, zz) 
				for (int i = 24; i < 30; i++) results[i] = QAA[i-24];
			}
		} else if (dBC > 1e-6) {
			//BB (xx, xy, xz, yy, yz, zz)
			for (int i = 24; i < 30; i++) results[i] = QBB[i-24].transpose(); 
			// AB (xx, xy, xz, yx, yy, yz, zx, zy, zz)
			for (int i = 6; i < 15; i++) {
				results[i] = QBB[jaas[i-6]].transpose();
				results[i].multiply(-1.0);
			}
			 //AA (xx, xy, xz, yy, yz, zz) 
			for (int i = 0; i < 6; i++) results[i] = QBB[i].transpose();
		} 
	}

}
