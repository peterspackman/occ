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

#include "angular.hpp"
#include "bessel.hpp"
#include "mathutil.hpp"
#include <cmath>

namespace libecpint {

	double AngularIntegral::calcG(const int l, const int m) const {
		double value = 0.0;
		double value1 = FAST_POW[l](2.0) * FAC[l];
		value1 = 1.0 / value1; 
		double value2 = (2.0 * l + 1) * FAC[l - m] / (2.0 * M_PI * FAC[l + m]);
		value2 = std::sqrt(value2); 
		value = value1 * value2;
		return value;
	} 

	double AngularIntegral::calcH1(
      const int i, const int j, const int l, const int m) const {
		double value = 0.0; 

		value = FAC[l]/(FAC[j]*FAC[l - i]*FAC[i-j]);
		value *= (1 - 2*(i%2)) * FAC[2*(l - i)] / (FAC[l - m - 2*i]);

		return value;
	}

	double AngularIntegral::calcH2(
      const int i, const int j, const int k, const int m) const {
		double value = 0.0; 
		int ki2 = k - 2*i;
		if ( m >= ki2 && ki2 >= 0 ) {
			value = FAC[j]*FAC[m]/(FAC[i] * FAC[j-i] * FAC[ki2] * FAC[m-ki2]);
			int p = (m - k + 2*i)/2;
			value *= (1.0 - 2.0*(p%2));
		}
		return value;
	}


	ThreeIndex<double> AngularIntegral::uklm(
      const int lam, const int mu) const {
		ThreeIndex<double> values(lam+1, lam+1, 2);
	 
		double or2 = 1.0/std::sqrt(2.0);
		double u = 0.0;
		double um = 0.0;
		double g = calcG(lam, mu);

		double u1, h1, h2;
		int j;
		for (int k = 0; k <= lam; k++) {
			for (int l = 0; l <= lam - k; l++) {
				u = um = 0.0;
				j = k + l - mu;
				if (j % 2 == 0 && j > -1) { 
					u1 = 0.0;
					j/=2;
					for (int i = j; i <= (lam - mu)/2; i++) u1 += calcH1(i, j, lam, mu);
			
					u = g * u1;
					u1 = 0;
					for (int i = 0; i <= j; i++) u1 += calcH2(i, j, k, mu);
					u *= u1;
					um = u;
			
					j = l % 2;
					u *= (1 - j);
					um *= j;
					if (mu == 0) {
						u *= or2;
						um = u;
					} 
				}
				values(k, l, 0) = u;
				values(k, l, 1) = um;
			}
		}
		return values;						
	}


	ThreeIndex<double> AngularIntegral::Pijk(const int maxI) const {
		int dim = maxI+1;
		ThreeIndex<double> values(dim, dim, dim);
		double pi4 = 4.0*M_PI;
	
		values(0, 0, 0) = pi4;
		for (int i = 1; i <= maxI; i++) {
			values(i, 0, 0) = pi4 / ((double) (2*i+1));
		
			for (int j = 1; j <= i; j++) {
				values(i, j, 0) = values(i, j-1, 0) * (2.0*j - 1.0) / (2.0 * ((double)(i + j)) + 1.0);
			
				for (int k = 1; k <= j; k++)
					values(i, j, k) = values(i, j, k-1) * (2.0*k - 1.0) / (2.0 * ((double)(i + j + k)) + 1.0);
			
			}
		}
		return values;
	}

	FiveIndex<double> AngularIntegral::makeU() const {
		int dim = maxL + 1;

		FiveIndex<double> values(dim, dim, dim, dim, 2);
		for (int lam = 0; lam <= maxL; lam++) {
			for (int mu = 0; mu <= lam; mu++) {
				ThreeIndex<double> Uij = uklm(lam, mu);
				for (int i = 0; i <= lam; i++) {
					for (int j = 0; j <= lam - i; j++){
						values(lam, mu, i, j, 0) = Uij(i, j, 0);
						values(lam, mu, i, j, 1) = Uij(i, j, 1);
					}
				}
			}
		}
	
		return values;
	}

	void AngularIntegral::makeW(const FiveIndex<double> &U) {
		int LB2 = 2*LB;
		int dim = wDim;
		int maxI = (maxL + dim)/2;
		int maxLam = maxL;
	
		FiveIndex<double> values{dim+1, dim+1, dim+1, maxLam+1, 2*(maxLam + 1)};
		ThreeIndex<double> pijk = Pijk(maxI);
	
		int plam, pmu;
		double smu, w;
		std::vector<int> ix(3);
		for (int k = 0; k <= dim; k++) {	
			for (int l = 0; l <= dim; l++) {	
				for(int m = 0; m <= dim; m++) {
					plam = (k + l + m)%2;
				
					int limit = maxLam > k+l+m ? k+l+m : maxLam;
					for(int lam = plam; lam <= limit; lam += 2){
						smu = 1 - 2*(l%2);
						pmu = (k+l) % 2;
					
						for (int mu = pmu; mu <= lam; mu+=2) {
							w = 0.0;
							for (int i = 0; i <= lam; i++) {
								for (int j = 0; j <= lam - i; j++) {
									ix[0] = k+i;
									ix[1] = l+j;
									ix[2] = m + lam - i - j; 
								
									if (ix[0]%2 + ix[1]%2 + ix[2]%2 == 0){
										std::sort(ix.begin(), ix.end()); 
										w += U(lam, mu, i, j, (1 - (int)(smu))/2)*pijk(ix[2]/2, ix[1]/2, ix[0]/2);
									}
								
								}
							}
						
							values(k, l, m, lam, lam+(int)(smu*mu)) = w;
						}
					}	
				}	
			}	
		}
		W = values;
	}

	void AngularIntegral::makeOmega(const FiveIndex<double> &U) {
	
		int lamDim = LE + LB; 
		int muDim = 2*lamDim + 1;
		SevenIndex<double> values{LB+1, LB+1, LB+1, lamDim+1, muDim+1, lamDim+1, muDim+1};
		
		double om_plus=0.0, om_minus=0.0;
		double wval; 
		for (int k = 0; k <= LB; k++) {
			for (int l = 0; l <= LB; l++) {
				for (int m = 0; m <= LB; m++) {
					
					for (int rho = 0; rho <= lamDim; rho++ ) {
						for (int sigma = -rho; sigma <= rho; sigma++) {
						
							for (int lam = 0; lam <= rho; lam++) {
	
								for (int mu = 0; mu <= lam; mu++) {
								
									om_plus = om_minus = 0.0;
									for (int i = 0; i<= lam; i++ ) {
										for (int j = 0; j <= lam - i; j++) {												
											wval = W(k+i, l+j, m+lam-i-j, rho, rho+sigma);
											om_plus += U(lam, mu, i, j, 0) * wval;
											om_minus += U(lam, mu, i, j, 1) * wval;
										}
									}
									if (mu == 0) om_minus = om_plus;
									values(k, l, m, rho, sigma+rho, lam, lam+mu) = om_plus;
									values(k, l, m, lam, lam+mu, rho, sigma+rho) = om_plus;
									values(k, l, m, rho, sigma+rho, lam, lam-mu) = om_minus;
									values(k, l, m, lam, lam-mu, rho, sigma+rho) = om_minus;
								
								}
							}
						
						}
					}
					
				}
			}
		}
	
		omega = values;
	}

	AngularIntegral::AngularIntegral() { init(0, 0); }
	AngularIntegral::AngularIntegral(const int _LB, const int _LE) { init(_LB, _LE); }
	void AngularIntegral::init(const int _LB, const int _LE ) {
		LB = _LB;
		LE = _LE;
		wDim = 4*LB > 3*LB + LE ? 4*LB : 3*LB + LE;
		maxL = 2*LB > LB + LE ? 2*LB : LB+LE;
	
	}

	void AngularIntegral::compute() {
		FiveIndex<double> U = makeU();
		makeW(U);
		makeOmega(U);
	}

	void AngularIntegral::clear() {}

	double AngularIntegral::getIntegral(
      const int k, const int l, const int m, const int lam, const int mu) const {
    return W(k, l, m, lam, lam+mu);
	}
	double AngularIntegral::getIntegral(
      const int k, const int l, const int m, const int lam, const int mu, const int rho, const int sigma) const {
	  return omega(k, l, m, lam, lam+mu, rho, rho+sigma);
	}

	bool AngularIntegral::isZero(
      const int k, const int l, const int m, const int lam, const int mu, const double tolerance) const {
		if (wDim > 0) return std::fabs(W(k, l, m, lam, lam+mu)) < tolerance;
		else return true;
	}
	bool AngularIntegral::isZero(
      const int k, const int l, const int m, const int lam, const int mu, const int rho, const int sigma, const double tolerance) const {
		if (wDim > 0) return std::fabs(omega(k, l, m, lam, lam+mu, rho, rho+sigma)) < tolerance;
		else return true;
	}

}
