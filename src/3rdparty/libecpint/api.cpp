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

#include "api.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include "mathutil.hpp"

namespace libecpint {
	
	void ECPIntegrator::set_gaussian_basis (
      const int nshells, const double* coords, const double* exponents, const double* coefs, const int* ams, const int* shell_lengths) {
		int ctr = 0;
		min_alpha = 100.0;
		for (int i = 0; i < nshells; i++) {
			ncart += (ams[i]+1)*(ams[i]+2)/2;
			std::array<double, 3> cvec = {coords[3*i], coords[3*i+1], coords[3*i+2]};
			GaussianShell newShell(cvec, ams[i]);
			if (ams[i] > maxLB) maxLB = ams[i];
			for (int n = 0; n < shell_lengths[i]; n++) {
				newShell.addPrim(exponents[ctr], coefs[ctr]);
				ctr++;
			}
			min_alpha = newShell.min_exp < min_alpha ? newShell.min_exp : min_alpha;
			shells.push_back(newShell);
		} 
		basis_is_set = true;
	}
	
	void ECPIntegrator::set_ecp_basis(
      const int necps, const double* coords, const double* exponents, const double* coefs, const int* ams, const int* ns, const int* shell_lengths) {
		int ctr = 0;
		for (int i = 0; i < necps; i++) {
			ECP newU(&coords[3*i]);
			for (int n = 0; n < shell_lengths[i]; n++) {
				newU.addPrimitive(ns[ctr], ams[ctr], exponents[ctr], coefs[ctr]); 
				ctr++;
			}
			newU.sort();
			ecps.addECP(newU, 0);
		}
		ecp_is_set = true;
	}
	
#ifdef HAS_PUGIXML
	void ECPIntegrator::set_ecp_basis_from_library(
      const int necps, const double* coords, const int* charges, const std::vector<std::string> & names, const std::string & share_dir) {
		for (int i = 0; i < necps; i++) {
			std::array<double, 3> center = {coords[3*i], coords[3*i+1], coords[3*i+2]};
			int q = charges[i];
			std::string filename = share_dir + "/xml/" + names[i] + ".xml"; 
			ecps.addECP_from_file(q, center, filename);
		}
		ecp_is_set = true;
	}
#endif
	
	void ECPIntegrator::update_gaussian_basis_coords(const int nshells, const double* coords) {
		assert(nshells == shells.size());
		
		for (int i = 0; i < nshells; i++){
			shells[i].localCenter[0] = coords[3*i];
			shells[i].localCenter[1] = coords[3*i+1];
			shells[i].localCenter[2] = coords[3*i+2];
		}
	}
	
	void ECPIntegrator::update_ecp_basis_coords(const int necps, const double* coords) {
		assert(necps == ecps.getN());
		
		for (int i = 0; i < necps; i++) 
			ecps.getECP(i).setPos(coords[3*i], coords[3*i+1], coords[3*i+2]);
	}
	
	void ECPIntegrator::init(const int deriv_) {
		assert(ecp_is_set);
		assert(basis_is_set);
		deriv = std::max(0, std::min(2, deriv_));
		ecpint = std::make_shared<ECPIntegral>(maxLB, ecps.getMaxL(), deriv);
		
		// Determine the internal atom ids
		natoms = 0;
		std::vector<std::array<double, 3>> centers;
		for (auto& s : shells) {
			int i = 0;
			bool found = false;
			while ( !found && (i < centers.size()) ) {
				double diff = std::abs(centers[i][0] - s.centerVec[0]);
				diff += std::abs(centers[i][1] - s.centerVec[1]);
				diff += std::abs(centers[i][2] - s.centerVec[2]);
				if (diff < 1e-4) {
					s.atom_id = i;
					found = true;
				}
				i++;
			}
			if (!found) {
				s.atom_id = natoms;
				natoms++;
				centers.push_back({s.centerVec[0], s.centerVec[1], s.centerVec[2]});
			}
		}
		
		for (int n = 0; n < ecps.getN(); n++) {
			ECP& U = ecps.getECP(n);
			int i = 0;
			bool found = false;
			while ( !found && (i < centers.size()) ) {
				double diff = std::abs(centers[i][0] - U.center_[0]);
				diff += std::abs(centers[i][1] - U.center_[1]);
				diff += std::abs(centers[i][2] - U.center_[2]);
				if (diff < 1e-4) {
					U.atom_id = i;
					found = true;
				}
				i++;
			}
			if (!found) {
				U.atom_id = natoms;
				natoms++;
				centers.push_back({U.center_[0], U.center_[1], U.center_[2]});
			}
		}
	}
	
	double shell_bound(const int la, const double alpha, const double A2, const double eta) {
		double sigma;
		if (A2 < 1e-6) {
			sigma = 0.5 * (1.0 + eta/alpha);
		} else {
			sigma = 1.0/(2.0*alpha*(eta*eta*A2 + la*(alpha + eta)));
			sigma = sigma * la * (alpha + eta) * (alpha + eta);
		}
		
		double atilde = (1.0 - sigma) * alpha;
		double Na = la / (2*M_EULER*alpha*sigma);
		Na = FAST_POW[la](std::sqrt(Na));
		double result = atilde * eta * A2 / (atilde + eta);
		result = std::exp(-result) * Na;
		return result;
	}
	
	void ECPIntegrator::compute_integrals() {
		// initialise all to zero
		integrals.assign(ncart, ncart, 0.0);
		
		// loop over shells
		TwoIndex<double> tempValues;
		int nshells = shells.size();
	
		double thresh = FAST_POW[maxLB+3]((maxLB+3.0)/min_alpha)*FAST_POW[3](M_PI/(2*maxLB+3.0));
		thresh /= FAST_POW[maxLB](2.0*M_EULER);
		thresh = TWO_C_TOLERANCE / std::sqrt(thresh);
		
		int n1 = 0;
		double acx, acy, acz, A2, sb;
		for(auto s1=0; s1<nshells; ++s1) {
			GaussianShell& shellA = shells[s1];
			int ncartA = shellA.ncartesian();
			std::vector<int> ns;
			
			for (int i = 0; i < ecps.getN(); i++) {
				ECP& U = ecps.getECP(i);	
				
				acx = shellA.center()[0] - U.center_[0]; 
				acy = shellA.center()[1] - U.center_[1];
				acz = shellA.center()[2] - U.center_[2];
				A2 = acx*acx + acy*acy + acz*acz;
				sb = shell_bound(shellA.l, shellA.min_exp, A2, U.min_exp);
				if (sb > thresh) ns.push_back(i);
			}
			
			if (ns.size() > 0) {
				int n2 = 0;
				for(auto s2=0; s2<=s1; ++s2) {
					GaussianShell& shellB = shells[s2];
					int ncartB = shellB.ncartesian();
			
					TwoIndex<double> shellPairInts(ncartA, ncartB, 0.0);
				
					for (auto i : ns) {
						ECP& U = ecps.getECP(i);
						ecpint->compute_shell_pair(U, shellA, shellB, tempValues);
						shellPairInts.add(tempValues);
					}

					for (int i = n1; i < n1 + ncartA; i++) {
						for (int j = n2; j < n2 + ncartB; j++) {
							integrals(i, j) = shellPairInts(i-n1, j-n2);
							integrals(j, i) = integrals(i, j);
						}
					}
			   
					n2 += ncartB;
				}
			}
			n1 += ncartA;
		} 
		
		//std::cout << "Total: " << ecpint->skipped + ecpint->zero + ecpint->nonzero << std::endl;
		//std::cout << "Skipped: " << ecpint->skipped << std::endl;
		//std::cout << "Zero: " << ecpint->zero << std::endl;
		//std::cout << "Non-zero: " << ecpint->nonzero << std::endl;
		
	}
	
	void ECPIntegrator::compute_first_derivs() {
		assert(deriv > 0);
		
		for (int n = 0; n < 3*natoms; n++)
			first_derivs.push_back(TwoIndex<double>(ncart, ncart, 0.0));
		
		// loop over shells
		std::array<TwoIndex<double>, 9> tempValues;
		int nshells = shells.size();
		
		int n1 = 0;
		int Aix, Bix, Cix;
		for(auto s1=0; s1<nshells; ++s1) {
			GaussianShell& shellA = shells[s1];
			int ncartA = shellA.ncartesian();
			Aix = shellA.atom_id;
			
			int n2 = 0;
			for(auto s2=0; s2<=s1; ++s2) {
				GaussianShell& shellB = shells[s2];
				int ncartB = shellB.ncartesian();
				Bix = shellB.atom_id;
				
				for (int i = 0; i < ecps.getN(); i++) {
					ECP& U = ecps.getECP(i);
					Cix = U.atom_id;
					ecpint->compute_shell_pair_derivative(U, shellA, shellB, tempValues);
					
					// work out where to put them
					for (int n = 0; n < 3; n++) {
						for (int k = n1; k < n1 + ncartA; k++) {
							for (int l = n2; l < n2 + ncartB; l++) {
								first_derivs[3*Aix+n](k, l) += tempValues[n](k-n1, l-n2);
								first_derivs[3*Bix+n](k, l) += tempValues[n+3](k-n1, l-n2);
								first_derivs[3*Cix+n](k, l) += tempValues[n+6](k-n1, l-n2);
								
								if (s2 < s1) {
									first_derivs[3*Aix+n](l, k) = first_derivs[3*Aix+n](k, l);
									first_derivs[3*Bix+n](l, k) = first_derivs[3*Bix+n](k, l);
									first_derivs[3*Cix+n](l, k) = first_derivs[3*Cix+n](k, l);
								}

							}
						}
					} 		
				}
			
				n2 += ncartB;
			}
		
			n1 += ncartA; 
		}
	}
	
	void ECPIntegrator::compute_second_derivs() {
		assert(deriv > 1);
		
		int nhess = (3*natoms*(3*natoms+1))/2;
		for (int n = 0; n < nhess; n++)
			second_derivs.push_back(TwoIndex<double>(ncart, ncart, 0.0));
		
		// loop over shells
		std::array<TwoIndex<double>, 45> tempValues;
		int nshells = shells.size();
		
		int n1 = 0;
		int Aix, Bix, Cix;
		int saa, sab, sac, sbb, sbc, scc;
		int ixes[6] = {0, 1, 2, 4, 5, 8};
		int back_ixes[6] = {0, 3, 6, 4, 7, 8};
		int jxes[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
		for(auto s1=0; s1<nshells; ++s1) {
			GaussianShell& shellA = shells[s1];
			int ncartA = shellA.ncartesian();
			Aix = shellA.atom_id;
			
			int n2 = 0;
			for(auto s2=0; s2<=s1; ++s2) {
				GaussianShell& shellB = shells[s2];
				int ncartB = shellB.ncartesian();
				Bix = shellB.atom_id;
				
				saa = H_START(Aix, Aix, natoms) + 3;
				sbb = H_START(Bix, Bix, natoms) + 3;
				sab = H_START(std::min(Aix, Bix), std::max(Aix, Bix), natoms);
				sab = Aix == Bix ? sab + 3 : sab;
				
				for (int i = 0; i < ecps.getN(); i++) {
					ECP& U = ecps.getECP(i);
					Cix = U.atom_id;
					ecpint->compute_shell_pair_second_derivative(U, shellA, shellB, tempValues);
				
					// work out where to put them
					scc = H_START(Cix, Cix, natoms) + 3;
					sac = H_START(std::min(Aix, Cix), std::max(Aix, Cix), natoms);
					sac = Aix == Cix ? sac + 3 : sac;
					sbc = H_START(std::min(Bix, Cix), std::max(Bix, Cix), natoms);
					sbc = Bix == Cix ? sbc + 3 : sbc;
					
					if ((Aix == Cix) || (Bix == Cix)) {
						if (Bix != Aix) {
							// two distinct atoms
							// only need to worry about AA, AB, and BB blocks
							for (int n = 0; n < 6; n++) {
								for (int k = n1; k < n1 + ncartA; k++) {
									for (int l = n2; l < n2 + ncartB; l++) {
										second_derivs[saa+n](k, l) += tempValues[n](k-n1, l-n2);
										second_derivs[sbb+n](k, l) += tempValues[n+24](k-n1, l-n2);
								
										if (s1 != s2) {
											second_derivs[saa+n](l, k) = second_derivs[saa+n](k, l);
											second_derivs[sbb+n](l, k) = second_derivs[sbb+n](k, l);
										}
									}
								}
							}
							
							for (int n = 0; n < 9; n++) {
								for (int k = n1; k < n1 + ncartA; k++) {
									for (int l = n2; l < n2 + ncartB; l++) {
										if (Aix > Bix) {
											second_derivs[sab+n](k, l) += tempValues[jxes[n]+6](k-n1, l-n2);
											if (s1 != s2) second_derivs[sab+n](l, k) = second_derivs[sab+n](k, l);
										} else {
											second_derivs[sab+n](k, l) += tempValues[n+6](k-n1, l-n2);
											if (s1 != s2) second_derivs[sab+n](l, k) = second_derivs[sab+n](k, l);
										}
									}
								}
							}
						} // else everything is zero
					} else if (Aix == Bix) {
						// two distinct atoms, need to worry about everything
						for (int n = 0; n < 6; n++) {
							for (int k = n1; k < n1 + ncartA; k++) {
								for (int l = n2; l < n2 + ncartB; l++) {
									second_derivs[saa+n](k, l) += tempValues[n](k-n1, l-n2); // aa
									second_derivs[saa+n](k, l) += tempValues[n+24](k-n1, l-n2); // bb = aa
									second_derivs[scc+n](k, l) += tempValues[n+39](k-n1, l-n2); // cc
									second_derivs[saa+n](k, l) += tempValues[ixes[n]+6](k-n1, l-n2); // ab = aa
									second_derivs[saa+n](k, l) += tempValues[back_ixes[n]+6](k-n1, l-n2); // ba = aa
									
									if (s1 != s2) {
										second_derivs[saa+n](l, k) = second_derivs[saa+n](k, l);
										second_derivs[scc+n](l, k) = second_derivs[scc+n](k, l);
									}
								}
							}
						}
					
						for (int n = 0; n < 9; n++) {
							for (int k = n1; k < n1 + ncartA; k++) {
								for (int l = n2; l < n2 + ncartB; l++) {						
									if (Aix > Cix) {
										second_derivs[sac+n](k, l) += tempValues[jxes[n]+15](k-n1, l-n2);
										second_derivs[sac+n](k, l) += tempValues[jxes[n]+30](k-n1, l-n2); // bc = ac
										
										if (s1 != s2) second_derivs[sac+n](l, k) = second_derivs[sac+n](k, l);
									} else {
										second_derivs[sac+n](k, l) += tempValues[n+15](k-n1, l-n2);
										second_derivs[sac+n](k, l) += tempValues[n+30](k-n1, l-n2); // bc = ac
										
										if (s1 != s2) second_derivs[sbc+n](l, k) = second_derivs[sbc+n](k, l);
									}
								}
							}
						}
					} else {
						for (int n = 0; n < 6; n++) {
							for (int k = n1; k < n1 + ncartA; k++) {
								for (int l = n2; l < n2 + ncartB; l++) {
									second_derivs[saa+n](k, l) += tempValues[n](k-n1, l-n2);
									second_derivs[sbb+n](k, l) += tempValues[n+24](k-n1, l-n2);
									second_derivs[scc+n](k, l) += tempValues[n+39](k-n1, l-n2);
								
									if (s1 != s2) {
										second_derivs[saa+n](l, k) = second_derivs[saa+n](k, l);
										second_derivs[sbb+n](l, k) = second_derivs[sbb+n](k, l);
										second_derivs[scc+n](l, k) = second_derivs[scc+n](k, l);
									}
								}
							}
						}
					
						for (int n = 0; n < 9; n++) {
							for (int k = n1; k < n1 + ncartA; k++) {
								for (int l = n2; l < n2 + ncartB; l++) {
									if (Aix > Bix) {
										second_derivs[sab+n](k, l) += tempValues[jxes[n]+6](k-n1, l-n2);
										if (s1 != s2) second_derivs[sab+n](l, k) = second_derivs[sab+n](k, l);
									} else {
										second_derivs[sab+n](k, l) += tempValues[n+6](k-n1, l-n2);
										if (s1 != s2) second_derivs[sab+n](l, k) = second_derivs[sab+n](k, l);
									}
								
									if (Aix > Cix) {
										second_derivs[sac+n](k, l) += tempValues[jxes[n]+15](k-n1, l-n2);
										if (s1 != s2) second_derivs[sac+n](l, k) = second_derivs[sac+n](k, l);
									} else {
										second_derivs[sac+n](k, l) += tempValues[n+15](k-n1, l-n2);
										if (s1 != s2) second_derivs[sac+n](l, k) = second_derivs[sac+n](k, l);
									}
								
									if (Bix > Cix) {
										second_derivs[sbc+n](k, l) += tempValues[jxes[n]+30](k-n1, l-n2);
										if (s1 != s2) second_derivs[sbc+n](l, k) = second_derivs[sbc+n](k, l);
									} else {
										second_derivs[sbc+n](k, l) += tempValues[n+30](k-n1, l-n2);
										if (s1 != s2) second_derivs[sbc+n](l, k) = second_derivs[sbc+n](k, l);
									}
								}
							}
						}
					}
					
				}
			
				n2 += ncartB;
			}
		
			n1 += ncartA; 
		}
	}
}
