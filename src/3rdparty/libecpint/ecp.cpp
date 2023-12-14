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

#include "ecp.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>
#include "mathutil.hpp"
#ifdef HAS_PUGIXML
#include "pugixml.hpp"
#endif

namespace libecpint {

	// GaussianECP constructor and copy constructor
	GaussianECP::GaussianECP() : n(0), l(0), a(0), d(0) {}
	GaussianECP::GaussianECP(
	    const int _n, const int _l, const double _a, const double _d) : n(_n-2), l(_l), a(_a), d(_d) {}
	GaussianECP::GaussianECP(const GaussianECP& other) : n(other.n), l(other.l), a(other.a), d(other.d) {}


	// class ECP

	ECP::ECP() : N(0), L(-1) {
		center_[0] = center_[1] = center_[2] = 0.0; 	
		min_exp = 1000.0;
		for (int i = 0; i < LIBECPINT_MAX_L + 1; i++) {
			 min_exp_l[i] = 1000.0;
			 l_starts[i] = 0;
		}
		l_starts[LIBECPINT_MAX_L+1] = 0;
	}
	
	ECP::ECP(const double *_center) : N(0), L(-1) {
		center_[0] = _center[0];
		center_[1] = _center[1];
		center_[2] = _center[2];
		min_exp = 1000.0;
		for (int i = 0; i < LIBECPINT_MAX_L + 1; i++) {
			 min_exp_l[i] = 1000.0;
			 l_starts[i] = 0;
		}
		l_starts[LIBECPINT_MAX_L+1] = 0;
	}

	ECP::ECP(const ECP &other) {
		gaussians = other.gaussians;
		N = other.N;
		L = other.L;
		atom_id=other.atom_id;
		min_exp = other.min_exp;
		for (int i = 0; i < LIBECPINT_MAX_L + 1; i++) {
			min_exp_l[i] = other.min_exp_l[i];
			l_starts[i] = other.l_starts[i];
		}
		l_starts[LIBECPINT_MAX_L+1] = other.l_starts[LIBECPINT_MAX_L+1];
		center_ = other.center_;
	}

	void ECP::addPrimitive(
      const int n, const int l, const double a, const double d, const bool needSort) {
		GaussianECP newEcp(n, l, a, d);
		gaussians.push_back(newEcp);
		N++;
		L = l > L ? l : L;
		min_exp = a < min_exp ? a : min_exp;
		min_exp_l[l] = a < min_exp_l[l] ? a : min_exp_l[l];
		for (int lx = l+1; lx < LIBECPINT_MAX_L + 2; lx++)
			l_starts[lx] += 1;
		if (needSort) sort();
	}

	void ECP::sort() {
		std::sort(gaussians.begin(), gaussians.end(),
		[&] (const GaussianECP& g1, const GaussianECP& g2) {return (g1.l < g2.l);});
	}
	
	bool ECP::noType1() const {
		bool zero = true;
		for (auto& g : gaussians)
			if (g.l == L && fabs(g.d) > 1e-12) zero = false; 
		return zero; 
	}

	// Evaluate U_l(r), assuming that gaussians sorted by angular momentum
	double ECP::evaluate(const double r, const int l) const {
		double value = 0.0;
		double r2 = r*r;
		int p;
		for (int i = l_starts[l]; i < l_starts[l+1]; i++) {
			p = gaussians[i].n > -1 ? gaussians[i].n : MAX_POW - gaussians[i].n;
			value += FAST_POW[p](r) * gaussians[i].d * exp(-gaussians[i].a * r2);
		} 
		return value; 
	}

	void ECP::setPos(const double x, const double y, const double z) {
		center_[0] = x; center_[1] = y; center_[2] = z;
	}

	ECPBasis::ECPBasis() : N(0), maxL(-1) {}

	void ECPBasis::addECP(const ECP &U, const int atom) {
		basis.push_back(U);
		atomList.push_back(atom);
		N++;
		maxL = U.getL() > maxL ? U.getL() : maxL;
	}

  ECP& ECPBasis::getECP(const int i) { return basis[i]; }
  const ECP& ECPBasis::getECP(const int i) const { return basis[i]; }

	int ECPBasis::getECPCore(const int q) const {
		int core = 0;
		auto it = core_electrons.find(q);
		if (it != core_electrons.end()) core = it->second;
		return core;
	}

#ifdef HAS_PUGIXML
	void ECPBasis::addECP_from_file(
      const int q, const std::array<double, 3> & coords, const std::string & filename) {
		ECP newECP;
		newECP.center_ = coords;

		std::string atom_name = q < 1 ? "X" : atom_names[q-1]; 
		pugi::xml_document doc;
		pugi::xml_parse_result result = doc.load_file(filename.c_str());
		pugi::xml_node atom_node = doc.child("root").child(atom_name.c_str()); 
		int maxl = std::stoi(atom_node.attribute("maxl").value());
		int ncore = std::stoi(atom_node.attribute("ncore").value()); 
		
		auto it = core_electrons.find(q);
		if (it == core_electrons.end())
			core_electrons[q] = ncore; 
	
		for (pugi::xml_node shell = atom_node.child("Shell"); shell; shell = shell.next_sibling("Shell")) {

			int l = std::stoi(shell.attribute("lval").value());
			
			for (pugi::xml_node nxc = shell.child("nxc"); nxc; nxc = nxc.next_sibling("nxc")) {
				int n = std::stoi(nxc.attribute("n").value()); 
				double x = std::stod(nxc.attribute("x").value()); 
				double c = std::stod(nxc.attribute("c").value()); 
				newECP.addPrimitive(n, l, x, c); 
			}
		}
		
		newECP.sort();
		addECP(newECP, 0);
	}
#endif
}
