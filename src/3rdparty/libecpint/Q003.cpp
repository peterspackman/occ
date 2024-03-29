// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_0_3(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 3, 3}	};

	ThreeIndex<double> radials(4, 4, 4);
	radint.type2(radial_triples_A, 2, 3, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
	};

	ThreeIndex<double> radials_B(4, 4, 4);
	radint.type2(radial_triples_B, 2, 3, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	values(0, 0, 0) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 0) += 1.14283e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 0) += 1.14283e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 0) += 8.27075e-33 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 1) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 2) += 8.27075e-33 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 2) += 1.14283e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 2) += 1.14283e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 2) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 3) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 3) += -1.0678e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 3) += -1.0678e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 3) += 7.22043e-33 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 3) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 4) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 4) += 7.07337e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 4) += 7.07337e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 4) += 3.16835e-31 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 5) += 7.22043e-33 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 5) += -1.0678e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 5) += -1.0678e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 5) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 5) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 5) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 3);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 5);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 0) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 3);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 5);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 1) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 3);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 5);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 2) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 3);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 5);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 3) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 3);
	values(0, 0, 6) += 3.16835e-31 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 5);
	values(0, 0, 6) += 7.07337e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 4) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 3);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 5);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 5) * SB(3, 6);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 0);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 1);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 2);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 3);
	values(0, 0, 6) += 7.07337e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 4);
	values(0, 0, 6) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 5);
	values(0, 0, 6) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 3, 3) * SA(3, 6) * SB(3, 6);
}
}
}
