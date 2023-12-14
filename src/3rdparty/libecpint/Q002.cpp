// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_0_2(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 2, 2}	};

	ThreeIndex<double> radials(3, 3, 3);
	radint.type2(radial_triples_A, 1, 2, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
	};

	ThreeIndex<double> radials_B(3, 3, 3);
	radint.type2(radial_triples_B, 1, 2, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	values(0, 0, 0) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 4);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 0);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 1);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 2);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 3);
	values(0, 0, 0) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 0);
	values(0, 0, 1) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 4);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 0);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 1);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 2);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 3);
	values(0, 0, 1) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 3);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 3);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 1);
	values(0, 0, 2) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 3);
	values(0, 0, 2) += -6.69202e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 4);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 0);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 1);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 2);
	values(0, 0, 2) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 3);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 4);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 0);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 1);
	values(0, 0, 2) += -6.69202e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 2);
	values(0, 0, 2) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 3);
	values(0, 0, 2) += 2.83593e-31 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 4);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 4);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 4);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 4);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 2);
	values(0, 0, 3) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 4);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 0);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 1);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 2);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 3);
	values(0, 0, 3) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 1);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 0) * SB(2, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 1);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 1) * SB(2, 4);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 0);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 1);
	values(0, 0, 4) += 2.83593e-31 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 2);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 3);
	values(0, 0, 4) += -6.69202e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 2) * SB(2, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 1);
	values(0, 0, 4) += -0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 3);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 3) * SB(2, 4);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 0);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 1);
	values(0, 0, 4) += -6.69202e-15 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 2);
	values(0, 0, 4) += 0 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 3);
	values(0, 0, 4) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 2, 2) * SA(2, 4) * SB(2, 4);
}
}
}
