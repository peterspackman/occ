// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_0_0(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 0, 0}	};

	ThreeIndex<double> radials(1, 1, 1);
	radint.type2(radial_triples_A, 0, 0, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
	};

	ThreeIndex<double> radials_B(1, 1, 1);
	radint.type2(radial_triples_B, 0, 0, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	values(0, 0, 0) += 157.914 * CA(0, 0, 0, 0, 0) * CB(0, 0, 0, 0, 0) * radials(0, 0, 0) * SA(0, 0) * SB(0, 0);
}
}
}
