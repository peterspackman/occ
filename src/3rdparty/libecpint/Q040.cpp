// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_4_0(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 0, 0},
		Triple{1, 0, 1},
		Triple{2, 0, 0},
		Triple{2, 0, 2},
		Triple{3, 0, 1},
		Triple{3, 0, 3},
		Triple{4, 0, 0},
		Triple{4, 0, 2},
		Triple{4, 0, 4}	};

	ThreeIndex<double> radials(5, 1, 5);
	radint.type2(radial_triples_A, 3, 0, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
	};

	ThreeIndex<double> radials_B(5, 5, 1);
	radint.type2(radial_triples_B, 3, 0, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(0, 0, 4, radials, CA, CB, SA, SB, angint, values);
}
}
}
