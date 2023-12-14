// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_3_4(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 4, 4},
		Triple{1, 4, 5},
		Triple{2, 4, 4},
		Triple{2, 4, 6},
		Triple{3, 4, 5},
		Triple{3, 4, 7}	};

	ThreeIndex<double> radials(8, 5, 8);
	radint.type2(radial_triples_A, 6, 4, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
		Triple{1, 3, 4},
		Triple{2, 2, 4},
		Triple{3, 1, 4},
		Triple{3, 3, 4}	};

	ThreeIndex<double> radials_B(8, 8, 5);
	radint.type2(radial_triples_B, 6, 4, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(4, 0, 3, radials, CA, CB, SA, SB, angint, values);
}
}
}
