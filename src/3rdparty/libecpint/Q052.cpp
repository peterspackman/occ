// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_5_2(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 2, 2},
		Triple{1, 2, 3},
		Triple{2, 2, 2},
		Triple{2, 2, 4},
		Triple{3, 2, 3},
		Triple{3, 2, 5},
		Triple{4, 2, 2},
		Triple{4, 2, 4},
		Triple{4, 2, 6},
		Triple{5, 2, 3},
		Triple{5, 2, 5},
		Triple{5, 2, 7}	};

	ThreeIndex<double> radials(8, 3, 8);
	radint.type2(radial_triples_A, 6, 2, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
		Triple{1, 1, 2},
		Triple{2, 0, 2},
		Triple{3, 1, 2},
		Triple{4, 0, 2},
		Triple{5, 1, 2}	};

	ThreeIndex<double> radials_B(8, 8, 3);
	radint.type2(radial_triples_B, 6, 2, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(2, 0, 5, radials, CA, CB, SA, SB, angint, values);
}
}
}
