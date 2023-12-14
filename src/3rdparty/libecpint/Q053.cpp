// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_5_3(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 3, 3},
		Triple{1, 3, 4},
		Triple{2, 3, 3},
		Triple{2, 3, 5},
		Triple{3, 3, 4},
		Triple{3, 3, 6},
		Triple{4, 3, 3},
		Triple{4, 3, 5},
		Triple{4, 3, 7},
		Triple{5, 3, 4},
		Triple{5, 3, 6},
		Triple{5, 3, 8}	};

	ThreeIndex<double> radials(9, 4, 9);
	radint.type2(radial_triples_A, 7, 3, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
		Triple{1, 2, 3},
		Triple{2, 1, 3},
		Triple{3, 0, 3},
		Triple{3, 2, 3},
		Triple{4, 1, 3},
		Triple{5, 0, 3},
		Triple{5, 2, 3}	};

	ThreeIndex<double> radials_B(9, 9, 4);
	radint.type2(radial_triples_B, 7, 3, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(3, 0, 5, radials, CA, CB, SA, SB, angint, values);
}
}
}
