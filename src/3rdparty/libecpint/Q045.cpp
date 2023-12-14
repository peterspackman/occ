// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q0_4_5(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 5, 5},
		Triple{1, 5, 6},
		Triple{2, 5, 5},
		Triple{2, 5, 7},
		Triple{3, 5, 6},
		Triple{3, 5, 8},
		Triple{4, 5, 5},
		Triple{4, 5, 7},
		Triple{4, 5, 9}	};

	ThreeIndex<double> radials(10, 6, 10);
	radint.type2(radial_triples_A, 8, 5, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
		Triple{1, 4, 5},
		Triple{2, 3, 5},
		Triple{3, 2, 5},
		Triple{3, 4, 5},
		Triple{4, 1, 5},
		Triple{4, 3, 5}	};

	ThreeIndex<double> radials_B(10, 10, 6);
	radint.type2(radial_triples_B, 8, 5, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(5, 0, 4, radials, CA, CB, SA, SB, angint, values);
}
}
}
