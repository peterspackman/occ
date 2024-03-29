// Generated as part of Libecpint, Copyright 2017 Robert A Shaw
#include "qgen.hpp"
namespace libecpint {
namespace qgen {
void Q5_5_5(const ECP& U, const GaussianShell& shellA, const GaussianShell& shellB, const FiveIndex<double> &CA, const FiveIndex<double> &CB, const TwoIndex<double> &SA, const TwoIndex<double> &SB, const double Am, const double Bm, const RadialIntegral &radint, const AngularIntegral& angint, const RadialIntegral::Parameters& parameters, ThreeIndex<double> &values) {

	std::vector<Triple> radial_triples_A = {
		Triple{0, 5, 5},
		Triple{1, 4, 5},
		Triple{1, 5, 6},
		Triple{2, 3, 5},
		Triple{2, 4, 4},
		Triple{2, 4, 6},
		Triple{2, 5, 5},
		Triple{2, 5, 7},
		Triple{2, 6, 6},
		Triple{3, 2, 5},
		Triple{3, 3, 4},
		Triple{3, 3, 6},
		Triple{3, 4, 5},
		Triple{3, 4, 7},
		Triple{3, 5, 6},
		Triple{3, 5, 8},
		Triple{3, 6, 7},
		Triple{4, 1, 5},
		Triple{4, 2, 4},
		Triple{4, 2, 6},
		Triple{4, 3, 3},
		Triple{4, 3, 5},
		Triple{4, 3, 7},
		Triple{4, 4, 4},
		Triple{4, 4, 6},
		Triple{4, 4, 8},
		Triple{4, 5, 5},
		Triple{4, 5, 7},
		Triple{4, 5, 9},
		Triple{4, 6, 6},
		Triple{4, 6, 8},
		Triple{4, 7, 7},
		Triple{5, 0, 5},
		Triple{5, 1, 4},
		Triple{5, 1, 6},
		Triple{5, 2, 3},
		Triple{5, 2, 5},
		Triple{5, 2, 7},
		Triple{5, 3, 4},
		Triple{5, 3, 6},
		Triple{5, 3, 8},
		Triple{5, 4, 5},
		Triple{5, 4, 7},
		Triple{5, 4, 9},
		Triple{5, 5, 6},
		Triple{5, 5, 8},
		Triple{5, 5, 10},
		Triple{5, 6, 7},
		Triple{5, 6, 9},
		Triple{5, 7, 8},
		Triple{6, 0, 4},
		Triple{6, 0, 6},
		Triple{6, 1, 3},
		Triple{6, 1, 5},
		Triple{6, 1, 7},
		Triple{6, 2, 2},
		Triple{6, 2, 4},
		Triple{6, 2, 6},
		Triple{6, 2, 8},
		Triple{6, 3, 3},
		Triple{6, 3, 5},
		Triple{6, 3, 7},
		Triple{6, 3, 9},
		Triple{6, 4, 4},
		Triple{6, 4, 6},
		Triple{6, 4, 8},
		Triple{6, 4, 10},
		Triple{6, 5, 5},
		Triple{6, 5, 7},
		Triple{6, 5, 9},
		Triple{6, 6, 6},
		Triple{6, 6, 8},
		Triple{6, 6, 10},
		Triple{6, 7, 7},
		Triple{6, 7, 9},
		Triple{6, 8, 8},
		Triple{7, 0, 3},
		Triple{7, 0, 5},
		Triple{7, 0, 7},
		Triple{7, 1, 2},
		Triple{7, 1, 4},
		Triple{7, 1, 6},
		Triple{7, 1, 8},
		Triple{7, 2, 3},
		Triple{7, 2, 5},
		Triple{7, 2, 7},
		Triple{7, 2, 9},
		Triple{7, 3, 4},
		Triple{7, 3, 6},
		Triple{7, 3, 8},
		Triple{7, 3, 10},
		Triple{7, 4, 5},
		Triple{7, 4, 7},
		Triple{7, 4, 9},
		Triple{7, 5, 6},
		Triple{7, 5, 8},
		Triple{7, 5, 10},
		Triple{7, 6, 7},
		Triple{7, 6, 9},
		Triple{7, 7, 8},
		Triple{7, 7, 10},
		Triple{7, 8, 9},
		Triple{8, 0, 2},
		Triple{8, 0, 4},
		Triple{8, 0, 6},
		Triple{8, 0, 8},
		Triple{8, 1, 1},
		Triple{8, 1, 3},
		Triple{8, 1, 5},
		Triple{8, 1, 7},
		Triple{8, 1, 9},
		Triple{8, 2, 2},
		Triple{8, 2, 4},
		Triple{8, 2, 6},
		Triple{8, 2, 8},
		Triple{8, 2, 10},
		Triple{8, 3, 3},
		Triple{8, 3, 5},
		Triple{8, 3, 7},
		Triple{8, 3, 9},
		Triple{8, 4, 4},
		Triple{8, 4, 6},
		Triple{8, 4, 8},
		Triple{8, 4, 10},
		Triple{8, 5, 5},
		Triple{8, 5, 7},
		Triple{8, 5, 9},
		Triple{8, 6, 6},
		Triple{8, 6, 8},
		Triple{8, 6, 10},
		Triple{8, 7, 7},
		Triple{8, 7, 9},
		Triple{8, 8, 8},
		Triple{8, 8, 10},
		Triple{8, 9, 9},
		Triple{9, 0, 1},
		Triple{9, 0, 3},
		Triple{9, 0, 5},
		Triple{9, 0, 7},
		Triple{9, 0, 9},
		Triple{9, 1, 2},
		Triple{9, 1, 4},
		Triple{9, 1, 6},
		Triple{9, 1, 8},
		Triple{9, 1, 10},
		Triple{9, 2, 3},
		Triple{9, 2, 5},
		Triple{9, 2, 7},
		Triple{9, 2, 9},
		Triple{9, 3, 4},
		Triple{9, 3, 6},
		Triple{9, 3, 8},
		Triple{9, 3, 10},
		Triple{9, 4, 5},
		Triple{9, 4, 7},
		Triple{9, 4, 9},
		Triple{9, 5, 6},
		Triple{9, 5, 8},
		Triple{9, 5, 10},
		Triple{9, 6, 7},
		Triple{9, 6, 9},
		Triple{9, 7, 8},
		Triple{9, 7, 10},
		Triple{9, 8, 9},
		Triple{9, 9, 10},
		Triple{10, 0, 0},
		Triple{10, 0, 2},
		Triple{10, 0, 4},
		Triple{10, 0, 6},
		Triple{10, 0, 8},
		Triple{10, 0, 10},
		Triple{10, 2, 2},
		Triple{10, 2, 4},
		Triple{10, 2, 6},
		Triple{10, 2, 8},
		Triple{10, 2, 10},
		Triple{10, 4, 4},
		Triple{10, 4, 6},
		Triple{10, 4, 8},
		Triple{10, 4, 10},
		Triple{10, 6, 6},
		Triple{10, 6, 8},
		Triple{10, 6, 10},
		Triple{10, 8, 8},
		Triple{10, 8, 10},
		Triple{10, 10, 10}	};

	ThreeIndex<double> radials(16, 11, 11);
	radint.type2(radial_triples_A, 19, 5, U, shellA, shellB, Am, Bm, radials);

	std::vector<Triple> radial_triples_B = {
		Triple{1, 4, 5},
		Triple{1, 5, 6},
		Triple{2, 3, 5},
		Triple{2, 4, 6},
		Triple{2, 5, 7},
		Triple{3, 3, 4},
		Triple{3, 2, 5},
		Triple{3, 4, 5},
		Triple{3, 3, 6},
		Triple{3, 5, 6},
		Triple{3, 4, 7},
		Triple{3, 6, 7},
		Triple{3, 5, 8},
		Triple{4, 2, 4},
		Triple{4, 1, 5},
		Triple{4, 3, 5},
		Triple{4, 2, 6},
		Triple{4, 4, 6},
		Triple{4, 3, 7},
		Triple{4, 5, 7},
		Triple{4, 4, 8},
		Triple{4, 6, 8},
		Triple{4, 5, 9},
		Triple{5, 2, 3},
		Triple{5, 1, 4},
		Triple{5, 3, 4},
		Triple{5, 0, 5},
		Triple{5, 2, 5},
		Triple{5, 4, 5},
		Triple{5, 1, 6},
		Triple{5, 3, 6},
		Triple{5, 5, 6},
		Triple{5, 2, 7},
		Triple{5, 4, 7},
		Triple{5, 6, 7},
		Triple{5, 3, 8},
		Triple{5, 5, 8},
		Triple{5, 7, 8},
		Triple{5, 4, 9},
		Triple{5, 6, 9},
		Triple{5, 5, 10},
		Triple{6, 1, 3},
		Triple{6, 0, 4},
		Triple{6, 2, 4},
		Triple{6, 1, 5},
		Triple{6, 3, 5},
		Triple{6, 0, 6},
		Triple{6, 2, 6},
		Triple{6, 4, 6},
		Triple{6, 1, 7},
		Triple{6, 3, 7},
		Triple{6, 5, 7},
		Triple{6, 2, 8},
		Triple{6, 4, 8},
		Triple{6, 6, 8},
		Triple{6, 3, 9},
		Triple{6, 5, 9},
		Triple{6, 7, 9},
		Triple{6, 4, 10},
		Triple{6, 6, 10},
		Triple{7, 1, 2},
		Triple{7, 0, 3},
		Triple{7, 2, 3},
		Triple{7, 1, 4},
		Triple{7, 3, 4},
		Triple{7, 0, 5},
		Triple{7, 2, 5},
		Triple{7, 4, 5},
		Triple{7, 1, 6},
		Triple{7, 3, 6},
		Triple{7, 5, 6},
		Triple{7, 0, 7},
		Triple{7, 2, 7},
		Triple{7, 4, 7},
		Triple{7, 6, 7},
		Triple{7, 1, 8},
		Triple{7, 3, 8},
		Triple{7, 5, 8},
		Triple{7, 7, 8},
		Triple{7, 2, 9},
		Triple{7, 4, 9},
		Triple{7, 6, 9},
		Triple{7, 8, 9},
		Triple{7, 3, 10},
		Triple{7, 5, 10},
		Triple{7, 7, 10},
		Triple{8, 0, 2},
		Triple{8, 1, 3},
		Triple{8, 0, 4},
		Triple{8, 2, 4},
		Triple{8, 1, 5},
		Triple{8, 3, 5},
		Triple{8, 0, 6},
		Triple{8, 2, 6},
		Triple{8, 4, 6},
		Triple{8, 1, 7},
		Triple{8, 3, 7},
		Triple{8, 5, 7},
		Triple{8, 0, 8},
		Triple{8, 2, 8},
		Triple{8, 4, 8},
		Triple{8, 6, 8},
		Triple{8, 1, 9},
		Triple{8, 3, 9},
		Triple{8, 5, 9},
		Triple{8, 7, 9},
		Triple{8, 2, 10},
		Triple{8, 4, 10},
		Triple{8, 6, 10},
		Triple{8, 8, 10},
		Triple{9, 0, 1},
		Triple{9, 1, 2},
		Triple{9, 0, 3},
		Triple{9, 2, 3},
		Triple{9, 1, 4},
		Triple{9, 3, 4},
		Triple{9, 0, 5},
		Triple{9, 2, 5},
		Triple{9, 4, 5},
		Triple{9, 1, 6},
		Triple{9, 3, 6},
		Triple{9, 5, 6},
		Triple{9, 0, 7},
		Triple{9, 2, 7},
		Triple{9, 4, 7},
		Triple{9, 6, 7},
		Triple{9, 1, 8},
		Triple{9, 3, 8},
		Triple{9, 5, 8},
		Triple{9, 7, 8},
		Triple{9, 0, 9},
		Triple{9, 2, 9},
		Triple{9, 4, 9},
		Triple{9, 6, 9},
		Triple{9, 8, 9},
		Triple{9, 1, 10},
		Triple{9, 3, 10},
		Triple{9, 5, 10},
		Triple{9, 7, 10},
		Triple{9, 9, 10},
		Triple{10, 0, 2},
		Triple{10, 0, 4},
		Triple{10, 2, 4},
		Triple{10, 0, 6},
		Triple{10, 2, 6},
		Triple{10, 4, 6},
		Triple{10, 0, 8},
		Triple{10, 2, 8},
		Triple{10, 4, 8},
		Triple{10, 6, 8},
		Triple{10, 0, 10},
		Triple{10, 2, 10},
		Triple{10, 4, 10},
		Triple{10, 6, 10},
		Triple{10, 8, 10}	};

	ThreeIndex<double> radials_B(16, 11, 11);
	radint.type2(radial_triples_B, 19, 5, U, shellB, shellA, Bm, Am, radials_B);
	for (Triple& t : radial_triples_B) radials(std::get<0>(t), std::get<2>(t), std::get<1>(t)) = radials_B(std::get<0>(t), std::get<1>(t), std::get<2>(t));

	rolled_up(5, 5, 5, radials, CA, CB, SA, SB, angint, values);
}
}
}
