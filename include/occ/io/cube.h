#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>
#include <occ/core/atom.h>
#include <iostream>

namespace occ::io {


class Cube {
public:

    Cube();

    using AtomList = std::vector<core::Atom>;;

    template<typename F>
    void fill_data_from_function(F &func) {
	size_t N = steps(0) * steps(1) *steps(2);
	Mat3N points(3, N);

	data = Vec::Zero(N);

	for(int x = 0, i = 0; x < steps(0); x++) {
	    for(int y = 0; y < steps(1); y++) {
		for(int z = 0; z < steps(2); z++) {
		    points.col(i) = basis * Vec3(x, y, z) + origin;
		    i++;
		}
	    }
	}
	func(points, data);
    }

    void write_data_to_file(const std::string &);
    void write_data_to_stream(std::ostream &);

    std::string name{"cube file from OCC"};
    std::string description{"cube file from OCC"};
    Vec3 origin{0, 0, 0};
    Mat3 basis;
    IVec3 steps{11, 11, 11};
    AtomList atoms;
    Vec charges;
    Vec data;

private:
    void write_header_to_stream(std::ostream &);

};

}
