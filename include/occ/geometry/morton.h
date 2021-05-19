#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace occ::geometry {

using integer_type = uint64_t;
using floating_type = double;

namespace constants {
// dimension powers
constexpr integer_type dim_pow_2{9}, dim_pow_1{3}, dim_pow_0{1};
// shifts
constexpr integer_type shift2A{18}, shift2B{36}, shift1A{6}, shift1B{12}, shift0A{2}, shift0B{4};
constexpr integer_type dilateM2{0x7fc0000ff80001ff}, // 0:9, 27:36, 54:63
                       dilateM1{0x01c0e070381c0e07}, // 0:3,  9:12, 18:21
                       dilateM0{0x9249249249249249}, // 0, 3, 6,  9, 12
                       dilateTZ{0x4924924924924924}, // 2, 5, 8, 11, 14
                       dilateTY{0x2492492492492492}, // 1, 4, 7, 10, 13
                       dilateTX{0x9249249249249249}, // 0, 3, 6,  9, 12
                       dilateT1{0xb6db6db6db6db6db}, // ~ dilateTZ
                       dilateT2{0xdb6db6db6db6db6d}, // ~ dilateTY
                       dilateT3{0x6db6db6db6db6db6}; // ~ dilateTX
constexpr floating_type inv_log2x3{0.48089834696}; // 1.0 / (ln(2) * 3);
constexpr integer_type max_level = (8 * 8 - 1) / 3; // ((sizeof(u64) in bits) - 1) / 3
}

struct MIndex
{
    integer_type code{1};
    struct Position
    {
        floating_type x, y, z;
    };

    integer_type level() const;
    floating_type size() const;

    Position center() const;

    MIndex operator+(const MIndex& other) const;
    MIndex operator-(const MIndex& other) const;
    MIndex parent() const;
    MIndex child(uint8_t idx) const;
    MIndex primal(integer_type lvl, integer_type idx) const;
    MIndex dual(integer_type lvl, integer_type idx) const;

    bool operator <(const MIndex& other) const { return code < other.code; }
    bool operator !=(const MIndex& other) const { return code != other.code; }
    bool operator ==(const MIndex& other) const { return code == other.code; }
    bool operator >(const MIndex& other) const { return code > other.code; }
};


struct MIndexHash
{
    std::size_t operator()(const occ::geometry::MIndex& k) const { return static_cast<size_t>(k.code); }
};

}
