#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <occ/core/macros.h>

namespace occ::geometry {

using integer_type = uint64_t;
using floating_type = double;

namespace constants {
// dimension powers
constexpr integer_type dim_pow_2{9}, dim_pow_1{3}, dim_pow_0{1};
// shifts
constexpr integer_type shift2A{18}, shift2B{36}, shift1A{6}, shift1B{12},
    shift0A{2}, shift0B{4};
constexpr integer_type dilateM2{0x7fc0000ff80001ff}, // 0:9, 27:36, 54:63
    dilateM1{0x01c0e070381c0e07},                    // 0:3,  9:12, 18:21
    dilateM0{0x9249249249249249},                    // 0, 3, 6,  9, 12
    dilateTZ{0x4924924924924924},                    // 2, 5, 8, 11, 14
    dilateTY{0x2492492492492492},                    // 1, 4, 7, 10, 13
    dilateTX{0x9249249249249249},                    // 0, 3, 6,  9, 12
    dilateT1{0xb6db6db6db6db6db},                    // ~ dilateTZ
    dilateT2{0xdb6db6db6db6db6d},                    // ~ dilateTY
    dilateT3{0x6db6db6db6db6db6};                    // ~ dilateTX
constexpr floating_type inv_log2x3{0.48089834696};   // 1.0 / (ln(2) * 3);
constexpr integer_type max_level =
    (8 * 8 - 1) / 3; // ((sizeof(u64) in bits) - 1) / 3
} // namespace constants

struct MIndex {
    integer_type code{1};
    struct Position {
        floating_type x, y, z;
    };

    OCC_ALWAYS_INLINE integer_type level() const {
        if (code == 0)
            return 0;
        return static_cast<integer_type>(std::floor(std::log2(code) / 3));
    }

    OCC_ALWAYS_INLINE floating_type size() const {
        return 1.0 / ((2 << level()));
    }

    OCC_ALWAYS_INLINE MIndex operator+(const MIndex &other) const {
        using namespace constants;
        return MIndex{
            (((code | dilateT1) + (other.code & dilateTZ)) & dilateTZ) |
            (((code | dilateT2) + (other.code & dilateTY)) & dilateTY) |
            (((code | dilateT3) + (other.code & dilateTX)) & dilateTX)};
    }

    OCC_ALWAYS_INLINE MIndex operator-(const MIndex &other) const {
        using namespace constants;
        return MIndex{
            (((code & dilateTZ) - (other.code & dilateTZ)) & dilateTZ) |
            (((code & dilateTY) - (other.code & dilateTY)) & dilateTY) |
            (((code & dilateTX) - (other.code & dilateTX)) & dilateTX)};
    }

    OCC_ALWAYS_INLINE MIndex parent() const {
        return MIndex{std::max(code >> 3, integer_type{1})};
    }

    OCC_ALWAYS_INLINE MIndex child(uint8_t idx) const {
        return MIndex{code << 3 | static_cast<integer_type>(idx)};
    }

    OCC_ALWAYS_INLINE void get_children(std::array<MIndex, 8> &arr) const {
        auto cleft3 = code << 3;

        for (integer_type idx = 0; idx < 8; idx++) {
            arr[idx] = MIndex{cleft3 | idx};
        }
    }

    OCC_ALWAYS_INLINE Position center() const {
        using namespace constants;
        integer_type bz = (code >> 2) & dilateM0;
        integer_type by = (code >> 1) & dilateM0;
        integer_type bx = code & dilateM0;

        const auto lvl = level();
        if (lvl > dim_pow_0) {
            bz = (bz | (bz >> shift0A) | (bz >> shift0B)) & dilateM1;
            by = (by | (by >> shift0A) | (by >> shift0B)) & dilateM1;
            bx = (bx | (bx >> shift0A) | (bx >> shift0B)) & dilateM1;
            if (lvl > dim_pow_1) {
                bz = (bz | (bz >> shift1A) | (bz >> shift1B)) & dilateM2;
                by = (by | (by >> shift1A) | (by >> shift1B)) & dilateM2;
                bx = (bx | (bx >> shift1A) | (bx >> shift1B)) & dilateM2;
                if (lvl > dim_pow_2) {
                    bz = bz | (bz >> shift2A) | (bz >> shift2B);
                    by = by | (by >> shift2A) | (by >> shift2B);
                    bx = bx | (bx >> shift2A) | (bx >> shift2B);
                }
            }
        }
        const integer_type mask_length = (1 << lvl) - 1;
        bz &= mask_length;
        by &= mask_length;
        bx &= mask_length;
        const auto s = size();
        const floating_type s2 = 2.0 * s;
        return Position{bx * s2 + s, by * s2 + s, bz * s2 + s};
    }

    OCC_ALWAYS_INLINE MIndex primal(integer_type lvl, integer_type idx) const {
        using namespace constants;
        integer_type k = 1 << (3 * lvl);
        integer_type k_plus_one = k << 1;
        MIndex vk = *this + MIndex{static_cast<integer_type>(idx)};
        integer_type dk = (vk - MIndex{k}).code;

        if (vk.code >= k_plus_one || ((dk & dilateTX) == 0) ||
            ((dk & dilateTY) == 0) || ((dk & dilateTZ) == 0)) {
            return MIndex{0};
        } else {
            return MIndex{vk.code << (3 * (max_level - lvl))};
        }
    }

    OCC_ALWAYS_INLINE std::array<MIndex, 8> primals(integer_type lvl) const {
        using namespace constants;
        integer_type k = 1 << (3 * lvl);
        integer_type k_plus_one = k << 1;
        std::array<MIndex, 8> result;
        for (integer_type idx = 0; idx < 8; idx++) {
            MIndex vk = *this + MIndex{idx};
            integer_type dk = (vk - MIndex{k}).code;
            if (vk.code >= k_plus_one || ((dk & dilateTX) == 0) ||
                ((dk & dilateTY) == 0) || ((dk & dilateTZ) == 0)) {
                result[idx] = MIndex{0};
            } else {
                result[idx] = MIndex{vk.code << (3 * (max_level - lvl))};
            }
        }
        return result;
    }

    OCC_ALWAYS_INLINE MIndex dual(integer_type lvl, integer_type idx) const {
        MIndex dk{code >> (3 * (constants::max_level - lvl))};
        return dk - MIndex{idx};
    }

    OCC_ALWAYS_INLINE void fill_duals(integer_type lvl,
                                      std::array<MIndex, 8> &values) const {
        for (integer_type idx = 0; idx < 8; idx++) {
            values[idx] = MIndex{code >> (3 * (constants::max_level - lvl))} -
                          MIndex{idx};
        }
    }

    bool operator<(const MIndex &other) const { return code < other.code; }
    bool operator!=(const MIndex &other) const { return code != other.code; }
    bool operator==(const MIndex &other) const { return code == other.code; }
    bool operator>(const MIndex &other) const { return code > other.code; }

    friend size_t hash_value(const MIndex &m) {
        return static_cast<size_t>(m.code);
    }
};

struct MIndexHash {
    uint64_t operator()(const MIndex &x) const noexcept { return x.code; }
};

} // namespace occ::geometry
