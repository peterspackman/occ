#include <occ/geometry/morton.h>

namespace occ::geometry {

MIndex MIndex::operator+(const MIndex &other) const {
    using namespace constants;
    return MIndex{(((code | dilateT1) + (other.code & dilateTZ)) & dilateTZ) |
                  (((code | dilateT2) + (other.code & dilateTY)) & dilateTY) |
                  (((code | dilateT3) + (other.code & dilateTX)) & dilateTX)};
}

MIndex MIndex::operator-(const MIndex &other) const {
    using namespace constants;
    return MIndex{(((code & dilateTZ) - (other.code & dilateTZ)) & dilateTZ) |
                  (((code & dilateTY) - (other.code & dilateTY)) & dilateTY) |
                  (((code & dilateTX) - (other.code & dilateTX)) & dilateTX)};
}

integer_type MIndex::level() const {
    if (code == 0)
        return 0;
    return static_cast<integer_type>(
        std::floor(std::log(code) * constants::inv_log2x3));
}

MIndex MIndex::parent() const {
    return MIndex{std::max(code >> 3, integer_type{1})};
}

MIndex MIndex::child(uint8_t idx) const {
    return MIndex{code << 3 | static_cast<integer_type>(idx)};
}

floating_type MIndex::size() const { return 1.0 / ((2 << level())); }

MIndex::Position MIndex::center() const {
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

MIndex MIndex::primal(integer_type lvl, integer_type idx) const {
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

MIndex MIndex::dual(integer_type lvl, integer_type idx) const {
    MIndex dk{code >> (3 * (constants::max_level - lvl))};
    return dk - MIndex{idx};
}

} // namespace occ::geometry
