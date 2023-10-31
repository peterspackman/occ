#pragma once

namespace occ::io {

struct BeckeGridSettings {
    size_t max_angular_points{302};
    size_t min_angular_points{50};
    size_t radial_points{65};
    double radial_precision{1e-8};
    bool reduced_first_row_element_grid{true};
    std::string pruning_scheme{"nwchem"};

    inline bool operator==(const BeckeGridSettings &rhs) const {
        return (max_angular_points == rhs.max_angular_points) &&
               (min_angular_points == rhs.min_angular_points) &&
               (radial_points == rhs.radial_points) &&
               (radial_precision == rhs.radial_precision);
    }

    inline bool operator!=(const BeckeGridSettings &rhs) const {
        return !(*this == rhs);
    }
};

} // namespace occ::io
