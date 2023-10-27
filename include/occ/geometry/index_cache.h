#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace occ::geometry {

using std::size_t;

struct IndexCache {
    struct Layer {
        std::array<uint32_t, 4> element{0};
    };
    struct Row {
        std::array<uint32_t, 3> element{0};
    };

    size_t size;
    std::vector<Layer> layer0, layer1;
    std::vector<Row> row0, row1;
    std::array<uint32_t, 2> cell0{0}, cell1{0};
    std::array<uint32_t, 12> current_cell{0};

    IndexCache(size_t);
    void put(size_t, size_t, size_t, size_t);
    uint32_t get(size_t, size_t, size_t);
    void advance_cell();
    void advance_row();
    void advance_layer();
};

} // namespace occ::geometry
