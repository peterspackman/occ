#include <vector>

namespace tonto::geometry {

struct IndexCache
{
    size_t size;
    std::vector<unsigned[4]> layers[2];
    std::vector<unsigned[3]> rows[2];
    unsigned cells[2][2];
    unsigned current_cell[12];

    IndexCache(size_t);        
    void put(size_t, size_t, size_t, size_t);
    unsigned get(size_t, size_t, size_t);
    void advance_cell();
    void advance_row();
    void advance_layer();
};

}
