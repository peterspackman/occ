#include <tonto/geometry/index_cache.h>

namespace tonto::geometry {


IndexCache::IndexCache(size_t s) : size(s),
        layers({std::vector<unsigned[4]>(size * size),
                std::vector<unsigned[4]>(size * size)}),
        rows({std::vector<unsigned[3]>(size * size),
              std::vector<unsigned[3]>(size * size)}),
        cells{0}
{
}

void IndexCache::put(size_t x, size_t y, size_t edge, size_t index)
{
    if(edge >=4 && edge <=7) layers[1][y * size +x][edge - 4] = index;

    switch(edge) {
        case 5:
            cells[1][0] = index;
            break;
        case 6:
            rows[1][x][0] = index;
            break;
        case 11:
            rows[1][x][1] = index;
            break;
        case 10:
            rows[1][x][2] = index;
            cells[1][0] = index;
            break;
        default:
            break;
    }
    current_cell[edge] = index;
}

unsigned IndexCache::get(size_t x, size_t y, size_t edge)
{
    unsigned result{0};
    switch(edge) {
        case 0:
        case 1:
        case 2:
        case 3:
            result = layers[0][y * size + x][edge];
            break;
        case 4:
            result = rows[0][x][0];
            break;
        case 8:
            result = rows[0][x][1];
            break;
        case 9:
            result = rows[0][x][2];
            break;
        case 7:
        case 11:
            result = cells[0][1];
            break;
        default:
            break;
    }

    if (result > 0) return result;
    return current_cell[edge];
}

void IndexCache::advance_cell()
{
    std::swap(cells[0], cells[1]);
    for(size_t i = 0; i < 12; i++) current_cell[i] = 0;
}

void IndexCache::advance_row()
{
    std::swap(rows[0], rows[1]);
    for(size_t i = 0; i < 2; i++) cells[0][i] = 0;
}

void IndexCache::advance_layer()
{
    std::swap(layers[0], layers[1]);
    for(size_t i = 0; i < 2; i++) cells[0][i] = 0;
    for(auto &row : rows[0]) {
        for(size_t i = 0; i < 3; i++) row[i] = 0;
    }
}

}
