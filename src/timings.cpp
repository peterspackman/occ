#include "timings.h"

namespace tonto::timing {

static StopWatch<static_cast<size_t>(category::_group_count)> sw{};

time_point_t start(category cat)
{
    return sw.start(static_cast<size_t>(cat));
}

duration_t stop(category cat)
{
    return sw.stop(static_cast<size_t>(cat));
}

double total(category cat)
{
    return sw.read(static_cast<size_t>(cat));
}

void clear_all()
{
    sw.clear_all();
}

}
