#include "table.h"
#include "catch.hpp"

TEST_CASE("Table constructor", "[table]")
{
    using tonto::io::Table;

    Table t({"test", "columns"});
    std::vector<int> test{1, 2, 3};
    std::vector<std::string> columns{"this", "is", "a", "test"};
    t.set_column("test", test);
    t.set_column("columns", columns);
    t.print();
}
