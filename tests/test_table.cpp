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


TEST_CASE("Table Eigen", "[table]")
{
    using tonto::io::Table;
    Eigen::MatrixXd r = Eigen::MatrixXd::Random(4, 3);
    Table t;

    t.set_column("random", r);
    t.print();
}
