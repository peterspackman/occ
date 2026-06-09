add_library(occ_opt_obj
    OBJECT
    angle_coordinate.cpp
    berny_optimizer.cpp
    bond_coordinate.cpp
    dihedral_coordinate.cpp
    internal_coordinates.cpp
    linear_search.cpp
    pseudoinverse.cpp
)
add_library(occ_opt $<TARGET_OBJECTS:occ_opt_obj>)
target_include_directories(occ_opt_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_opt_obj PUBLIC occ_core fmt::fmt Eigen3::Eigen)

target_include_directories(occ_opt PUBLIC ${OCC_INCLUDE_DIR})

target_link_libraries(occ_opt PUBLIC occ_core fmt::fmt Eigen3::Eigen)
