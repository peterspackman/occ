add_library(
    libocc
    $<TARGET_OBJECTS:occ_core_obj>
    $<TARGET_OBJECTS:occ_qm_obj>
    $<TARGET_OBJECTS:occ_dft_obj>
    $<TARGET_OBJECTS:occ_disp_obj>
    $<TARGET_OBJECTS:occ_dma_obj>
    $<TARGET_OBJECTS:occ_driver_obj>
    $<TARGET_OBJECTS:occ_solvent_obj>
    $<TARGET_OBJECTS:occ_interaction_obj>
    $<TARGET_OBJECTS:occ_isosurface_obj>
    $<TARGET_OBJECTS:occ_crystal_obj>
    $<TARGET_OBJECTS:occ_slater_obj>
    $<TARGET_OBJECTS:occ_geometry_obj>
    $<TARGET_OBJECTS:occ_gto_obj>
    $<TARGET_OBJECTS:occ_sht_obj>
    $<TARGET_OBJECTS:occ_descriptors_obj>
    $<TARGET_OBJECTS:occ_cg_obj>
    $<TARGET_OBJECTS:occ_elastic_fit_obj>
    $<TARGET_OBJECTS:occ_mults_obj>
)
target_link_libraries(libocc PUBLIC occ_xtb)
target_include_directories(
    libocc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
set_target_properties(libocc PROPERTIES PREFIX "")
install(
    TARGETS libocc
    EXPORT liboccTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/gau2grid/gau2grid
    DESTINATION include
)
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../src/3rdparty/libecpint/libecpint.hpp
    DESTINATION include
)
