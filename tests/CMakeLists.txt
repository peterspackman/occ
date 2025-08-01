CPMAddPackage(
    NAME catch
    GITHUB_REPOSITORY "catchorg/Catch2"
    GIT_TAG v3.5.2
)

list(APPEND CMAKE_MODULE_PATH "${catch_SOURCE_DIR}/extras")
include(CTest)
include(Catch)

add_executable(core_tests "${CMAKE_CURRENT_SOURCE_DIR}/core_tests.cpp")
target_link_libraries(core_tests PRIVATE occ_core Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(core_tests)

add_executable(crystal_tests "${CMAKE_CURRENT_SOURCE_DIR}/crystal_tests.cpp")
target_link_libraries(crystal_tests occ_crystal Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(crystal_tests)

add_executable(dft_tests "${CMAKE_CURRENT_SOURCE_DIR}/dft_tests.cpp")
target_link_libraries(dft_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(dft_tests)

add_executable(disp_tests "${CMAKE_CURRENT_SOURCE_DIR}/disp_tests.cpp")
target_link_libraries(disp_tests PRIVATE occ_disp Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(disp_tests)

add_executable(geometry_tests "${CMAKE_CURRENT_SOURCE_DIR}/geometry_tests.cpp")
target_link_libraries(geometry_tests PRIVATE occ_geometry occ_io Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(geometry_tests)

add_executable(gto_tests "${CMAKE_CURRENT_SOURCE_DIR}/gto_tests.cpp")
target_link_libraries(gto_tests PRIVATE occ_gto occ_dft Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(gto_tests)

add_executable(io_tests "${CMAKE_CURRENT_SOURCE_DIR}/io_tests.cpp")
target_link_libraries(io_tests PRIVATE occ_io Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(io_tests)

add_executable(io_periodic_grid_tests "${CMAKE_CURRENT_SOURCE_DIR}/io_periodic_grid_tests.cpp")
target_link_libraries(io_periodic_grid_tests PRIVATE occ_io occ_dft Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(io_periodic_grid_tests)

add_executable(wavefunction_io_tests "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction_io_tests.cpp")
target_link_libraries(wavefunction_io_tests PRIVATE occ_io Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(wavefunction_io_tests)

add_executable(interaction_tests "${CMAKE_CURRENT_SOURCE_DIR}/interaction_tests.cpp")
target_link_libraries(interaction_tests PRIVATE occ_interaction Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(interaction_tests)

add_executable(qm_tests "${CMAKE_CURRENT_SOURCE_DIR}/qm_tests.cpp")
target_link_libraries(qm_tests PRIVATE occ_qm Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(qm_tests)

add_executable(mp2_tests "${CMAKE_CURRENT_SOURCE_DIR}/mp2_tests.cpp")
target_link_libraries(mp2_tests PRIVATE occ_qm Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(mp2_tests)

add_executable(sht_tests "${CMAKE_CURRENT_SOURCE_DIR}/sht_tests.cpp")
target_link_libraries(sht_tests PRIVATE occ_sht Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(sht_tests)

add_executable(slater_tests "${CMAKE_CURRENT_SOURCE_DIR}/slater_tests.cpp")
target_link_libraries(slater_tests PRIVATE occ_slater Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(slater_tests)

add_executable(descriptors_tests "${CMAKE_CURRENT_SOURCE_DIR}/descriptors_tests.cpp")
target_link_libraries(descriptors_tests PRIVATE occ_descriptors Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(descriptors_tests)

add_executable(solvent_tests "${CMAKE_CURRENT_SOURCE_DIR}/solvent_tests.cpp")
target_link_libraries(solvent_tests PRIVATE occ_solvent occ_io Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(solvent_tests)

add_executable(xdm_tests "${CMAKE_CURRENT_SOURCE_DIR}/xdm_tests.cpp")
target_link_libraries(xdm_tests PRIVATE occ_xdm Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(xdm_tests)

add_executable(xtb_tests "${CMAKE_CURRENT_SOURCE_DIR}/xtb_tests.cpp")
target_link_libraries(xtb_tests PRIVATE occ_xtb Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(xtb_tests)

add_executable(cg_tests "${CMAKE_CURRENT_SOURCE_DIR}/cg_tests.cpp")
target_link_libraries(cg_tests PRIVATE occ_cg Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(cg_tests)

add_executable(dma_tests "${CMAKE_CURRENT_SOURCE_DIR}/dma_tests.cpp")
target_link_libraries(dma_tests PRIVATE occ_dma occ_io Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(dma_tests)

add_executable(volume_calculator_tests "${CMAKE_CURRENT_SOURCE_DIR}/volume_calculator_tests.cpp")
target_link_libraries(volume_calculator_tests PRIVATE occ_isosurface occ_main occ_io occ_crystal Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(volume_calculator_tests)

