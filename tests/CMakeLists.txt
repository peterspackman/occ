CPMAddPackage(
    NAME catch
    GITHUB_REPOSITORY "catchorg/Catch2"
    GIT_TAG v3.5.2
)

set(CMAKE_UNITY_BUILD OFF)


#Use ccache to speed up rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache in ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(WARNING "ccache not found, compiling without it")  
endif()

#Use lld as faster alternative to ld
find_program(LLD_PROGRAM lld)
if(LLD_PROGRAM)
    message(STATUS "Using lld linker at ${LLD_PROGRAM}")
    set(CMAKE_LINKER "${LLD_PROGRAM}")
else()
    message(WARNING "lld linker not found, using default linker")
endif()


list(APPEND CMAKE_MODULE_PATH "${catch_SOURCE_DIR}/extras")
include(CTest)
include(Catch)

# Under ctest each test process runs with OCC_TEST_THREADS threads (see
# test_threads.cpp), so that `ctest -jN` does not run N processes that each
# claim every core; running a test binary directly keeps them all.
set(OCC_TEST_THREADS 2 CACHE STRING "Threads per test process under ctest")
add_library(occ_test_threads OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/test_threads.cpp")
target_link_libraries(occ_test_threads PRIVATE occ_core Catch2::Catch2)
# Extra arguments are further test properties.
function(occ_discover_tests target)
    target_link_libraries(${target} PRIVATE occ_test_threads)
    catch_discover_tests(${target} PROPERTIES ENVIRONMENT_MODIFICATION
        "OCC_TEST_THREADS=set:${OCC_TEST_THREADS}" ${ARGN})
endfunction()

add_executable(core_tests "${CMAKE_CURRENT_SOURCE_DIR}/core_tests.cpp")
target_link_libraries(core_tests PRIVATE occ_core Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(core_tests PRIVATE OCC_TEST_PROJECT_VERSION="${PROJECT_VERSION}")
occ_discover_tests(core_tests)

add_executable(crystal_tests "${CMAKE_CURRENT_SOURCE_DIR}/crystal_tests.cpp")
target_link_libraries(crystal_tests PRIVATE occ_crystal Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(crystal_tests)

add_executable(dft_tests "${CMAKE_CURRENT_SOURCE_DIR}/dft_tests.cpp")
target_link_libraries(dft_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(dft_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(dft_tests)

add_executable(dft_gradient_tests "${CMAKE_CURRENT_SOURCE_DIR}/dft_gradient_tests.cpp")
target_link_libraries(dft_gradient_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(dft_gradient_tests)

add_executable(isolated_xc_tests "${CMAKE_CURRENT_SOURCE_DIR}/isolated_xc_tests.cpp")
target_link_libraries(isolated_xc_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(isolated_xc_tests)

add_executable(disp_tests "${CMAKE_CURRENT_SOURCE_DIR}/disp_tests.cpp")
target_link_libraries(disp_tests PRIVATE occ_disp Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(disp_tests)

add_executable(geometry_tests "${CMAKE_CURRENT_SOURCE_DIR}/geometry_tests.cpp")
target_link_libraries(geometry_tests PRIVATE occ_geometry occ_io occ_isosurface Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(geometry_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(geometry_tests)

add_executable(gto_tests "${CMAKE_CURRENT_SOURCE_DIR}/gto_tests.cpp")
target_link_libraries(gto_tests PRIVATE occ_gto occ_dft Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(gto_tests)

add_executable(io_tests "${CMAKE_CURRENT_SOURCE_DIR}/io_tests.cpp")
target_link_libraries(io_tests PRIVATE occ_io occ_isosurface Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(io_tests)

add_executable(structure_format_test "${CMAKE_CURRENT_SOURCE_DIR}/structure_format_test.cpp")
target_link_libraries(structure_format_test PRIVATE occ_io Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(structure_format_test)

add_executable(io_periodic_grid_tests "${CMAKE_CURRENT_SOURCE_DIR}/io_periodic_grid_tests.cpp")
target_link_libraries(io_periodic_grid_tests PRIVATE occ_io occ_dft Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(io_periodic_grid_tests)

add_executable(wavefunction_io_tests "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction_io_tests.cpp")
target_link_libraries(wavefunction_io_tests PRIVATE occ_io occ_qm Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(wavefunction_io_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(wavefunction_io_tests)

add_executable(interaction_tests "${CMAKE_CURRENT_SOURCE_DIR}/interaction_tests.cpp")
target_link_libraries(interaction_tests PRIVATE occ_interaction Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(interaction_tests)

add_executable(qm_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/qm_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/qm_benchmarks.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/guess_tests.cpp"
)
target_link_libraries(qm_tests PRIVATE occ_qm occ_correlation Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(qm_tests PRIVATE OCC_DATA_DIR="${CMAKE_SOURCE_DIR}/share")
occ_discover_tests(qm_tests)

add_executable(qm_gradient_tests "${CMAKE_CURRENT_SOURCE_DIR}/qm_gradient_tests.cpp")
target_link_libraries(qm_gradient_tests PRIVATE occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(qm_gradient_tests)

add_executable(qm_hessian_tests "${CMAKE_CURRENT_SOURCE_DIR}/qm_hessian_tests.cpp")
target_link_libraries(qm_hessian_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(qm_hessian_tests)

add_executable(mp2_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/mp2_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mp2_benchmarks.cpp"
)
target_link_libraries(mp2_tests PRIVATE occ_qm occ_correlation Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(mp2_tests)

add_executable(cc_tests "${CMAKE_CURRENT_SOURCE_DIR}/cc_tests.cpp")
target_link_libraries(cc_tests PRIVATE occ_qm occ_cc occ_correlation Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(cc_tests)

add_executable(cc_density_tests "${CMAKE_CURRENT_SOURCE_DIR}/cc_density_tests.cpp")
target_link_libraries(cc_density_tests PRIVATE occ_qm occ_cc occ_correlation Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(cc_density_tests)

add_executable(sht_tests "${CMAKE_CURRENT_SOURCE_DIR}/sht_tests.cpp")
target_link_libraries(sht_tests PRIVATE occ_sht Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(sht_tests)

add_executable(slater_tests "${CMAKE_CURRENT_SOURCE_DIR}/slater_tests.cpp")
target_link_libraries(slater_tests PRIVATE occ_slater Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(slater_tests)

add_executable(descriptors_tests "${CMAKE_CURRENT_SOURCE_DIR}/descriptors_tests.cpp")
target_link_libraries(descriptors_tests PRIVATE occ_descriptors Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(descriptors_tests)

add_executable(rinse_tests "${CMAKE_CURRENT_SOURCE_DIR}/rinse_tests.cpp")
target_link_libraries(rinse_tests PRIVATE occ_descriptors occ_io occ_main Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(rinse_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(rinse_tests)

add_executable(solvent_tests "${CMAKE_CURRENT_SOURCE_DIR}/solvent_tests.cpp")
target_link_libraries(solvent_tests PRIVATE occ_solvent occ_driver occ_cg occ_crystal occ_dft occ_qm occ_io Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(solvent_tests PRIVATE OCC_DATA_DIR="${CMAKE_SOURCE_DIR}/share" OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(solvent_tests)

add_executable(xdm_tests "${CMAKE_CURRENT_SOURCE_DIR}/xdm_tests.cpp")
target_link_libraries(xdm_tests PRIVATE occ_xdm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(xdm_tests)

add_executable(xtb_tests "${CMAKE_CURRENT_SOURCE_DIR}/xtb_tests.cpp")
target_link_libraries(xtb_tests PRIVATE occ_xtb Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(xtb_tests)

add_executable(xtb_native_tests "${CMAKE_CURRENT_SOURCE_DIR}/xtb_native_tests.cpp")
target_link_libraries(xtb_native_tests PRIVATE occ_xtb Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(xtb_native_tests PRIVATE
    OCC_GFN2_DATA_DIR="${CMAKE_SOURCE_DIR}/share")
occ_discover_tests(xtb_native_tests)

add_executable(xtb_wavefunction_io_tests "${CMAKE_CURRENT_SOURCE_DIR}/xtb_wavefunction_io_tests.cpp")
target_link_libraries(xtb_wavefunction_io_tests PRIVATE occ_xtb occ_dma Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(xtb_wavefunction_io_tests PRIVATE
    OCC_GFN2_DATA_DIR="${CMAKE_SOURCE_DIR}/share")
occ_discover_tests(xtb_wavefunction_io_tests)

add_executable(cg_tests "${CMAKE_CURRENT_SOURCE_DIR}/cg_tests.cpp")
target_link_libraries(cg_tests PRIVATE occ_cg occ_driver Catch2::Catch2WithMain Threads::Threads)
# Resolve the GFN2 parameter file from the source share/ tree at runtime —
# matches the xtb / xtb_wavefunction_io test executables. Without this
# the test's DataDirGuard falls back to the relative `"share"` default
# and fails on a clean build dir.
target_compile_definitions(cg_tests PRIVATE
    OCC_GFN2_DATA_DIR="${CMAKE_SOURCE_DIR}/share")
occ_discover_tests(cg_tests)

add_executable(dma_tests "${CMAKE_CURRENT_SOURCE_DIR}/dma_tests.cpp")
target_link_libraries(dma_tests PRIVATE occ_dma occ_io occ_driver occ_mults Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(dma_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(dma_tests)

add_executable(volume_calculator_tests "${CMAKE_CURRENT_SOURCE_DIR}/volume_calculator_tests.cpp")
target_link_libraries(volume_calculator_tests PRIVATE occ_isosurface occ_main occ_io occ_crystal Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(volume_calculator_tests PRIVATE OCC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
occ_discover_tests(volume_calculator_tests)

add_executable(mesh_quality_tests "${CMAKE_CURRENT_SOURCE_DIR}/mesh_quality_tests.cpp")
target_link_libraries(mesh_quality_tests PRIVATE occ_isosurface Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(mesh_quality_tests)

add_executable(opt_tests "${CMAKE_CURRENT_SOURCE_DIR}/opt_tests.cpp")
target_link_libraries(opt_tests PRIVATE occ_opt nlohmann_json::nlohmann_json Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(opt_tests)

add_executable(cphf_tests "${CMAKE_CURRENT_SOURCE_DIR}/cphf_tests.cpp")
target_link_libraries(cphf_tests PRIVATE occ_dft occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(cphf_tests)

add_executable(ecp_tests "${CMAKE_CURRENT_SOURCE_DIR}/ecp_tests.cpp")
target_link_libraries(ecp_tests PRIVATE occ_qm occ_ints Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(ecp_tests ENVIRONMENT "OCC_DATA_PATH=${PROJECT_SOURCE_DIR}/share")

add_executable(ints_tests "${CMAKE_CURRENT_SOURCE_DIR}/ints_tests.cpp")
target_link_libraries(ints_tests PRIVATE occ_ints occ_qm Catch2::Catch2WithMain Threads::Threads)
occ_discover_tests(ints_tests)

add_executable(mults_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/mults_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/derivative_transform_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/short_range_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rigid_body_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/coordinate_system_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rotation_validation_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orient_reference_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cartesian_interaction_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cartesian_force_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cutoff_spline_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mults_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole_source_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole_hessian_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole_coarsening_test.cpp"
)
target_link_libraries(mults_tests PRIVATE occ_mults Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(mults_tests PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
occ_discover_tests(mults_tests)

add_custom_target(all_tests ALL
    DEPENDS
        core_tests
        crystal_tests
        dft_tests
        dft_gradient_tests
        isolated_xc_tests
        disp_tests
        geometry_tests
        gto_tests
        io_tests
        structure_format_test
        io_periodic_grid_tests
        wavefunction_io_tests
        interaction_tests
        qm_tests
        qm_gradient_tests
        qm_hessian_tests
        mp2_tests
        cc_tests
        cc_density_tests
        sht_tests
        slater_tests
        descriptors_tests
        rinse_tests
        solvent_tests
        xdm_tests
        xtb_tests
        xtb_native_tests
        xtb_wavefunction_io_tests
        cg_tests
        dma_tests
        volume_calculator_tests
        mesh_quality_tests
        opt_tests
        cphf_tests
        ecp_tests
        ints_tests
        mults_tests
)

# Timing harnesses: not tests, run by hand (see the usage at the top of each).
foreach(bench gradient_bench xc_bench cosx_grid_bench cosx_grad_bench rij_bench)
    add_executable(${bench} "${CMAKE_CURRENT_SOURCE_DIR}/${bench}.cpp")
endforeach()
target_link_libraries(gradient_bench PRIVATE occ_dft occ_io Threads::Threads)
target_link_libraries(xc_bench PRIVATE occ_dft occ_io Threads::Threads)
target_link_libraries(cosx_grid_bench PRIVATE occ_qm occ_io Threads::Threads)
target_link_libraries(cosx_grad_bench PRIVATE occ_qm occ_io Threads::Threads)
target_link_libraries(rij_bench PRIVATE occ_qm occ_io Threads::Threads)

# Lua binding tests: scripts run through `occ lua`, failing by raising an error
if(WITH_LUA_BINDINGS)
    function(occ_lua_test name script)
        add_test(NAME ${name}
            COMMAND $<TARGET_FILE:occ> lua "${CMAKE_CURRENT_SOURCE_DIR}/lua/${script}" ${ARGN})
        set_tests_properties(${name} PROPERTIES
            ENVIRONMENT "OCC_DATA_PATH=${CMAKE_SOURCE_DIR}/share;OCC_TEST_THREADS=${OCC_TEST_THREADS}")
    endfunction()

    occ_lua_test(lua_rinse rinse_tests.lua
        "${CMAKE_CURRENT_SOURCE_DIR}/data/rinse"
        "${CMAKE_CURRENT_SOURCE_DIR}/rinse_reference_data.h")
endif()
