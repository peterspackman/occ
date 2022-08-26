set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)

CPMAddPackage(
    NAME catch
    GITHUB_REPOSITORY "catchorg/Catch2"
    GIT_TAG v3.1.0
)

enable_testing()
add_subdirectory(core)
add_subdirectory(dft)
add_subdirectory(crystal)
add_subdirectory(geometry)
add_subdirectory(gto)
add_subdirectory(io)
add_subdirectory(interaction)
add_subdirectory(qm)
add_subdirectory(sht)
add_subdirectory(slater)
add_subdirectory(solvent)
