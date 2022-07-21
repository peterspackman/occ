find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_occpy MODULE "${CMAKE_CURRENT_SOURCE_DIR}/module.cpp")
target_compile_definitions(_occpy PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _occpy DESTINATION .)
