set(CMAKE_UNITY_BUILD OFF)

CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    VERSION "11.1.0"
    GIT_TAG "11.1.0"
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.x"
)

if(NOT "${USE_SYSTEM_TBB}")

set(TBB_OPTIONS "")
list(APPEND TBB_OPTIONS "TBB_TEST OFF")
list(APPEND TBB_OPTIONS "TBB_STRICT OFF")
# Enable HWLOC for better NUMA topology detection on native builds, disable for Emscripten
if(EMSCRIPTEN)
    list(APPEND TBB_OPTIONS "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON")
else()
    list(APPEND TBB_OPTIONS "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH OFF")
endif()

# Windows MinGW-specific workarounds for TBB compilation issues
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND TBB_OPTIONS "CMAKE_CXX_FLAGS=-O2 -fno-stack-clash-protection -fno-fcf-protection")
endif()

CPMAddPackage(
    NAME oneTBB
    GITHUB_REPOSITORY "uxlfoundation/oneTBB"
    VERSION "2022.2.0"
    OPTIONS ${TBB_OPTIONS}
)
endif()

CPMAddPackage(
    NAME unordered_dense
    GITHUB_REPOSITORY "martinus/unordered_dense"
    VERSION "4.5.0"
)

CPMAddPackage(
    NAME scnlib
    GITHUB_REPOSITORY "eliaskosunen/scnlib"
    VERSION "4.0.1"
    OPTIONS "SIMDUTF_TESTS OFF" "ENABLE_FULL OFF"
)


# Header-only CLI11 doesn't emit the ConfigBase/Formatter vtables into a
# linkable location, which breaks executables in a shared-library build (the
# App ctor in occ.cpp references them). Default to precompiled CLI11 there.
# Set as a cache default before CPMAddPackage so CLI11's own option() picks it
# up; users can still override with -DCLI11_PRECOMPILED=OFF.
if(BUILD_SHARED_LIBS AND NOT DEFINED CLI11_PRECOMPILED)
    set(CLI11_PRECOMPILED ON CACHE BOOL "Precompile CLI11 (needed for shared-lib executable linking)")
endif()
CPMAddPackage(
    NAME CLI11
    GITHUB_REPOSITORY "CLIUtils/CLI11"
    VERSION "2.4.2"
)

if("${USE_MLX}")
CPMAddPackage(
    NAME mlx
    GITHUB_REPOSITORY "ml-explore/mlx"
    VERSION "0.6.0"
)
else()
CPMAddPackage(
    NAME "nlohmann_json"
    GITHUB_REPOSITORY "nlohmann/json"
    VERSION "3.11.3"
)

endif()

if(NOT "${USE_SYSTEM_EIGEN}")
CPMAddPackage(
    NAME eigen3
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    DOWNLOAD_ONLY YES  # Eigen CMakelists.txt not intended for library use
)

# Add the Eigen library target to link against
if("${eigen3_ADDED}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen3_SOURCE_DIR}")
endif()

endif()

CPMAddPackage(
    NAME gemmi
    GITHUB_REPOSITORY project-gemmi/gemmi
    VERSION "0.6.5"
    DOWNLOAD_ONLY YES
)

# Add Gemmi as a header only library for linking
if("${gemmi_ADDED}")
    add_library(gemmi::gemmi INTERFACE IMPORTED GLOBAL)
    target_include_directories(gemmi::gemmi INTERFACE "${gemmi_SOURCE_DIR}/include")
endif()

CPMAddPackage(
    NAME LBFGSpp
    GITHUB_REPOSITORY yixuan/LBFGSpp
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

# Add LBFGSpp as a header only library for linking
if("${LBFGSpp_ADDED}")
    add_library(LBFGSpp::LBFGSpp INTERFACE IMPORTED GLOBAL)
    target_include_directories(LBFGSpp::LBFGSpp INTERFACE "${LBFGSpp_SOURCE_DIR}/include")
endif()

if("${USE_QCINT}")
    set(LIBCINT_REPOSITORY "sunqm/qcint")
else()
    set(LIBCINT_REPOSITORY "peterspackman/libcint")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBCINT_BUILD_TYPE "CMAKE_BUILD_TYPE Release")
endif ()

# Add compiler flags to fix the implicit function declaration error
set(LIBCINT_C_FLAGS "-Wno-implicit-function-declaration -Wno-deprecated-non-prototype -D_GNU_SOURCE")

CPMAddPackage(
    NAME libcint
    GITHUB_REPOSITORY "${LIBCINT_REPOSITORY}"
    GIT_TAG master
    OPTIONS
        "WITH_FORTRAN OFF"
        "WITH_CINT2_INTERFACE OFF"
        "ENABLE_STATIC ON"
        "BUILD_SHARED_LIBS OFF"
        "PYPZPX ON"
        "BUILD_MARCH_NATIVE ON"
        "WITH_RANGE_COULOMB ON"
        "CMAKE_C_FLAGS ${LIBCINT_C_FLAGS}"
        ${LIBCINT_BUILD_TYPE}
    VERSION "6.1.2"
)

# Apply the flags to the target after it's created
if(TARGET cint)
    target_compile_options(cint PRIVATE 
        -Wno-implicit-function-declaration
        -Wno-deprecated-non-prototype
    )
endif()

add_library(libcint::libcint INTERFACE IMPORTED GLOBAL) 
target_link_libraries(libcint::libcint INTERFACE cint)
target_include_directories(
    libcint::libcint INTERFACE
    "${libcint_SOURCE_DIR}/include"
    "${libcint_SOURCE_DIR}/src"
    "${libcint_BINARY_DIR}/include"
)

if(NOT "${USE_SYSTEM_LIBXC}")
CPMAddPackage(
    NAME Libxc
    URL "https://gitlab.com/libxc/libxc/-/archive/6.2.2/libxc-6.2.2.tar.gz"
    OPTIONS
        "BUILD_TESTING OFF"
        "ENABLE_XHOST ${ENABLE_HOST_OPT}"
        "BUILD_FPIC ${OCC_ENABLE_PIC}"
        "CMAKE_POLICY_VERSION_MINIMUM 3.5"
)
add_library(Libxc::xc INTERFACE IMPORTED GLOBAL) 
target_link_libraries(Libxc::xc INTERFACE xc)
target_include_directories(
    Libxc::xc INTERFACE
    "${Libxc_SOURCE_DIR}/src"
    "${Libxc_BINARY_DIR}/src"
    "${Libxc_BINARY_DIR}"
)

endif()


if("${WITH_PYTHON_BINDINGS}")
CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY "wjakob/nanobind"
    GIT_TAG "v2.9.2"
)
endif()

if("${WITH_LUA_BINDINGS}")
# Upstream Lua ships only a Makefile, so fetch the source and build a
# plain static library ourselves. Keeping it in-tree avoids depending on
# a third-party CMake fork.
CPMAddPackage(
    NAME lua
    GITHUB_REPOSITORY "lua/lua"
    GIT_TAG "v5.4.7"
    DOWNLOAD_ONLY YES
)

if(lua_ADDED)
    set(LUA_CORE_SRC
        lapi.c lcode.c lctype.c ldebug.c ldo.c ldump.c lfunc.c lgc.c
        llex.c lmem.c lobject.c lopcodes.c lparser.c lstate.c lstring.c
        ltable.c ltm.c lundump.c lvm.c lzio.c
    )
    set(LUA_LIB_SRC
        lauxlib.c lbaselib.c lcorolib.c ldblib.c liolib.c lmathlib.c
        loadlib.c loslib.c lstrlib.c ltablib.c lutf8lib.c linit.c
    )
    set(LUA_ALL_SRC ${LUA_CORE_SRC} ${LUA_LIB_SRC})
    list(TRANSFORM LUA_ALL_SRC PREPEND "${lua_SOURCE_DIR}/")
    add_library(occ_lua_runtime STATIC ${LUA_ALL_SRC})
    target_include_directories(occ_lua_runtime PUBLIC "${lua_SOURCE_DIR}")
    set_target_properties(occ_lua_runtime PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 99
    )
    if(APPLE)
        target_compile_definitions(occ_lua_runtime PUBLIC LUA_USE_MACOSX)
    elseif(UNIX)
        target_compile_definitions(occ_lua_runtime PUBLIC LUA_USE_LINUX)
        target_link_libraries(occ_lua_runtime PUBLIC dl)
    endif()
    if(UNIX)
        target_link_libraries(occ_lua_runtime PUBLIC m)
    endif()
    add_library(lua::lua ALIAS occ_lua_runtime)
endif()

# LuaBridge3 — header-only Lua binding library. Replaced sol2 in
# the binding layer for substantially faster compile times
# (~50% wall-time and ~67% backend-codegen reduction on the
# interaction_bindings PoC). Lua-headers must be included BEFORE
# LuaBridge ones; eigen_conv.h handles that for every binding TU.
CPMAddPackage(
    NAME LuaBridge3
    GITHUB_REPOSITORY "kunitoki/LuaBridge3"
    GIT_TAG "3.0-rc11"
    DOWNLOAD_ONLY YES
)

if(LuaBridge3_ADDED)
    add_library(occ_luabridge3 INTERFACE)
    target_include_directories(occ_luabridge3
        INTERFACE "${LuaBridge3_SOURCE_DIR}/Source")
    target_link_libraries(occ_luabridge3 INTERFACE lua::lua)
    target_compile_features(occ_luabridge3 INTERFACE cxx_std_20)
    add_library(luabridge3::luabridge3 ALIAS occ_luabridge3)
endif()

# linenoise: tiny single-file line editor for the `occ lua` REPL.
# POSIX-only (Linux/macOS/BSD). Windows builds skip the REPL path
# entirely and fall back to stdin-slurp, so we only fetch the source
# on non-Windows platforms.
if(NOT WIN32)
CPMAddPackage(
    NAME linenoise
    GITHUB_REPOSITORY "antirez/linenoise"
    GIT_TAG "1.0"
    DOWNLOAD_ONLY YES
)

if(linenoise_ADDED)
    add_library(occ_linenoise STATIC
        "${linenoise_SOURCE_DIR}/linenoise.c"
    )
    target_include_directories(occ_linenoise PUBLIC
        "${linenoise_SOURCE_DIR}"
    )
    set_target_properties(occ_linenoise PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 99
    )
    add_library(linenoise::linenoise ALIAS occ_linenoise)
endif()
endif()
endif()
