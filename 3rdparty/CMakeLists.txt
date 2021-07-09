include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.x
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(scnlib
    GIT_REPOSITORY https://github.com/eliaskosunen/scnlib.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(scnlib)

FetchContent_Declare(gau2grid
    GIT_REPOSITORY https://github.com/dgasmith/gau2grid.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(gau2grid)

FetchContent_Declare(toml11
    GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(toml11)


if(NOT "${USE_SYSTEM_LIBINT2}")
set(LIBINT2_URL "https://github.com/evaleev/libint/releases/download/v2.7.0-beta.6/libint-2.7.0-beta.6.tgz")
FetchContent_Declare(Libint2
    URL "${LIBINT2_URL}"
)
FetchContent_MakeAvailable(Libint2)
add_library(Libint2::cxx ALIAS libint2_cxx)
endif()

if(NOT "${USE_SYSTEM_LIBXC}")
set(BUILD_TESTING OFF)
FetchContent_Declare(Libxc
    GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    GIT_TAG 5.0.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(Libxc)
add_library(Libxc::xc ALIAS xc)
unset(BUILD_TESTING)
endif()


