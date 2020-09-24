# FindLibxc.cmake
#
# Finds the Libxc header and library using the pkg-config program.
#
# Use this module by invoking find_package() as follows:
#   find_package(Libxc
#                [version] [EXACT]      # Minimum or EXACT version e.g. 2.5.0
#                [REQUIRED]             # Fail with error if Libxc is not found
#               )
#
# The behavior can be controlled by setting the following variables
#
#    LIBXC_SHARED_LIBRARY_ONLY                if true, will look for shared lib only; may be needed for some platforms
#                                             where linking errors or worse, e.g. duplicate static data, occur if
#                                             linking shared libraries against static Libint2 library.
#    PKG_CONFIG_PATH (environment variable)   Add the libint2 install prefix directory (e.g. /usr/local)
#                                             to specify where to look for libint2
#    CMAKE_MODULE_PATH                        Add the libint2 install prefix directory (e.g. /usr/local)
#                                             to specify where to look for libint2
#
# This will define the following CMake cache variables
#
#    LIBXC_FOUND           - true if libint2.h header and libint2 library were found
#    LIBXC_VERSION         - the libint2 version
#    LIBXC_INCLUDE_DIRS    - (deprecated: use the CMake IMPORTED targets listed below) list of libint2 include directories
#    LIBXC_LIBRARIES       - (deprecated: use the CMake IMPORTED targets listed below) list of libint2 libraries
#
# and the following imported targets
#
#     Libxc::xc          - library only
#
# Author: Eduard Valeyev - libint@valeyev.net

# need cmake 3.8 for cxx_std_11 compile feature
if(CMAKE_VERSION VERSION_LESS 3.8.0)
    message(FATAL_ERROR "This file relies on consumers using CMake 3.8.0 or greater.")
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    # point pkg-config to the location of this tree's libint2.pc
    set(ENV{PKG_CONFIG_PATH} "${CMAKE_CURRENT_LIST_DIR}/../../pkgconfig:$ENV{PKG_CONFIG_PATH}")
    if(LIBXC_FIND_QUIETLY)
        pkg_check_modules(PC_LIBXC QUIET libxc)
    else()
        pkg_check_modules(PC_LIBXC libxc)
    endif()
    set(LIBXC_VERSION ${PC_LIBXC_VERSION})

    find_path(LIBXC_INCLUDE_DIR
            NAMES xc.h
            PATHS ${PC_LIBXC_INCLUDE_DIRS}
    )

    if (LIBXC_SHARED_LIBRARY_ONLY)
        set(_LIBXC_LIB_NAMES "libxc.so" "libxc.dylib")
    else (LIBXC_SHARED_LIBRARY_ONLY)
        set(_LIBXC_LIB_NAMES "libxc.a")
    endif(LIBXC_SHARED_LIBRARY_ONLY)

    find_library(LIBXC_LIBRARY NAMES ${_LIBXC_LIB_NAMES} HINTS ${PC_LIBXC_LIBRARY_DIRS})

    mark_as_advanced(LIBXC_FOUND LIBXC_INCLUDE_DIR LIBXC_LIBRARY LIBXC_VERSION)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Libxc
            FOUND_VAR LIBXC_FOUND
            REQUIRED_VARS LIBXC_INCLUDE_DIR
            VERSION_VAR LIBXC_VERSION
            )

    if(LIBXC_FOUND)
        set(LIBXC_LIBRARIES ${LIBXC_LIBRARY})
        set(LIBXC_INCLUDE_DIRS ${LIBXC_INCLUDE_DIR} ${PC_LIBXC_INCLUDE_DIRS})
        # sanitize LIBXC_INCLUDE_DIRS: remove duplicates and non-existent entries
        list(REMOVE_DUPLICATES LIBXC_INCLUDE_DIRS)
        set(LIBXC_INCLUDE_DIRS_SANITIZED )
        foreach(DIR IN LISTS LIBXC_INCLUDE_DIRS)
            if (EXISTS ${DIR})
                list(APPEND LIBXC_INCLUDE_DIRS_SANITIZED ${DIR})
            endif()
        endforeach()
        set(LIBXC_INCLUDE_DIRS ${LIBXC_INCLUDE_DIRS_SANITIZED})
    endif()

    if(LIBXC_FOUND AND NOT TARGET Libxc::xc)
        add_library(Libxc::xc INTERFACE IMPORTED)
        set_target_properties(Libxc::xc PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${LIBXC_INCLUDE_DIR}"
                )
        set_target_properties(Libxc::xc PROPERTIES
                INTERFACE_LINK_LIBRARIES ${LIBXC_LIBRARY}
                )
    endif()

else(PKG_CONFIG_FOUND)

    message(FATAL_ERROR "Could not find the required pkg-config executable")

endif(PKG_CONFIG_FOUND)
