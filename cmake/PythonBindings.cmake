# PythonBindings.cmake
# CMake module for building and configuring Python bindings with nanobind

# Function to set up Python stub generation for a nanobind module
#
# Arguments:
#   TARGET - The nanobind target name
#   MODULE - The Python module name
#   OUTPUT_DIR - Directory for generated stub files
function(add_nanobind_stubs TARGET MODULE OUTPUT_DIR)
    if(SKBUILD)
        # When building with scikit-build, generate stubs at install time
        # This ensures the module is built and available before stub generation
        nanobind_add_stub(
            ${TARGET}_stub
            INSTALL_TIME
            MODULE ${MODULE}
            OUTPUT_PATH occpy
            PYTHON_PATH "\${CMAKE_INSTALL_PREFIX}/occpy"
            MARKER_FILE py.typed
        )

        message(STATUS "Python stub generation enabled for ${MODULE} (install-time)")
    else()
        # For non-scikit-build, provide a custom target for manual stub generation
        add_custom_target(${TARGET}_stubs
            COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen
                    -m ${MODULE}
                    -o ${OUTPUT_DIR}
                    -M py.typed
            DEPENDS ${TARGET}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TARGET}>
            COMMENT "Generating Python stub files for ${MODULE}"
            VERBATIM
        )
        message(STATUS "Python stub generation available via '${TARGET}_stubs' target")
    endif()
endfunction()
