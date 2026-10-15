# Writes the build-info source (version + git commit) in script mode:
#   cmake -DSRC_DIR=... -DOCC_VERSION=... -DTEMPLATE=... -DOUT=... -P BuildInfo.cmake
# Run on every build rather than at configure time, so the recorded commit
# can't go stale; configure_file leaves OUT untouched when nothing changed,
# so an unchanged checkout triggers no recompilation.

set(OCC_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --always --dirty --abbrev=9 --exclude=*
        WORKING_DIRECTORY "${SRC_DIR}"
        OUTPUT_VARIABLE _commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _result
        ERROR_QUIET
    )
    if(_result EQUAL 0 AND _commit)
        set(OCC_GIT_COMMIT "${_commit}")
    endif()
endif()

configure_file("${TEMPLATE}" "${OUT}" @ONLY)
