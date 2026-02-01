add_subdirectory(trajan/core)
# add_subdirectory(trajan/energy)
add_subdirectory(trajan/io)
add_subdirectory(trajan/main)

set(TRAJAN_INCLUDE_FILES
    "${TRAJAN_CORE_INCLUDE_FILES}" "${TRAJAN_IO_INCLUDE_FILES}"
    "${TRAJAN_MAIN_INCLUDE_FILES}" 
    # "${TRAJAN_MAIN_INCLUDE_FILES}" "${TRAJAN_ENERGY_INCLUDE_FILES}"
    PARENT_SCOPE)

set(TRAJAN_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)
