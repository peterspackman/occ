set(AVAILABLE_TOOLS
    trajan
)

foreach(TOOL_NAME ${AVAILABLE_TOOLS})
    string(TOUPPER ${TOOL_NAME} TOOL_NAME_UPPER)
    
    if(BUILD_${TOOL_NAME_UPPER})
        message(STATUS "Building tool: ${TOOL_NAME}")
        
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TOOL_NAME}")
            add_subdirectory(${TOOL_NAME})
        else()
            message(WARNING "Tool directory ${TOOL_NAME} not found, skipping...")
        endif()
    else()
        message(STATUS "Skipping tool: ${TOOL_NAME}")
    endif()
endforeach()

install(TARGETS ${AVAILABLE_TOOLS}
    EXPORT occ-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL 
)
