add_library(occ_solvent
    "${CMAKE_CURRENT_SOURCE_DIR}/cosmo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/smd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvation_correction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp"
    ${OCC_SOLVENT_INCLUDE_FILES}
)

target_link_libraries(occ_solvent
    PUBLIC 
    occ_core
    occ_dft
)
target_include_directories(occ_solvent PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

if("${PCMSolver_FOUND}")
enable_language(Fortran)
target_link_libraries(occ_solvent PUBLIC PCMSolver::pcm "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")
endif()

target_compile_features(occ_solvent PUBLIC cxx_std_17)
target_compile_definitions(occ_solvent PUBLIC "${OCC_DEFINITIONS}")
