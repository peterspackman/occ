add_library(occ_solvent
    "${CMAKE_CURRENT_SOURCE_DIR}/draco.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameters.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/smd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvation_correction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp"
    # occ_scrf was merged into occ_solvent: the two are mutually dependent
    # (ReactionFieldEngine owns solvent types; ContinuumSolvationModel owns a
    # ReactionFieldEngine), so they form one library. Namespaces (occ::scrf /
    # occ::solvent) are unchanged.
    "${CMAKE_CURRENT_SOURCE_DIR}/../scrf/cosmo_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scrf/legacy_cosmo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scrf/reaction_field.cpp"
    ${OCC_SOLVENT_INCLUDE_FILES}
)

target_link_libraries(occ_solvent
    PUBLIC
    occ_core
    occ_numint
)
target_include_directories(occ_solvent PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_solvent PUBLIC cxx_std_20)
