add_library(occ_xtb
    STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/xtb_wrapper.cpp"
    ${OCC_XTB_INCLUDE_FILES}
)

target_link_libraries(occ_xtb
    PUBLIC 
    occ_core
    occ_crystal
    occ_qm
    occ_dft
)

target_include_directories(occ_xtb PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)



if("${WITH_TBLITE}")
# need this to get the fortran libraries populated
enable_language(Fortran)
CPMAddPackage(
    NAME TBLite 
    GITHUB_REPOSITORY "tblite/tblite"
    OPTIONS "WITH_TESTS OFF"
    GIT_TAG "main"
)
add_library(TBLite::TBLite INTERFACE IMPORTED GLOBAL) 
# bit of a hack but the best I could do for linking gfortran etc. in a cross platform way
target_link_libraries(TBLite::TBLite INTERFACE tblite-lib tblite "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")

add_library(occ_tblite
    STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/tblite_wrapper.cpp"
    ${OCC_XTB_INCLUDE_FILES}
)

target_link_libraries(occ_tblite
    PUBLIC 
    occ_core
    occ_crystal
    occ_qm
    occ_dft
    TBLite::TBLite
)
message(STATUS "TBLite found: ${TBLITE_FOUND}")
message(STATUS "TBLite paths: ${TBLITE_LIBRARY}")

target_include_directories(occ_tblite PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
    ${TBLITE_INCLUDE_DIR}
)

target_compile_features(occ_tblite PUBLIC cxx_std_20)
target_compile_definitions(occ_tblite PUBLIC OCC_HAVE_TBLITE)

target_link_libraries(occ_xtb PUBLIC occ_tblite)
endif()
