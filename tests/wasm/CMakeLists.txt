# Find Node.js and npm
find_program(NODE_EXECUTABLE NAMES node nodejs)
find_program(NPM_EXECUTABLE NAMES npm)
if(NOT NODE_EXECUTABLE)
    message(FATAL_ERROR "Node.js not found! Required for WASM tests.")
endif()
if(NOT NPM_EXECUTABLE)
    message(FATAL_ERROR "npm not found! Required for WASM tests.")
endif()

# Create test directory in build
set(WASM_TEST_DIR ${CMAKE_BINARY_DIR}/wasm_tests)
file(MAKE_DIRECTORY ${WASM_TEST_DIR})

# Copy test files to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_core.js
    ${WASM_TEST_DIR}/test_core.js
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/package.json
    ${WASM_TEST_DIR}/package.json
    COPYONLY
)

# Add custom target to install dependencies and prepare tests
add_custom_target(prepare_wasm_tests
    COMMAND ${NPM_EXECUTABLE} install
    WORKING_DIRECTORY ${WASM_TEST_DIR}
    COMMENT "Installing Node.js dependencies for WASM tests"
)

# Add custom target to run tests
add_custom_target(wasm_tests
    COMMAND ${NODE_EXECUTABLE} ${WASM_TEST_DIR}/test_core.js
    WORKING_DIRECTORY ${WASM_TEST_DIR}
    DEPENDS occ_wasm prepare_wasm_tests
    COMMENT "Running WASM tests"
)

# Add to CTest
add_test(
    NAME wasm_core_tests
    COMMAND ${CMAKE_COMMAND} --build . --target wasm_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
