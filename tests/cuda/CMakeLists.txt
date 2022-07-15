
function(add_compile_test)
  set(options)
  set(args NAME COMPILE_OPTIONS)
  set(list_args SOURCES)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "TEST" # variable prefix
    "${options}"
    "${args}"
    "${list_args}")

  get_target_property(CHIP_INCLUDES CHIP INCLUDE_DIRECTORIES)

  add_test(NAME ${TEST_NAME}
    COMMAND bash -c "${CMAKE_BINARY_DIR}/bin/cuspvc ${TEST_COMPILE_OPTIONS} ${TEST_SOURCES} 2>&1"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

add_compile_test(NAME activemask SOURCES activemask.cu
  COMPILE_OPTIONS -fsyntax-only)
set_tests_properties(activemask PROPERTIES
  PASS_REGULAR_EXPRESSION "error: '__activemask' is unavailable: unsupported in CHIP-SPV")
