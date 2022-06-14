# Read includes, prepend each one with -I.
set(prop "$<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>")
set(CHIP_INCLUDES "$<$<BOOL:${prop}>:-I$<JOIN:${prop}, -I>>")

string(REPLACE ";" " " HIP_CLANG_OPTIONS "${OFFLOAD_ARCH_STR}")

# Generate cuspvc tool that is usable within build directory.
configure_file(cuspvc-build.in cuspvc-build.in @ONLY)
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuspvc
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/cuspvc-build.in)

# Generate cuspv tool that will be installed.
configure_file(cuspvc-install.in cuspvc-install @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cuspvc-install
  DESTINATION bin RENAME cuspvc)
