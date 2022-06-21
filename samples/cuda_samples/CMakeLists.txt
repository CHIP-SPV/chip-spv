
# ARGN = additional source files
function(add_cuda_sample TEST_NAME MAIN_SOURCE)
  set(SOURCES ${MAIN_SOURCE} ${ARGN})
  set(EXEC_NAME ${TEST_NAME})
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" ${SOURCES})

  target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})

  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP)
  target_include_directories("${EXEC_NAME}"
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Common)
  install(TARGETS "${EXEC_NAME}"
    RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}/cuda_samples")

  add_test(NAME "${EXEC_NAME}"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}")
endfunction()

add_cuda_sample(cuda-vectorAdd vectorAdd/vectorAdd.cu)
add_cuda_sample(cuda-matrixMul matrixMul/matrixMul.cu)
add_cuda_sample(cuda-asyncAPI asyncAPI/asyncAPI.cu)
add_cuda_sample(cuda-template template/template.cu template/template_cpu.cpp)
add_cuda_sample(cuda-lambda lambda.cu)

# DISABLED: Due to the following error:
#
# ../cppOverload.cu:138:11: error: assigning to 'void (*)(const int *, int *, int)' from incompatible type '<overloaded function type>'
#  func1 = simple_kernel;
#          ^~~~~~~~~~~~~
#
# The correct overload is not compatible for some unknown reason. It might be
# a Clang issue which only appear in HIP-SPIRV path.
#
# add_cuda_sample(cuda-cppOverload cppOverload/cppOverload.cu)
