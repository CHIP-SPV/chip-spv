
# ARGN = additional source files
function(add_cuda_sample TEST_NAME SUBDIR MAIN_SOURCE)
  set(SOURCES "")
  foreach(SOURCE ${MAIN_SOURCE} ${ARGN})
    list(APPEND SOURCES "${SUBDIR}/${SOURCE}")
  endforeach()

  set(EXEC_NAME ${TEST_NAME})
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" ${SOURCES})

  target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})

  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP)
  target_include_directories("${EXEC_NAME}"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common ${SUBDIR})
  install(TARGETS "${EXEC_NAME}"
    RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}/cuda_samples")

  add_test(NAME "${TEST_NAME}"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}")
  set_property(TEST "${EXEC_NAME}" PROPERTY LABELS "internal;cuda")

endfunction()

## 0_Simple

add_cuda_sample(cuda-vectorAdd 0_Simple/vectorAdd vectorAdd.cu)
add_cuda_sample(cuda-matrixMul 0_Simple/matrixMul matrixMul.cu)
add_cuda_sample(cuda-asyncAPI 0_Simple/asyncAPI asyncAPI.cu)
add_cuda_sample(cuda-template 0_Simple/template template.cu template_cpu.cpp)
add_cuda_sample(cuda-lambda 0_Simple/lambda lambda.cu)

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


add_cuda_sample(cuda-binomialoptions  4_Finance/binomialOptions binomialOptions.cpp  binomialOptions_gold.cpp  binomialOptions_kernel.cu)
add_cuda_sample(cuda-blackscholes  4_Finance/BlackScholes BlackScholes.cu  BlackScholes_gold.cpp)
add_cuda_sample(cuda-qrng  4_Finance/quasirandomGenerator quasirandomGenerator.cpp  quasirandomGenerator_gold.cpp  quasirandomGenerator_kernel.cu)

add_cuda_sample(cuda-mergesort  6_Advanced/mergeSort bitonic.cu  main.cpp  mergeSort.cu  mergeSort_host.cpp  mergeSort_validate.cpp)
add_cuda_sample(cuda-scalarprod  6_Advanced/scalarProd scalarProd_cpu.cpp  scalarProd.cu)
add_cuda_sample(cuda-scan  6_Advanced/scan scan.cu  main.cpp scan_gold.cpp)
add_cuda_sample(cuda-sortnet  6_Advanced/sortingNetworks bitonicSort.cu  main.cpp  oddEvenMergeSort.cu  sortingNetworks_validate.cpp)
add_cuda_sample(cuda-FDTD3d  6_Advanced/FDTD3d FDTD3dReference.cpp FDTD3dGPU.cu FDTD3d.cpp)
add_cuda_sample(cuda-sobolqrng  4_Finance/SobolQRNG sobol.cpp  sobol_gold.cpp  sobol_gpu.cu  sobol_primitives.cpp)
