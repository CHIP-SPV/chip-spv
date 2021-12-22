add_subdirectory(onemkl_gemm_wrapper)

add_chip_binary(hip_sycl_interop hip_sycl_interop.cpp)
target_link_libraries(hip_sycl_interop onemkl_gemm_wrapper)

add_test(NAME "hip_sycl_interop"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hip_sycl_interop" 
         )

set_tests_properties("hip_sycl_interop" PROPERTIES
         PASS_REGULAR_EXPRESSION "${TEST_PASS}"
         TIMEOUT 60
         )