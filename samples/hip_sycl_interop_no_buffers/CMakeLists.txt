add_subdirectory(onemkl_gemm_wrapper_no_buffers)

add_chip_binary(hip_sycl_interop_no_buffers hip_sycl_interop.cpp)
target_link_libraries(hip_sycl_interop_no_buffers onemkl_gemm_wrapper_no_buffers)

add_test(NAME "hip_sycl_interop_no_buffers"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hip_sycl_interop_no_buffers" 
         )

set_tests_properties("hip_sycl_interop_no_buffers" PROPERTIES
         PASS_REGULAR_EXPRESSION "${TEST_PASS}"
         TIMEOUT 60
         )