set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "")

add_executable(sycl_chip_interop sycl_chip_interop.cpp)
add_dependencies(sycl_chip_interop CHIP hipMatrixMul)
target_link_options(sycl_chip_interop PRIVATE -fsycl  -lze_loader )
target_link_libraries(sycl_chip_interop PRIVATE -L${CHIP_DIR_}/build/samples/sycl_hip_interop -lhipMatrixMul  -L${CHIP_DIR_}/build -lCHIP -lOpenCL -pthread)
target_compile_options(sycl_chip_interop PRIVATE -fsycl  -Wno-deprecated-declarations)
install(TARGETS sycl_chip_interop
RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

add_executable(sycl_chip_interop_usm sycl_chip_interop.cpp)
add_dependencies(sycl_chip_interop_usm CHIP hipMatrixMul)
target_link_options(sycl_chip_interop_usm PRIVATE -fsycl  -lze_loader )
target_link_libraries(sycl_chip_interop_usm PRIVATE -L${CHIP_DIR_}/build/samples/sycl_hip_interop -lhipMatrixMul  -L${CHIP_DIR_}/build -lCHIP -lOpenCL -pthread)
target_compile_options(sycl_chip_interop_usm PRIVATE -fsycl  -DUSM -Wno-deprecated-declarations)
install(TARGETS sycl_chip_interop_usm
RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")