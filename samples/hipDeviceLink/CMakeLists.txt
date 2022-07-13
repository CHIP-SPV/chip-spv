# Test symbol access
add_chip_binary(
	hipTestDeviceLink
	hipDeviceLink.cpp
	hipDeviceLinkRead.cpp
	hipDeviceLinkWrite.cpp)

target_compile_options(hipTestDeviceLink PRIVATE -fPIE -fgpu-rdc)
if(DEFINED LLVM_VERSION AND "${LLVM_VERSION}" VERSION_LESS 14)
  target_link_options(hipTestDeviceLink
    PRIVATE -fPIE -fgpu-rdc --hip-link ${HIP_LINK_LINE})
else()
  target_link_options(hipTestDeviceLink
    PRIVATE -fPIE -fgpu-rdc --hip-link --offload=spirv64 ${HIP_LINK_LINE}
    ${SPIRV_EMITTER_OPTS})
endif()

add_test(NAME hipTestDeviceLink
	 	COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipTestDeviceLink")

set_tests_properties(hipTestDeviceLink PROPERTIES
	PASS_REGULAR_EXPRESSION PASSED)
