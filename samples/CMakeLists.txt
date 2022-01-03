
#include(/home/pvelesko/CHIP-SPV/install/cmake/CHIP/CHIPTargets.cmake)
option(SAVE_TEMPS "Save temporary compilation products" OFF)
option(VERBOSE "Verbose compilation" OFF)

if(SAVE_TEMPS)
  add_compile_options("--save-temps")
endif()

if(VERBOSE)
  add_compile_options("-v")
endif()

# ARGN = test args
function(add_chip_test EXEC_NAME TEST_NAME TEST_PASS SOURCE)

    set(TEST_EXEC_ARGS ${ARGN})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCE})

    target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})
    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" CHIP)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_DIR_}/HIP/include ${CHIP_DIR_}/include)
    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

    add_test(NAME "${TEST_NAME}"
             COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}" ${TEST_EXEC_ARGS}
             )

    set_tests_properties("${TEST_NAME}" PROPERTIES
             PASS_REGULAR_EXPRESSION "${TEST_PASS}")


endfunction()


# ARGN = sources
function(add_chip_binary EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_executable("${EXEC_NAME}" ${SOURCES})

    target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})
    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_DIR_}/HIP/include ${CHIP_DIR_}/include)
    target_link_libraries("${EXEC_NAME}" CHIP)

    install(TARGETS "${EXEC_NAME}"
            RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}")

endfunction()

function(add_chip_library EXEC_NAME)

    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)

    add_library("${EXEC_NAME}" STATIC ${SOURCES})

    target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})
    set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_link_libraries("${EXEC_NAME}" CHIP)
    target_include_directories("${EXEC_NAME}" PUBLIC ${CHIP_DIR_}/HIP/include ${CHIP_DIR_}/include)

endfunction()

set(SAMPLES
    hipmath
    hiptest
    bit_extract
    hcc_dialects
    fp16
    0_MatrixTranspose
    0_MatrixMultiply
    1_hipEvent
    2_vecadd
    3_shared_memory
    4_shfl
    5_2dshfl
    6_dynamic_shared
    7_streams
    9_unroll
    10_memcpy3D
    11_device
    hipStreamSemantics
    hipInfo
    hipSymbol
    hip-cuda
    hipDeviceLink # must be last in this list due to how hacky using multiple compilers with CMake is
)

# Add samples that depend on Intel's oneAPI compiler - if available.
find_program(ICPX_EXECUTABLE NAMES icpx)
if (ICPX_EXECUTABLE)
  list(APPEND SAMPLES
    sycl_hip_interop
    hip_sycl_interop
    hip_sycl_interop_no_buffers)
else()
  message(STATUS "icpx is not found: Not building SYCL interop samples.")
endif()

add_custom_target(samples DEPENDS ${SAMPLES})
foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()

#add_subdirectory(hip-cuda)
