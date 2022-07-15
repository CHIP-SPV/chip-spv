# temporary
set(CMAKE_WARN_DEPRECATED OFF)
add_compile_options(-Wno-format-extra-args)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier \
-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result \
-Wno-delete-abstract-non-virtual-dtor")
set(CHIP_DIR_ ${CMAKE_CURRENT_SOURCE_DIR})

# end temporary
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(CHIP-SPV
  VERSION 0.1.0
  DESCRIPTION "HIP implementation for runtimes that support SPIR-V"
  LANGUAGES C CXX)

include(chip_spv_cmake_macros)

# Cmake options that can be passed in by the user
option(ERROR_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in CHIP-SPV. If such a function is called, throw hipErrorNotSupported" ON)
option(LAZY_JIT "JIT kernels at the time of the first kernel invocation instead of at the start of the program execution" OFF)
option(CACHE_KERNELS "Save the compiled kernel to disk to speed up program startup" OFF)
option(VERBOSE "Verbose compilation" OFF)
option(BUILD_SHARED_LIBS "Build CHIP-SPV as a shared library" ON)
option(ENFORCE_QUEUE_SYNCHRONIZATION "Enforce the correct HIP stream semantics of synchronizing queues with the default queue" ON)
option(PER_THREAD_DEFAULT_STREAM "Each thread has a thread local default stream. Default: OFF - use legacy stream" OFF)
option(LEVEL_ZERO_IMMEDIATE_QUEUES "Enable/disable the use of immediate command queues" OFF)
set(HIP_SPIRV_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Path to HIP SPIRV directory containing HIP_SPIRV_DIR/share/kernellib.bc and HIP_SPIRV_DIR/lib/llvm")

# If custom HIP_SPIRV_DIR is given, check that kernellib.bc exists
# TODO: Should this be deprecated now? This does not work for LLVM 14+.
if(NOT "${HIP_SPIRV_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}"
  AND NOT EXISTS "${HIP_SPIRV_DIR}/share/kernellib.bc")
  message(FATAL_ERROR "Custom HIP_SPIRV_DIR=${HIP_SPIRV_DIR} was provided H that does not contain <HIP_SPIRV_DIR>/share/kernellib.bc")
endif()

set(CHIP_SPV_LINK_FLAGS "")
set(CHIP_SPV_COMPILE_FLAGS "")

if(ENFORCE_QUEUE_SYNCHRONIZATION)
  list(APPEND CHIP_SPV_DEFINITIONS ENFORCE_QUEUE_SYNC)
endif()

if(PER_THREAD_DEFAULT_STREAM)
  list(APPEND CHIP_SPV_DEFINITIONS HIP_API_PER_THREAD_DEFAULT_STREAM)
endif()

if(LEVEL_ZERO_IMMEDIATE_QUEUES)
  list(APPEND CHIP_SPV_DEFINITIONS L0_IMM_QUEUES)
endif()

set(HIP_PLATFORM spirv)
set(HIP_COMPILER clang)

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  add_compile_options("-Wall")
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
else()
  list(APPEND CHIP_SPV_DEFINITIONS SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN)
endif()

if(ERROR_IF_NOT_IMPLEMENTED)
  list(APPEND CHIP_SPV_DEFINITIONS CHIP_ERROR_ON_UNIMPL)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/CHIP-SPV" CACHE PATH "Install path prefix" FORCE)
endif()

message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")

if(NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library_dynamic(OpenCL)
endif()

# Compiler CHECK
if((CMAKE_C_COMPILER_ID MATCHES "[Cc]lang") OR(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "this project requires clang >= 8.0")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0.0)
    message(WARNING "Deprecated clang version '${CMAKE_CXX_COMPILER_VERSION}'. \
            Support for Clang < 14.0 will be discontinued in the future.")
  endif()
else()
  message(FATAL_ERROR "this project must be compiled with clang. CMAKE_C_COMPILER_ID = ${CMAKE_C_COMPILER_ID}")
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

set(CHIP_SRC
  src/spirv.cc
  src/spirv_hiprtc.cc
  src/CHIPDriver.cc
  src/CHIPBackend.cc
  src/CHIPBindings.cc
  src/backend/Level0/CHIPBackendLevel0.cc
  src/backend/OpenCL/CHIPBackendOpenCL.cc
  src/backend/OpenCL/SVMemoryRegion.cc
  src/logging.cc
)
set(DRIVER_SRC src/main.cc)

find_library_dynamic(ze_loader)
message(STATUS "\n\nCHIP-SPV_LIBRARIES=${CHIP-SPV_LIBRARIES}\n\n")

if(BUILD_SHARED_LIBS)
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP SHARED ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.so")
else()
  message(STATUS "Buiding CHIP-SPV as a static library")
  add_library(CHIP STATIC ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.a")
endif()

include(cmake/FindLLVM.cmake)

set(HIP_PATH HIP)

if(DEFINED LLVM_VERSION AND "${LLVM_VERSION}" VERSION_LESS 14)
  # Definitions for the older forked, experimental HIP-Clang.
  set(HIP_LINK_LINE
    $<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}/share>
    $<INSTALL_INTERFACE:--hip-device-lib-path=${CMAKE_INSTALL_PREFIX}/share>
    --hip-device-lib=kernellib.bc
    $<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>
    $<INSTALL_INTERFACE:--hip-llvm-pass-path=${CMAKE_INSTALL_PREFIX}/lib/llvm>)
  set(OFFLOAD_ARCH_STR
    -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu ${HIP_LINK_LINE})
  set(HIP_OFFLOAD_ARCH -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu --hip-device-lib-path=${CMAKE_INSTALL_PREFIX}/share --hip-device-lib=kernellib.bc --hip-llvm-pass-path=${CMAKE_INSTALL_PREFIX}/lib/llvm)
else()
  # Definitions for upstream HIP-Clang.
  set(HIP_LINK_LINE
    $<BUILD_INTERFACE:--hip-path=${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:--hip-path=${CMAKE_INSTALL_PREFIX}>

    # By default RocmInstallationDetector::AddHIPIncludeArgs in the
    # HIP-Clang inserts include wrapper headers which cause compile
    # errors when compiling HIP sources for SPIR-V target.  This flag
    # excludes the wrappers.
    "-nohipwrapperinc")
  set(OFFLOAD_ARCH_STR

    # TODO: Consider integrating the '-D__HIP_PLATFORM_SPIRV__=' into
    # HIPSPV's tool chain.
    -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu
    --offload=spirv64 ${HIP_LINK_LINE})

  set(HIP_OFFLOAD_ARCH -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu --offload=spirv64 --hip-path=${CMAKE_INSTALL_PREFIX} -nohipwrapperinc)
endif()

# Workaround istead of trying to generate the CMake generator expression
string(REPLACE ";" " " HIP_OFFLOAD_ARCH_STR "${HIP_OFFLOAD_ARCH}")

# For building on top of HIP-COMMON
target_compile_options(CHIP PUBLIC -D__HIP_PLATFORM_SPIRV__= ${CHIP_SPV_COMPILE_FLAGS})
target_compile_definitions(CHIP PUBLIC ${CHIP_SPV_DEFINITIONS})

set(CHIP_SPV_LINK_FLAGS ${OpenCL_LIBRARY} ${CHIP-SPV_LIBRARIES})

# TODO: Check if pthread flag is needed for hipcc
target_link_libraries(CHIP PUBLIC ${CHIP_SPV_LINK_FLAGS} ${PTHREAD_LIBRARY})

# Previously these were set to SYSTEM which resulted in CMake picking up
# OpenCL headers from the system where the version might differ resulting in errors.
target_include_directories(CHIP
  PUBLIC
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cuspv>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/HIP/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuspv>"
  PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

add_executable(driver ${DRIVER_SRC})
target_compile_options(driver PRIVATE ${OFFLOAD_ARCH_STR})
target_link_libraries(driver -stdlib=libstdc++ CHIP Threads::Threads ${OpenCL_LIBARY} ${CHIP-SPV_LIBRARIES})
add_dependencies(driver LLVMHipPasses)

enable_testing()
add_subdirectory(HIP/tests/catch catch)
add_subdirectory(tests/cuda)

set(CHIP_SAMPLE_BINDIR ${CMAKE_INSTALL_PREFIX}/bin/samples)
add_subdirectory(./samples samples)

# Make CHIP depend on kernellib_bc and LLVMHipPasses for
# convenience. The CHIP module itself does not depend on these but
# HIP program compilation does.
add_dependencies(CHIP kernellib_bc LLVMHipPasses)
add_dependencies(samples CHIP)

# ####################
# Setup .hipVersion #
# ####################
# Install .hipVersion for hipcc. The file is used to detect a HIP
# runtime and clang extracts HIP version from it.
set(_versionInfo "# Auto-generated by cmake\n")
set(HIP_VERSION_MAJOR 5)
set(HIP_VERSION_MINOR 1)
set(HIP_VERSION_PATCH 0)
add_to_config(_versionInfo HIP_VERSION_MAJOR)
add_to_config(_versionInfo HIP_VERSION_MINOR)
add_to_config(_versionInfo HIP_VERSION_PATCH)

# Generate and install .hipVersion file.
file(WRITE ${PROJECT_BINARY_DIR}/bin/.hipVersion ${_versionInfo})
install(FILES ${PROJECT_BINARY_DIR}/bin/.hipVersion DESTINATION bin)

# ####################
# Setup .hipInfo #
# ####################
set(HIP_COMPILER clang)
set(HIP_RUNTIME spirv)
set(HIP_PLATFORM spirv)

string(TIMESTAMP _timestamp UTC)
set(_buildInfo "# Auto-generated by cmake on ${_timestamp} UTC\n")
message(STATUS "HIP Runtime: " ${HIP_RUNTIME})
message(STATUS "HIP Compiler: " ${HIP_COMPILER})
add_to_config(_buildInfo HIP_RUNTIME)
add_to_config(_buildInfo HIP_COMPILER)
add_to_config(_buildInfo HIP_OFFLOAD_ARCH_STR)

# Convert absolute path to library into -L<...> and -l<...> components
set(CHIP_SPV_LINK_FLAGS_STR "")

foreach(_library ${CHIP_SPV_LINK_FLAGS})
  get_filename_component(libpath ${_library} DIRECTORY)
  get_filename_component(libname ${_library} NAME)
  string(REGEX MATCH "lib(.*)\.(so*|a)" libname_stripped ${libname})
  set(libname_stripped ${CMAKE_MATCH_1})
  set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR} -L${libpath} -l${libname_stripped}")
endforeach()

# Add link directory fir libCHIP.so
set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR}" " -L${CMAKE_INSTALL_PREFIX}/lib -lCHIP")

# Add an rpath for libCHIP.so so that you don't need to add <CHIP_INSTALL_DIR> to LD_LIBRARY_PATH
set(CHIP_SPV_LINK_FLAGS_STR "${CHIP_SPV_LINK_FLAGS_STR}" " -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib")
set(HIP_LINK_OPTIONS ${CHIP_SPV_LINK_FLAGS_STR})
add_to_config(_buildInfo HIP_LINK_OPTIONS)

# Generate .hipInfo
file(WRITE "${PROJECT_BINARY_DIR}/lib/.hipInfo" ${_buildInfo})
install(FILES ${PROJECT_BINARY_DIR}/lib/.hipInfo DESTINATION lib)

# ####################
# Copy hipconfig, hipvars, etc to bin
# ####################
file(COPY "${CMAKE_SOURCE_DIR}/HIP/bin" DESTINATION "${PROJECT_BINARY_DIR}")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/bin DESTINATION . USE_SOURCE_PERMISSIONS)

set(CHIP_CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(CHIP_CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS CHIP
  EXPORT CHIPTargets
  LIBRARY DESTINATION ${CHIP_CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CHIP_CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${PROJECT_BINARY_DIR}/${CHIP_LIB_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/cmake DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
  FILE CHIPTargets.cmake
  NAMESPACE CHIP::
  DESTINATION cmake/CHIP)

add_subdirectory(bin)
