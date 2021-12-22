# temporary
set(CMAKE_WARN_DEPRECATED OFF)
add_compile_options(-Wno-format-extra-args)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier \
-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result \
-Wno-delete-abstract-non-virtual-dtor")
set(CHIP_DIR_ ${CMAKE_CURRENT_SOURCE_DIR})
# end temporary

cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

project(CHIP-SPV
        VERSION 0.1.0
        DESCRIPTION "HIP implementation for runtimes that support SPIR-V"
        LANGUAGES C CXX)
include(cmake/chip_spv_cmake_macros.cmake)

# Cmake options that can be passed in by the user
option(ABORT_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in CHIP-SPV. If such a function is called, abort execution via std::abort()" ON)
option(LAZY_JIT "JIT kernels at the time of the first kernel invocation instead of at the start of the program execution" OFF)
option(CACHE_KERNELS "Save the compiled kernel to disk to speed up program startup" OFF)
option(VERBOSE "Verbose compilation" OFF)
option(BUILD_SHARED_LIBS "Build CHIP-SPV as a shared library" OFF)
option(ENFORCE_QUEUE_SYNCHRONIZATION "Enforce the correct HIP stream semantics of synchronizing queues with the default queue" ON)
set(HIP_SPIRV_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Path to HIP SPIRV directory containing HIP_SPIRV_DIR/share/kernellib.bc and HIP_SPIRV_DIR/lib/llvm")

# If custom HIP_SPIRV_DIR is given, check that kernellib.bc exists
if(NOT "${HIP_SPIRV_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" 
   AND NOT EXISTS "${HIP_SPIRV_DIR}/share/kernellib.bc" )
    message(FATAL_ERROR "Custom HIP_SPIRV_DIR=${HIP_SPIRV_DIR} was provided H that does not contain <HIP_SPIRV_DIR>/share/kernellib.bc")
endif()

set(CHIP_SPV_INCLUDE_FLAGS "" )
set(CHIP_SPV_LINK_FLAGS "" )
set(CHIP_SPV_COMPILE_FLAGS "" )
set(CHIP_SPV_DEFINITIONS "" )

if(ENFORCE_QUEUE_SYNCHRONIZATION)
  set(CHIP_SPV_DEFINITIONS "${CHIP_SPV_DEFINITIONS} -DENFORCE_QUEUE_SYNC")
endif()


set(HIP_PLATFORM spirv)
set(HIP_COMPILER clang)

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

if (ABORT_IF_NOT_IMPLEMENTED)
  set(CHIP_SPV_DEFINITIONS "${CHIP_SPV_DEFINITIONS} -DCHIP_ABORT_ON_UNIMPL")
endif()

set(CHIP_SPV_DEFINITIONS "${CHIP_SPV_DEFINITIONS} -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE")


set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/CHIP-SPV" CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")

# Compiler CHECK
 if((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
     message(FATAL_ERROR "this project requires clang >= 8.0")
   endif()
 else()
   message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

if (NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library_dynamic(OpenCL)
endif()

set(CHIP_SPV_INCLUDE_FLAGS ${CHIP_SPV_INCLUDE_FLAGS} src backend include . HIP/include ${OpenCL_INCLUDE_DIR})

# set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
# set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(PTHREAD_LIBRARY Threads::Threads)

set(CHIP_SRC 
    src/spirv.cc 
    src/spirv_hiprtc.cc 
    src/CHIPDriver.cc 
    src/CHIPBackend.cc 
    src/CHIPBindings.cc
    src/backend/Level0/Level0Backend.cc
    src/backend/OpenCL/CHIPBackendOpenCL.cc
    src/backend/OpenCL/SVMemoryRegion.cc
    src/logging.cc
)
set(DRIVER_SRC src/main.cc)

find_library_dynamic(ze_loader)
message(STATUS "\n\nCHIP-SPV_LIBRARIES=${CHIP-SPV_LIBRARIES}\n\n")

if (BUILD_SHARED_LIBS)
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP SHARED ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.so")
else()
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP STATIC ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.a")
endif()

set(HIP_PATH HIP)
set(OFFLOAD_ARCH_STR
  -D__HIP_PLATFORM_SPIRV__=
  -x hip --target=x86_64-linux-gnu
  $<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:--hip-device-lib-path=${CMAKE_INSTALL_PREFIX}/share>
  --hip-device-lib=kernellib.bc
  $<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_BINARY_DIR}/llvm_passes>
  $<INSTALL_INTERFACE:--hip-llvm-pass-path=${CMAKE_INSTALL_PREFIX}/lib/llvm>)
set(HIP_DEVICE_LIB 
"$<INSTALL_INTERFACE:--hip-device-lib-path=${HIP_SPIRV_DIR}/share>"
"$<BUILD_INTERFACE:--hip-device-lib-path=${CMAKE_CURRENT_BINARY_DIR}>")
set(HIP_LLVM_PASS_DIR
"$<INSTALL_INTERFACE:--hip-llvm-pass-path=${HIP_SPIRV_DIR}/lib/llvm>"
"$<BUILD_INTERFACE:--hip-llvm-pass-path=${CMAKE_CURRENT_BINARY_DIR}/llvm_passes>")


# For building on top of HIP-COMMON
target_compile_options(CHIP PUBLIC -D__HIP_PLATFORM_SPIRV__= ${CHIP_SPV_COMPILE_FLAGS})

target_link_libraries(CHIP PUBLIC ${OpenCL_LIBRARY} ${ze_loader_LIBRARY}
  ${CHIP-SPV_LIBRARIES} ${CHIP_SPV_LINK_FLAGS} ${PTHREAD_LIBRARY})
target_include_directories(CHIP
    PUBLIC
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/HIP/include;${CMAKE_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/HIP/include"
        "${OpenCL_INCLUDE_DIR}"
)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include(cmake/FindLLVM.cmake)
add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

add_subdirectory(HIP/tests/catch catch)

add_executable(driver ${DRIVER_SRC})
target_compile_options(driver PRIVATE ${OFFLOAD_ARCH_STR})
target_link_libraries(driver -stdlib=libstdc++ CHIP Threads::Threads ${OpenCL_LIBARY} ${CHIP-SPV_LIBRARIES})
add_dependencies(driver LLVMHipPasses)


set(CHIP_SAMPLE_BINDIR ${CMAKE_INSTALL_PREFIX}/bin/samples)
add_subdirectory(./samples samples)


add_dependencies(CHIP kernellib_bc)

install(TARGETS CHIP
        EXPORT CHIPTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${PROJECT_BINARY_DIR}/${CHIP_LIB_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
        FILE CHIPTargets.cmake
        NAMESPACE CHIP::
        DESTINATION  cmake/CHIP)
