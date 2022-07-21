set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)

execute_process(COMMAND "${LLVM_CONFIG}" "--cmakedir"
		OUTPUT_VARIABLE LLVM_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE RES)

if(NOT RES EQUAL 0)
  message(FATAL_ERROR "failed to run llvm-config (${LLVM_CONFIG})")
endif()

message(STATUS "LLVM CMake directory: ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}" PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)

if(NOT DEFINED LLVM_VERSION)
  message(FATAL "Could not determine LLVM version.")
endif()

######################################

add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options("-fno-rtti")
endif()

if(NOT LLVM_ENABLE_EH)
  add_compile_options("-fno-exceptions")
endif()

add_library(LLVMHipDynMem MODULE HipDynMem.cpp)
add_library(LLVMHipStripUsedIntrinsics MODULE HipStripUsedIntrinsics.cpp)
install(TARGETS LLVMHipDynMem LLVMHipStripUsedIntrinsics
        LIBRARY DESTINATION lib/llvm
        ARCHIVE DESTINATION lib/llvm
        )

if("${LLVM_VERSION}" VERSION_GREATER 9.0)
  add_library(LLVMHipDefrost MODULE HipDefrost.cpp)
  install(TARGETS LLVMHipDefrost
          LIBRARY DESTINATION lib/llvm
          ARCHIVE DESTINATION lib/llvm
          )
endif()

if("${LLVM_VERSION}" VERSION_GREATER_EQUAL 11.0)
  add_library(LLVMHipPasses MODULE HipPasses.cpp HipBreakConstantGEPs.cpp
    HipDynMem.cpp HipStripUsedIntrinsics.cpp HipDefrost.cpp
    HipPrintf.cpp HipGlobalVariables.cpp HipTextureLowering.cpp HipAbort.cpp)

  if("${LLVM_VERSION}" VERSION_GREATER_EQUAL 14.0)
    set_target_properties(LLVMHipPasses PROPERTIES
      # HIP-Clang 14+ automatically searches for libLLVMHipSpvPasses.so in
      # <HIP-PATH>/lib where <HIP-PATH> is path given to --hip-path for
      # HIP program compilation.
      OUTPUT_NAME "LLVMHipSpvPasses"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endif()

  install(TARGETS LLVMHipPasses
          LIBRARY DESTINATION lib/llvm
          ARCHIVE DESTINATION lib/llvm)

endif()
