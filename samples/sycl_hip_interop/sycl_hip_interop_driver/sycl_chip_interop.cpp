#include <stdint.h>
#include "level_zero/ze_api.h"
#include <CL/sycl.hpp>
#include <CL/sycl/stl.hpp>
#include "CL/sycl/backend/level_zero.hpp"

#include "sycl_chip_interop.h"

// include this file with relative path,
// because -I<CHIP>/include causes CL header conflict
#include "../../../include/hip/hip_interop.h"

#include <vector>
#include <iostream>

using namespace sycl;

// #define USM

const int WIDTH = 16;

const int size = 1024;

// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float *__restrict A,
                                const float *__restrict B,
                                float *__restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
      float acc = 0.0f;
      for (uint k = 0; k < WIDTH; k++) {
        acc += B[i * WIDTH + k] * A[k * WIDTH + j];
      }
      C[i * WIDTH + j] = acc;
    }
  }
}

int main() {
  std::string envVar = std::getenv("CHIP_BE");
  if (!envVar.compare("opencl")) {
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    exit(0);
  }
  queue myQueue;

#ifdef USM
  // USM allocator for data of type int in shared memory
  typedef usm_allocator<float, usm::alloc::shared> vec_alloc;
  // Create allocator for device associated with q
  vec_alloc myAlloc(myQueue);
  // Create std vectors with the allocator
  std::vector<float, vec_alloc> a(size, myAlloc), b(size, myAlloc),
      c(size, myAlloc);
#else
  std::vector<float> a(size), b(size), c(size);
#endif

  for (int i = 0; i < size; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }

  // Get pointer to vector data for access in kernel
  auto A = a.data();
  auto B = b.data();
  auto C = c.data();

  context Ctx = myQueue.get_context();
  std::vector<device> Devs = Ctx.get_devices();

  sycl::device Dev{sycl::default_selector{}};
  sycl::platform Plt = Dev.get_platform();

  if (Devs.size() >= 1) {
    // Initialize CHIP-SPV Level-Zero Backend via providing native runtime
    // information
    uintptr_t Args[] = {
        (uintptr_t)Plt.template get_native<sycl::backend::level_zero>(),
        (uintptr_t)Devs[0].template get_native<sycl::backend::level_zero>(),
        (uintptr_t)Ctx.template get_native<sycl::backend::level_zero>(),
        (uintptr_t)myQueue.template get_native<sycl::backend::level_zero>()};
    hipInitFromNativeHandles(Args, 4);

#ifdef USM
    // Run GEMM test via CHIP-SPV Level-Zero Backend and USM data transfer
    hipMatrixMultiplicationUSMTest(A, B, C, WIDTH, WIDTH);
#else
    // Run GEMM test via CHIP-SPV Level-Zero Backend
    hipMatrixMultiplicationTest(A, B, C, WIDTH, WIDTH);
#endif
  }

  // Alloc memory on host
  std::vector<float> c_ref(size);
  auto C_ref = c_ref.data();
  // Get CPU result for refereence
  matrixMultiplyCPUReference(A, B, C_ref);

  int err = 0;
  for (int i = 0; i < 256; i++) {
    err += (int)(c[i] - c_ref[i]) * 1000;
  }

  if (err != 0) {
    std::cout << "FAIL: " << err << " failures \n";
  } else {
    std::cout << "PASSED\n";
  }

  return 0;
}
