
#include "hip/hip_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(1);                                                                \
    }                                                                         \
  }

__global__ void addOne(int *__restrict A) {
  const uint i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  A[i] = A[i] + 1;
}

int main() {
  int numBlocks = 5120000;
  int dimBlocks = 32;
  const size_t NUM = numBlocks * dimBlocks;
  int *A_h, *A_d, *Ref;

  static int device = 0;
  CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);

  A_h = (int *)calloc(NUM, sizeof(int));
  CHECK(hipMalloc((void **)&A_d, NUM * sizeof(int)));
  printf("info: copy Host2Device\n");
  CHECK(hipMemcpy(A_d, A_d, NUM * sizeof(int), hipMemcpyHostToDevice));

  hipStream_t q;
  uint32_t flags = hipStreamNonBlocking;
  CHECK(hipStreamCreateWithFlags(&q, flags));

  size_t sharedMem = 0;
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, 0,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, q,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, q,
                     A_d);
  hipLaunchKernelGGL(addOne, dim3(numBlocks), dim3(dimBlocks), sharedMem, 0,
                     A_d);
  hipMemcpy(A_h, A_d, NUM * sizeof(int), hipMemcpyDeviceToHost);

  bool pass = true;
  int num_errors = 0;
  for (int i = 0; i < NUM; i++) {
    if (A_h[i] != 4) {
      pass = false;
      num_errors++;
    }
  }

  std::cout << "Num Errors: " << num_errors << std::endl;
  std::cout << (pass ? "PASSED!" : "FAIL") << std::endl;
}
