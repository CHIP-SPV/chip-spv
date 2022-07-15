#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__device__ int a;

template<class T>
__device__ T overloaded() { return 3; }

template<class T>
__host__ T overloaded() { return 0xdeadbeef; }

template<class F>
__global__ void g(int *Out, F f) { *Out = f(); }

int main() {
  int OutH = 0, *OutD;
  (void)cudaMalloc(&OutD, sizeof(int));

  auto L0 = []() -> int { return 1; };
  g<<<1, 1>>>(OutD, L0);
  (void)cudaMemcpy(&OutH, OutD, sizeof(int), cudaMemcpyDeviceToHost);
  assert(OutH == 1);

  int InH = 2, *InD;
  (void)cudaMalloc(&InD, sizeof(int));
  (void)cudaMemcpy(InD, &InH, sizeof(int), cudaMemcpyHostToDevice);
  auto L1 = [=]() -> int { return *InD; };
  g<<<1, 1>>>(OutD, L1);
  (void)cudaMemcpy(&OutH, OutD, sizeof(int), cudaMemcpyDeviceToHost);
  assert(OutH == 2);

  g<<<1, 1>>>(OutD, []() { return overloaded<int>(); });
  (void)cudaMemcpy(&OutH, OutD, sizeof(int), cudaMemcpyDeviceToHost);
  assert(OutH == 3);

  printf("PASSED\n");
  return 0;
}
