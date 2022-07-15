// Test an unsupported device function.

#include <cuda_runtime.h>
__global__ void test(unsigned *Mask) {
  *Mask = __activemask();
}

int main() {
  unsigned *MaskD;
  cudaMalloc(&MaskD, sizeof(unsigned));
  test<<<1, 1>>>(MaskD);
  return 0;
}
