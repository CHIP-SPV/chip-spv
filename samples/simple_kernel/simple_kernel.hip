#include "hip/hip_runtime.h"
#include <cstdio>

__global__ void kernel(int *Out, int *In) { *Out = *In; }

int main() {
  int InH = 123, OutH = 0, *InD, *OutD;
  (void)hipMalloc(&OutD, sizeof(int));
  (void)hipMalloc(&InD, sizeof(int));
  (void)hipMemcpy(InD, &InH, sizeof(int), hipMemcpyHostToDevice);
  kernel<<<1, 1>>>(OutD, InD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  printf("OutH=%d\n", OutH);
  if (OutH == 123) {
    printf("PASSED\n");
    return 0;
  }
  return 1;
}
