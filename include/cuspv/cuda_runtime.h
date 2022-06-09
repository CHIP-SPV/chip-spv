/*
Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/**
 * A header for replacing cuda_runtime.h. Purpose of the header is to
 * traslate CUDA API functions to corresponding ones in HIP.
 */

#include <hip/hip_runtime.h>

// CUDA runtime header include guard. Some CUDA programs may use this
// to detect CUDA compilation.
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif

// Needed for some CUDA samples.
#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__
#endif

#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotReady hipErrorNotReady
// TODO: other Cuda error codes

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
// TODO: other memcpy flags.

#define cudaDevAttrComputeMode hipDeviceAttributeComputeMode

#undef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor                                      \
  hipDeviceAttributeComputeCapabilityMajor

#undef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor                                      \
  hipDeviceAttributeComputeCapabilityMinor

#undef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

#undef cudaComputeModeProhibited
#define cudaComputeModeProhibited hipComputeModeProhibited

#undef cudaDevAttrClockRate
#define cudaDevAttrClockRate hipDeviceAttributeClockRate

#undef cudaDevAttrIntegrated
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated

#undef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking

using cudaError_t = hipError_t;
using cudaDeviceAttribute_t = hipDeviceAttribute_t;
using cudaComputeMode = hipComputeMode;
using cudaMemcpyKind = hipMemcpyKind;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

static inline const char *cudaGetErrorString(cudaError_t cudaError) {
  return hipGetErrorString(cudaError);
}

static inline const char *cudaGetErrorName(cudaError_t error) {
  return hipGetErrorName(error);
}

static inline cudaError_t cudaGetDeviceCount(int *count) {
  return hipGetDeviceCount(count);
}

static inline cudaError_t cudaGetDevice(int *deviceId) {
  return hipGetDevice(deviceId);
}

static inline cudaError_t cudaSetDevice(int deviceId) {
  return hipSetDevice(deviceId);
}

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop,
                                                  int deviceId) {
  return hipGetDeviceProperties(prop, deviceId);
}

static inline cudaError_t
cudaDeviceGetAttribute(int *pi, cudaDeviceAttribute_t attr, int deviceId) {
  return hipDeviceGetAttribute(pi, attr, deviceId);
}

static inline cudaError_t cudaMalloc(void **ptr, size_t size) {
  return hipMalloc(ptr, size);
}

template <class T> static inline cudaError_t cudaMalloc(T **ptr, size_t size) {
  return hipMalloc((void **)ptr, size);
}

static inline cudaError_t cudaMallocHost(void **ptr, size_t size) {
  return hipHostMalloc(ptr, size);
}

template <class T>
static inline cudaError_t cudaMallocHost(T **ptr, size_t size) {
  return hipHostMalloc((void **)ptr, size);
}

static inline cudaError_t cudaFree(void *ptr) { return hipFree(ptr); }

static inline cudaError_t cudaFreeHost(void *ptr) { return hipHostFree(ptr); }

static inline cudaError_t cudaMemset(void *dst, int value, size_t sizeBytes) {
  return hipMemset(dst, value, sizeBytes);
}

static inline cudaError_t cudaMemcpy(void *dst, const void *src,
                                     size_t sizeBytes, cudaMemcpyKind kind) {
  return hipMemcpy(dst, src, sizeBytes, kind);
}

static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                          size_t sizeBytes, cudaMemcpyKind kind,
                                          cudaStream_t stream __dparm(0)) {
  return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
  return hipStreamCreateWithFlags(stream, flags);
}

static inline cudaError_t cudaDeviceSynchronize(void) {
  return hipDeviceSynchronize();
}

static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return hipStreamSynchronize(stream);
}

static inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return hipEventCreate(event);
}

static inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return hipEventDestroy(event);
}

static inline cudaError_t cudaEventRecord(cudaEvent_t event,
                                          cudaStream_t stream = NULL) {
  return hipEventRecord(event, stream);
}

static inline cudaError_t cudaEventQuery(cudaEvent_t event) {
  return hipEventQuery(event);
}

static inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return hipEventSynchronize(event);
}

static inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                               cudaEvent_t stop) {
  return hipEventElapsedTime(ms, start, stop);
}

static inline cudaError_t cudaGetLastError() { return hipGetLastError(); }
