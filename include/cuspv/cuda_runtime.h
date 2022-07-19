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

// Using same include guard name as in CUDA SDK's cuda_runtime.h so it
// gets excluded if it happens to be in the include search paths.
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__

#include <hip/hip_runtime.h>

// Needed for some CUDA samples.
#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__
#endif

// Pretend to be CUDA 8.0.
#ifndef __CUDART_VERSION__
#define __CUDART_VERSION__ 8000
#endif

// Pretend compute capability to be 2.0
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 200
#endif

#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorOutOfMemory hipErrorOutOfMemory
// Deprecated
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNotInitialized hipErrorNotInitialized
// Deprecated
#define cudaErrorInitializationError hipErrorInitializationError
#define cudaErrorDeinitialized hipErrorDeinitialized
#define cudaErrorProfilerDisabled hipErrorProfilerDisabled
#define cudaErrorProfilerNotInitialized hipErrorProfilerNotInitialized
#define cudaErrorProfilerAlreadyStarted hipErrorProfilerAlreadyStarted
#define cudaErrorProfilerAlreadyStopped hipErrorProfilerAlreadyStopped
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorInvalidPitchValue hipErrorInvalidPitchValue
#define cudaErrorInvalidSymbol hipErrorInvalidSymbol
#define cudaErrorInvalidDevicePointer hipErrorInvalidDevicePointer
#define cudaErrorInvalidMemcpyDirection hipErrorInvalidMemcpyDirection
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorMissingConfiguration hipErrorMissingConfiguration
#define cudaErrorPriorLaunchFailure hipErrorPriorLaunchFailure
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorInvalidImage hipErrorInvalidImage
#define cudaErrorInvalidContext hipErrorInvalidContext
#define cudaErrorContextAlreadyCurrent hipErrorContextAlreadyCurrent
#define cudaErrorMapFailed hipErrorMapFailed
// Deprecated
#define cudaErrorMapBufferObjectFailed hipErrorMapBufferObjectFailed
#define cudaErrorUnmapFailed hipErrorUnmapFailed
#define cudaErrorArrayIsMapped hipErrorArrayIsMapped
#define cudaErrorAlreadyMapped hipErrorAlreadyMapped
#define cudaErrorNoBinaryForGpu hipErrorNoBinaryForGpu
#define cudaErrorAlreadyAcquired hipErrorAlreadyAcquired
#define cudaErrorNotMapped hipErrorNotMapped
#define cudaErrorNotMappedAsArray hipErrorNotMappedAsArray
#define cudaErrorNotMappedAsPointer hipErrorNotMappedAsPointer
#define cudaErrorECCNotCorrectable hipErrorECCNotCorrectable
#define cudaErrorUnsupportedLimit hipErrorUnsupportedLimit
#define cudaErrorContextAlreadyInUse hipErrorContextAlreadyInUse
#define cudaErrorPeerAccessUnsupported hipErrorPeerAccessUnsupported
#define cudaErrorInvalidKernelFile hipErrorInvalidKernelFile
#define cudaErrorInvalidGraphicsContext hipErrorInvalidGraphicsContext
#define cudaErrorInvalidSource hipErrorInvalidSource
#define cudaErrorFileNotFound hipErrorFileNotFound
#define cudaErrorSharedObjectSymbolNotFound hipErrorSharedObjectSymbolNotFound
#define cudaErrorSharedObjectInitFailed hipErrorSharedObjectInitFailed
#define cudaErrorOperatingSystem hipErrorOperatingSystem
#define cudaErrorInvalidHandle hipErrorInvalidHandle
// Deprecated
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaErrorIllegalState hipErrorIllegalState
#define cudaErrorNotFound hipErrorNotFound
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorIllegalAddress hipErrorIllegalAddress
#define cudaErrorLaunchOutOfResources hipErrorLaunchOutOfResources
#define cudaErrorLaunchTimeOut hipErrorLaunchTimeOut
#define cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled =
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaErrorSetOnActiveProcess hipErrorSetOnActiveProcess
#define cudaErrorContextIsDestroyed hipErrorContextIsDestroyed
#define cudaErrorAssert hipErrorAssert
#define cudaErrorHostMemoryAlreadyRegistered = hipErrorHostMemoryAlreadyRegistered =
#define cudaErrorHostMemoryNotRegistered = hipErrorHostMemoryNotRegistered =
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorCooperativeLaunchTooLarge = hipErrorCooperativeLaunchTooLarge =
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaErrorStreamCaptureInvalidated hipErrorStreamCaptureInvalidated
#define cudaErrorStreamCaptureMerge hipErrorStreamCaptureMerge
#define cudaErrorStreamCaptureUnmatched hipErrorStreamCaptureUnmatched
#define cudaErrorStreamCaptureUnjoined hipErrorStreamCaptureUnjoined
#define cudaErrorStreamCaptureIsolation hipErrorStreamCaptureIsolation
#define cudaErrorStreamCaptureImplicit hipErrorStreamCaptureImplicit
#define cudaErrorCapturedEvent hipErrorCapturedEvent
#define cudaErrorStreamCaptureWrongThread hipErrorStreamCaptureWrongThread
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorRuntimeMemory hipErrorRuntimeMemory
#define cudaErrorRuntimeOther hipErrorRuntimeOther
#define cudaErrorTbd hipErrorTbd
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost

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

using cudaArray = hipArray;
using cudaArray_t = hipArray_t;
using cudaArray_const_t = hipArray_const_t;
using cudaChannelFormatDesc = hipChannelFormatDesc;
using cudaComputeMode = hipComputeMode;
using cudaDeviceAttribute_t = hipDeviceAttribute_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaDevice_t = hipDevice_t;
using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaExtent = hipExtent;
using cudaFuncAttributes = hipFuncAttributes;
using cudaFuncCache = hipFuncCache_t;
using cudaLimit = hipLimit_t;
using cudaMemoryAdvise = hipMemoryAdvise;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemRangeAttribute = hipMemRangeAttribute;
using cudaMipmappedArray_const_t = hipMipmappedArray_const_t;
using cudaPitchedPtr = hipPitchedPtr;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaResourceDesc = hipResourceDesc;
using cudaResourceViewDesc = hipResourceViewDesc;
using cudaTextureDesc = hipTextureDesc;
using cudaTextureObject_t = hipTextureObject_t;
using cudaSharedMemConfig = hipSharedMemConfig;
using cudaStream_t = hipStream_t;
using cudaStreamCallback_t = hipStreamCallback_t;
using cudaSurfaceObject_t = hipSurfaceObject_t;
#define CUDART_CB
//#define CUDA_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR

// Flags that can be used with hipStreamCreateWithFlags.
/** Default stream creation flags. These are used with hipStreamCreate().*/
#define cudaStreamDefault hipStreamDefault

/** Stream does not implicitly synchronize with null stream.*/
#define cudaStreamNonBlocking hipStreamNonBlocking

// Flags that can be used with hipEventCreateWithFlags.
/** Default flags.*/
#define cudaEventDefault hipEventDefault

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define cudaEventBlockingSync hipEventBlockingSync

/** Disable event's capability to record timing information. May improve performance.*/
#define cudaEventDisableTiming hipEventDisableTiming

/** Event can support IPC. Warnig: It is not supported in HIP.*/
#define cudaEventInterprocess hipEventInterprocess

/** Use a device-scope release when recording this event. This flag is useful to obtain more
 * precise timings of commands between events.  The flag is a no-op on CUDA platforms.*/
#define cudaEventReleaseToDevice hipEventReleaseToDevice

/** Use a system-scope release when recording this event. This flag is useful to make
 * non-coherent host memory visible to the host. The flag is a no-op on CUDA platforms.*/
#define cudaEventReleaseToSystem hipEventReleaseToSystem

// Flags that can be used with hipHostMalloc.
/** Default pinned memory allocation on the host.*/
#define cudaHostMallocDefault hipHostMallocDefault

/** Memory is considered allocated by all contexts.*/
#define cudaHostAllocPortable hipHostMallocPortable

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define cudaHostMallocMapped hipHostMallocMapped

/** Allocates the memory as write-combined. On some system configurations, write-combined allocation
 * may be transferred faster across the PCI Express bus, however, could have low read efficiency by
 * most CPUs. It's a good option for data tranfer from host to device via mapped pinned memory.*/
#define cudaHostMallocWriteCombined hipHostMallocWriteCombined

/** Host memory allocation will follow numa policy set by user.*/
#define cudaHostMallocNumaUser hipHostMallocNumaUser

/** Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define cudaHostMallocCoherent hipHostMallocCoherent

/** Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define cudaHostMallocNonCoherent hipHostMallocNonCoherent

/** Memory can be accessed by any stream on any device*/
#define cudaMemAttachGlobal hipMemAttachGlobal

/** Memory cannot be accessed by any stream on any device.*/
#define cudaMemAttachHost hipMemAttachHost

/** Memory can only be accessed by a single stream on the associated device.*/
#define cudaMemAttachSingle hipMemAttachSingle

#define cudaDeviceMallocDefault hipDeviceMallocDefault

/** Memory is allocated in fine grained region of device.*/
#define cudaDeviceMallocFinegrained hipDeviceMallocFinegrained

/** Memory represents a HSA signal.*/
#define cudaMallocSignalMemory hipMallocSignalMemory

// Flags that can be used with hipHostRegister.
/** Memory is Mapped and Portable.*/
#define cudaHostRegisterDefault hipHostRegisterDefault

/** Memory is considered registered by all contexts.*/
#define cudaHostRegisterPortable hipHostRegisterPortable

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define cudaHostRegisterMapped hipHostRegisterMapped

/** Not supported.*/
#define cudaHostRegisterIoMemory hipHostRegisterIoMemory

/** Coarse Grained host memory lock.*/
#define cudaExtHostRegisterCoarseGrained hipExtHostRegisterCoarseGrained

/** Automatically select between Spin and Yield.*/
#define cudaDeviceScheduleAuto hipDeviceScheduleAuto

/** Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
 * consume more power.*/
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin

/** Yield the CPU to the operating system when waiting. May increase latency, but lowers power
 * and is friendlier to other threads in the system.*/
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaDeviceBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceScheduleMask hipDeviceScheduleMask
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaDeviceLmemResizeToMax hipDeviceLmemResizeToMax
/** Default HIP array allocation flag.*/
#define cudaArrayDefault hipArrayDefault
#define cudaArrayLayered hipArrayLayered
#define cudaArraySurfaceLoadStore hipArraySurfaceLoadStore
#define cudaArrayCubemap hipArrayCubemap
#define cudaArrayTextureGather hipArrayTextureGather
#define cudaOccupancyDefault hipOccupancyDefault
#define cudaCooperativeLaunchMultiDeviceNoPreSync hipCooperativeLaunchMultiDeviceNoPreSync
#define cudaCooperativeLaunchMultiDeviceNoPostSync hipCooperativeLaunchMultiDeviceNoPostSync
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaInvalidDeviceId hipInvalidDeviceId
// Flags that can be used with hipExtLaunch Set of APIs.
/** AnyOrderLaunch of kernels.*/
#define cudaExtAnyOrderLaunch hipExtAnyOrderLaunch
// Flags to be used with hipStreamWaitValue32 and hipStreamWaitValue64.
#define cudaStreamWaitValueGte hipStreamWaitValueGte
#define cudaStreamWaitValueEq hipStreamWaitValueEq
#define cudaStreamWaitValueAnd hipStreamWaitValueAnd
#define cudaStreamWaitValueNor hipStreamWaitValueNor
// Stream per thread
/** Implicit stream per application thread.*/
#define cudaStreamPerThread hipStreamPerThread

// Textures
using cudaTextureReadMode = hipTextureReadMode;
#define cudaReadModeElementType  hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat

using cudaSurfaceBoundaryMode = hipSurfaceBoundaryMode;
#define cudaBoundaryModeZero hipBoundaryModeZero
#define cudaBoundaryModeTrap hipBoundaryModeTrap
#define cudaBoundaryModeClamp hipBoundaryModeClamp

using cudaTextureAddressMode = hipTextureAddressMode;
#define cudaAddressModeWrap  hipAddressModeWrap
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaAddressModeMirror  hipAddressModeMirror
#define cudaAddressModeBorder hipAddressModeBorder

using cudaTextureFilterMode = hipTextureFilterMode;
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaFilterModePoint hipFilterModePoint

using cudaChannelFormatKind = hipChannelFormatKind;
#define cudaChannelFormatKindSigned hipChannelFormatKindSigned
#define cudaChannelFormatKindUnsigned hipChannelFormatKindUnsigned
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaChannelFormatKindNone hipChannelFormatKindNone

//#################
static inline cudaError_t cudaGetDeviceCount(int *Count) { return hipGetDeviceCount(Count); }

static inline cudaError_t cudaSetDevice(int DeviceId) { return hipSetDevice(DeviceId); }
static inline cudaError_t cudaGetDevice(int *DeviceId) { return hipGetDevice(DeviceId); }
static inline cudaError_t cudaDeviceSynchronize(void) { return hipDeviceSynchronize(); }
static inline cudaError_t cudaDeviceReset(void) { return hipDeviceReset(); }
static inline cudaError_t cudaDeviceGet(cudaDevice_t *Device, int Ordinal) {
  return hipDeviceGet(Device, Ordinal);
}
static inline cudaError_t cudaDeviceComputeCapability(int *Major, int *Minor,
                                       cudaDevice_t Device) {
  return hipDeviceComputeCapability(Major, Minor, Device);
}
static inline cudaError_t cudaDeviceGetAttribute(int *RetPtr, cudaDeviceAttribute_t Attr,
                                  int DeviceId) {
  return hipDeviceGetAttribute(RetPtr, Attr, DeviceId);
}
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *Prop, int DeviceId) {
  return hipGetDeviceProperties(Prop, DeviceId);
}
static inline cudaError_t cudaDeviceGetLimit(size_t *PValue, cudaLimit Limit) {
  return hipDeviceGetLimit(PValue, Limit);
}
static inline cudaError_t cudaDeviceGetName(char *Name, int Len, cudaDevice_t Device) {
  return hipDeviceGetName(Name, Len, Device);
}
static inline cudaError_t cudaDeviceTotalMem(size_t *Bytes, cudaDevice_t Device) {
  return hipDeviceTotalMem(Bytes, Device);
}
static inline cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache CacheCfg) {
  return hipDeviceSetCacheConfig(CacheCfg);
}
static inline cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *CacheCfg) {
  return hipDeviceGetCacheConfig(CacheCfg);
}
static inline cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *Cfg) {
  return hipDeviceGetSharedMemConfig(Cfg);
}
static inline cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig Cfg) {
  return hipDeviceSetSharedMemConfig(Cfg);
}
static inline cudaError_t cudaFuncSetCacheConfig(const void *Func, cudaFuncCache Cfg) {
  return hipFuncSetCacheConfig(Func, Cfg);
}
static inline cudaError_t cudaDeviceGetPCIBusId(char *PciBusId, int Len, int DeviceId) {
  return hipDeviceGetPCIBusId(PciBusId, Len, DeviceId);
}
static inline cudaError_t cudaDeviceGetByPCIBusId(int *DeviceId, const char *PciBusId) {
  return hipDeviceGetByPCIBusId(DeviceId, PciBusId);
}
static inline cudaError_t cudaSetDeviceFlags(unsigned Flags) {
  return hipSetDeviceFlags(Flags);
}
static inline cudaError_t cudaDeviceCanAccessPeer(int *CanAccessPeer, int DeviceId,
                                   int PeerDeviceId) {
  return hipDeviceCanAccessPeer(CanAccessPeer, DeviceId, PeerDeviceId);
}
static inline cudaError_t cudaDeviceEnablePeerAccess(int PeerDeviceId, unsigned int Flags) {
  return hipDeviceEnablePeerAccess(PeerDeviceId, Flags);
}
static inline cudaError_t cudaDeviceDisablePeerAccess(int PeerDeviceId) {
  return hipDeviceDisablePeerAccess(PeerDeviceId);
}
static inline cudaError_t cudaChooseDevice(int *DeviceId, const cudaDeviceProp *Prop) {
  return hipChooseDevice(DeviceId, Prop);
}
static inline cudaError_t cudaGetDeviceFlags(unsigned int *Flags) {
  return hipGetDeviceFlags(Flags);
}
//####################
static inline cudaError_t cudaDriverGetVersion(int *DriverVersion) {
  return hipDriverGetVersion(DriverVersion);
}
static inline cudaError_t cudaRuntimeGetVersion(int *RuntimeVersion) {
  return hipRuntimeGetVersion(RuntimeVersion);
}
static inline cudaError_t cudaGetLastError(void) { return hipGetLastError(); }
static inline cudaError_t cudaPeekAtLastError(void) { return hipPeekAtLastError(); }
static inline const char *cudaGetErrorName(cudaError_t HipError) {
  return hipGetErrorName(HipError);
}
static inline const char *cudaGetErrorString(cudaError_t HipError) {
  return hipGetErrorString(HipError);
}

//##############
static inline cudaError_t cudaStreamCreate(cudaStream_t *Stream) {
  return hipStreamCreate(Stream);
}
static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *Stream, unsigned int Flags) {
  return hipStreamCreateWithFlags(Stream, Flags);
}
static inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t *Stream, unsigned int Flags,
                                        int Priority) {
  return hipStreamCreateWithPriority(Stream, Flags, Priority);
}
static inline cudaError_t cudaDeviceGetStreamPriorityRange(int *LeastPriority,
                                            int *GreatestPriority) {
  return hipDeviceGetStreamPriorityRange(LeastPriority, GreatestPriority);
}
static inline cudaError_t cudaStreamDestroy(cudaStream_t Stream) {
  return hipStreamDestroy(Stream);
}
static inline cudaError_t cudaStreamQuery(cudaStream_t Stream) {
  return hipStreamQuery(Stream);
}
static inline cudaError_t cudaStreamSynchronize(cudaStream_t Stream) {
  return hipStreamSynchronize(Stream);
}
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t Stream, cudaEvent_t Event,
                               unsigned int Flags) {
  return hipStreamWaitEvent(Stream, Event, Flags);
}
static inline cudaError_t cudaStreamGetFlags(cudaStream_t Stream, unsigned int *Flags) {
  return hipStreamGetFlags(Stream, Flags);
}
static inline cudaError_t cudaStreamGetPriority(cudaStream_t Stream, int *Priority) {
  return hipStreamGetPriority(Stream, Priority);
}
static inline cudaError_t cudaStreamAddCallback(cudaStream_t Stream,
                                 hipStreamCallback_t Callback, void *UserData,
                                 unsigned int Flags) {
  return hipStreamAddCallback(Stream, Callback, UserData, Flags);
}

//################

/*
using CUdeviceptr = something..;

static inline cudaError_t cuMemGetAddressRange(CUdeviceptr *Base, size_t *Size,
                                  CUdeviceptr Ptr) {
  return hipMemGetAddressRange(Base, Size, Ptr);
}
static inline cudaError_t cuDevicePrimaryCtxGetState(cudaDevice_t Device, unsigned int *Flags,
                                        int *Active) {
  return hipDevicePrimaryCtxGetState(Device, Flags, Active);
}
static inline cudaError_t cuDevicePrimaryCtxRelease(cudaDevice_t Device) {
  return hipDevicePrimaryCtxRelease(Device);
}
static inline cudaError_t cuDevicePrimaryCtxRetain(cudaContext_t *Context, cudaDevice_t Device) {
  return hipDevicePrimaryCtxRetain(Context, Device);
}
static inline cudaError_t cuDevicePrimaryCtxReset(cudaDevice_t Device) {
  return hipDevicePrimaryCtxReset(Device);
}
static inline cudaError_t cuDevicePrimaryCtxSetFlags(cudaDevice_t Device,
                                        unsigned int Flags) {
  return hipDevicePrimaryCtxSetFlags(Device, Flags);
}
*/

//#################
static inline cudaError_t cudaEventCreate(cudaEvent_t *Event) { return hipEventCreate(Event); }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *Event, unsigned Flags) {
  return hipEventCreateWithFlags(Event, Flags);
}
static inline cudaError_t cudaEventRecord(cudaEvent_t Event, cudaStream_t Stream) {
  return hipEventRecord(Event, Stream);
}

static inline cudaError_t cudaEventDestroy(cudaEvent_t Event) {
  return hipEventDestroy(Event);
}
static inline cudaError_t cudaEventSynchronize(cudaEvent_t Event) {
  return hipEventSynchronize(Event);
}
static inline cudaError_t cudaEventElapsedTime(float *Ms, cudaEvent_t Start, cudaEvent_t Stop) {
  return hipEventElapsedTime(Ms, Start, Stop);
}
static inline cudaError_t cudaEventQuery(cudaEvent_t Event) { return hipEventQuery(Event); }

//#################
static inline cudaError_t cudaMalloc(void **Ptr, size_t Size) { return hipMalloc(Ptr, Size); }
template <typename T>
static inline cudaError_t cudaMalloc(T **ptr, size_t size) {
  return hipMalloc((void **)ptr, size);
}
static inline cudaError_t cudaMallocManaged(void **DevPtr, size_t Size, unsigned int Flags) {
  return hipMallocManaged(DevPtr, Size, Flags);
}
static inline cudaError_t cudaMallocHost(void **Ptr, size_t Size) {
  return hipHostMalloc(Ptr, Size, 0);
}
template <class T>
static inline cudaError_t cudaMallocHost(T **ptr, size_t size) {
  return hipHostMalloc((void **)ptr, size);
}
static inline cudaError_t cudaHostMalloc(void **Ptr, size_t Size, unsigned int Flags) {
  return hipHostMalloc(Ptr, Size, Flags);
}
static inline cudaError_t cudaHostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  return hipHostMalloc(Ptr, Size, Flags);
}
template <class T>
static inline cudaError_t cudaHostAlloc(T **Ptr, size_t Size, unsigned int Flags) {
  return hipHostMalloc((void**)Ptr, Size, Flags);
}
static inline cudaError_t cudaFree(void *Ptr) { return hipFree(Ptr); }
static inline cudaError_t cudaHostFree(void *Ptr) { return hipHostFree(Ptr); }
static inline cudaError_t cudaFreeHost(void *Ptr) { return hipHostFree(Ptr); }
static inline cudaError_t cudaMemPrefetchAsync(const void *Ptr, size_t Count, int DstDevId,
                                cudaStream_t Stream) {
  return hipMemPrefetchAsync(Ptr, Count, DstDevId, Stream);
}

static inline cudaError_t cudaMemAdvise(const void *Ptr, size_t Count, cudaMemoryAdvise Advice,
                         int DstDevId) {
  return hipMemAdvise(Ptr, Count, Advice, DstDevId);
}
static inline cudaError_t cudaHostGetDevicePointer(void **DevPtr, void *HostPtr,
                                    unsigned int Flags) {
  return hipHostGetDevicePointer(DevPtr, HostPtr, Flags);
}
static inline cudaError_t cudaHostGetFlags(unsigned int *FlagsPtr, void *HostPtr) {
  return hipHostGetFlags(FlagsPtr, HostPtr);
}
static inline cudaError_t cudaHostRegister(void *HostPtr, size_t SizeBytes,
                            unsigned int Flags) {
  return hipHostRegister(HostPtr, SizeBytes, Flags);
}
static inline cudaError_t cudaHostUnregister(void *HostPtr) {
  return hipHostUnregister(HostPtr);
}
static inline cudaError_t cudaMallocPitch(void **Ptr, size_t *Pitch, size_t Width,
                           size_t Height) {
  return hipMallocPitch(Ptr, Pitch, Width, Height);
}
static inline cudaError_t cudaMalloc3DArray(cudaArray_t *Array,
                             const cudaChannelFormatDesc *Desc,
                             cudaExtent Extent, unsigned int Flags) {
  return hipMalloc3DArray(Array, Desc, Extent, Flags);
}

static inline cudaError_t cudaMallocArray(cudaArray_t *Array, const cudaChannelFormatDesc *Desc,
                           size_t Width, size_t Height = 0, unsigned int Flags = 0) {
  return hipMallocArray(Array, Desc, Width, Height, Flags);
}
static inline cudaError_t cudaFreeArray(cudaArray_t Array) { return hipFreeArray(Array); }

static inline cudaError_t cudaMalloc3D(cudaPitchedPtr *PitchedDevPtr, cudaExtent Extent) {
  return hipMalloc3D(PitchedDevPtr, Extent);
}
static inline cudaError_t cudaMemGetInfo(size_t *Free, size_t *Total) {
  return hipMemGetInfo(Free, Total);
}
static inline cudaError_t cudaMemPtrGetInfo(void *Ptr, size_t *Size) {
  return hipMemPtrGetInfo(Ptr, Size);
}
static inline cudaError_t cudaMemcpyAsync(void *Dst, const void *Src, size_t SizeBytes,
                           cudaMemcpyKind Kind, cudaStream_t Stream) {
  return hipMemcpyAsync(Dst, Src, SizeBytes, Kind, Stream);
}
static inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                    const void *ptr) {
  return hipPointerGetAttributes(attributes, ptr);
}

/*
static inline cudaError_t cuArrayCreate(cudaArray_t *Array,
                           const CUDA_ARRAY_DESCRIPTOR *AllocateArray) {
  return hipArrayCreate(Array, AllocateArray);
}
*/

//########################
static inline cudaError_t cudaMemcpyWithStream(void *Dst, const void *Src, size_t SizeBytes,
                                cudaMemcpyKind Kind, cudaStream_t Stream) {
  return hipMemcpyWithStream(Dst, Src, SizeBytes, Kind, Stream);
}
static inline cudaError_t cudaMemcpyPeer(void *Dst, int DstDeviceId, const void *Src,
                          int SrcDeviceId, size_t SizeBytes) {
  return hipMemcpyPeer(Dst, DstDeviceId, Src, SrcDeviceId, SizeBytes);
}
static inline cudaError_t cudaMemRangeGetAttribute(void *Data, size_t DataSize,
                                    cudaMemRangeAttribute Attribute,
                                    const void *DevPtr, size_t Count) {
  return hipMemRangeGetAttribute(Data, DataSize, Attribute, DevPtr, Count);
}
static inline cudaError_t cudaMemcpyPeerAsync(void *Dst, int DstDeviceId, const void *Src,
                               int SrcDevice, size_t SizeBytes,
                               cudaStream_t Stream) {
  return hipMemcpyPeerAsync(Dst, DstDeviceId, Src, SrcDevice, SizeBytes,
                            Stream);
}
static inline cudaError_t cudaMemcpyParam2DAsync(const hip_Memcpy2D *PCopy,
                                  cudaStream_t Stream) {
  return hipMemcpyParam2DAsync(PCopy, Stream);
}
static inline cudaError_t cudaMemcpyParam2D(const hip_Memcpy2D *PCopy) {
  return hipMemcpyParam2D(PCopy);
}

static inline cudaError_t cudaMemcpy(void *Dst, const void *Src, size_t SizeBytes,
                      cudaMemcpyKind Kind) {
  return hipMemcpy(Dst, Src, SizeBytes, Kind);
}

/*
static inline cudaError_t cuMemcpyDtoDAsync(CUdeviceptr Dst, CUdeviceptr Src,
                               size_t SizeBytes, cudaStream_t Stream) {
  return hipMemcpyDtoDAsync(Dst, Src, SizeBytes, Stream);
}
static inline cudaError_t cuMemcpyDtoD(CUdeviceptr Dst, CUdeviceptr Src,
                          size_t SizeBytes) {
  return hipMemcpyDtoD(Dst, Src, SizeBytes);
}
static inline cudaError_t cuMemcpyHtoDAsync(CUdeviceptr Dst, void *Src, size_t SizeBytes,
                               cudaStream_t Stream) {
  return hipMemcpyHtoDAsync(Dst, Src, SizeBytes, Stream);
}
static inline cudaError_t cuMemcpyHtoD(CUdeviceptr Dst, void *Src, size_t SizeBytes) {
  return hipMemcpyHtoD(Dst, Src, SizeBytes);
}
static inline cudaError_t cuMemcpyDtoHAsync(void *Dst, CUdeviceptr Src, size_t SizeBytes,
                               cudaStream_t Stream) {
  return hipMemcpyDtoHAsync(Dst, Src, SizeBytes, Stream);
}
static inline cudaError_t cuMemcpyDtoH(void *Dst, CUdeviceptr Src, size_t SizeBytes) {
  return hipMemcpyDtoH(Dst, Src, SizeBytes);
}
static inline cudaError_t cudaMemcpyAtoH(void *Dst, hipArray *SrcArray, size_t SrcOffset,
                          size_t Count) {
  return hipMemcpyAtoH(Dst, SrcArray, SrcOffset, Count);
}
static inline cudaError_t cudaMemcpyHtoA(hipArray *DstArray, size_t DstOffset,
                          const void *SrcHost, size_t Count) {
  return hipMemcpyHtoA(DstArray, DstOffset, SrcHost, Count);
}
*/

static inline cudaError_t cudaMemcpy2DAsync(void *Dst, size_t DPitch, const void *Src,
                             size_t SPitch, size_t Width, size_t Height,
                             cudaMemcpyKind Kind, cudaStream_t Stream) {
  return hipMemcpy2DAsync(Dst, DPitch, Src, SPitch, Width, Height, Kind,
                          Stream);
}
static inline cudaError_t cudaMemcpy2D(void *Dst, size_t DPitch, const void *Src,
                        size_t SPitch, size_t Width, size_t Height,
                        cudaMemcpyKind Kind) {
  return hipMemcpy2D(Dst, DPitch, Src, SPitch, Width, Height, Kind);
}
static inline cudaError_t cudaMemcpy2DToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                               const void *Src, size_t SPitch, size_t Width,
                               size_t Height, cudaMemcpyKind Kind) {
  return hipMemcpy2DToArray(Dst, WOffset, HOffset, Src, SPitch, Width, Height,
                            Kind);
}
static inline cudaError_t cudaMemcpy2DToArrayAsync(hipArray *Dst, size_t WOffset,
                                    size_t HOffset, const void *Src,
                                    size_t SPitch, size_t Width, size_t Height,
                                    cudaMemcpyKind Kind, cudaStream_t Stream) {
  return hipMemcpy2DToArrayAsync(Dst, WOffset, HOffset, Src, SPitch, Width,
                                 Height, Kind, Stream);
}
static inline cudaError_t cudaMemcpy2DFromArray(void *Dst, size_t DPitch, hipArray_const_t Src,
                                 size_t WOffset, size_t HOffset, size_t Width,
                                 size_t Height, cudaMemcpyKind Kind) {
  return hipMemcpy2DFromArray(Dst, DPitch, Src, WOffset, HOffset, Width, Height,
                              Kind);
}
static inline cudaError_t cudaMemcpy2DFromArrayAsync(void *Dst, size_t DPitch,
                                      hipArray_const_t Src, size_t WOffset,
                                      size_t HOffset, size_t Width,
                                      size_t Height, cudaMemcpyKind Kind,
                                      cudaStream_t Stream) {
  return hipMemcpy2DFromArrayAsync(Dst, DPitch, Src, WOffset, HOffset, Width,
                                   Height, Kind, Stream);
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static inline cudaError_t cudaMemcpyToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                             const void *Src, size_t Count,
                             cudaMemcpyKind Kind) {
  return hipMemcpyToArray(Dst, WOffset, HOffset, Src, Count, Kind);
}
static inline cudaError_t cudaMemcpyFromArray(void *Dst, hipArray_const_t SrcArray,
                               size_t WOffset, size_t HOffset, size_t Count,
                               cudaMemcpyKind Kind) {
  return hipMemcpyFromArray(Dst, SrcArray, WOffset, HOffset, Count, Kind);
}
#pragma GCC diagnostic pop
static inline cudaError_t cudaMemcpy3D(const struct hipMemcpy3DParms *Params) {
  return hipMemcpy3D(Params);
}
static inline cudaError_t cudaMemcpy3DAsync(const struct hipMemcpy3DParms *Params,
                             cudaStream_t Stream) {
  return hipMemcpy3DAsync(Params, Stream);
}
//###################

static inline cudaError_t cudaMemsetAsync(void *Dst, int Value, size_t SizeBytes,
                           cudaStream_t Stream) {
  return hipMemsetAsync(Dst, Value, SizeBytes, Stream);
}
static inline cudaError_t cudaMemset(void *Dst, int Value, size_t SizeBytes) {
  return hipMemset(Dst, Value, SizeBytes);
}

static inline cudaError_t cudaMemset2D(void *Dst, size_t Pitch, int Value, size_t Width,
                        size_t Height) {
  return hipMemset2D(Dst, Pitch, Value, Width, Height);
}

static inline cudaError_t cudaMemset2DAsync(void *Dst, size_t Pitch, int Value, size_t Width,
                            size_t Height, cudaStream_t Stream) {
  return hipMemset2DAsync(Dst, Pitch, Value, Width, Height, Stream);
}

static inline cudaError_t cudaMemset3DAsync(hipPitchedPtr PitchedDevPtr, int Value,
                             hipExtent Extent, cudaStream_t Stream) {
  return hipMemset3DAsync(PitchedDevPtr, Value, Extent, Stream);
}
static inline cudaError_t cudaMemset3D(hipPitchedPtr PitchedDevPtr, int Value,
                        hipExtent Extent) {
  return hipMemset3D(PitchedDevPtr, Value, Extent);
}

//###################

/*
static inline cudaError_t cuMemsetD8Async(hipDeviceptr_t Dest, unsigned char Value,
                             size_t Count, cudaStream_t Stream) {
  return hipMemsetD8Async(Dest, Value, Count, Stream);
}
static inline cudaError_t cuMemsetD8(hipDeviceptr_t Dest, unsigned char Value,
                        size_t SizeBytes) {
  return hipMemsetD8(Dest, Value, SizeBytes);
}
static inline cudaError_t cuMemsetD16Async(hipDeviceptr_t Dest, unsigned short Value,
                              size_t Count, cudaStream_t Stream) {
  return hipMemsetD16Async(Dest, Value, Count, Stream);
}
static inline cudaError_t cuMemsetD16(hipDeviceptr_t Dest, unsigned short Value,
                         size_t Count) {
  return hipMemsetD16(Dest, Value, Count);
}
static inline cudaError_t cuMemsetD32Async(hipDeviceptr_t Dst, int Value, size_t Count,
                              cudaStream_t Stream) {
  return hipMemsetD32Async(Dst, Value, Count, Stream);
}
static inline cudaError_t cuMemsetD32(hipDeviceptr_t Dst, int Value, size_t Count) {
  return hipMemsetD32(Dst, Value, Count);
}
*/

//##################
template <typename T>
static inline cudaError_t cudaGetSymbolSize(size_t *Size, const T& Symbol) {
  return hipGetSymbolSize(Size, (const void*)(&Symbol));
}
template <typename T>
static inline cudaError_t cudaGetSymbolAddress(void **DevPtr, const T& Symbol) {
  return hipGetSymbolAddress(DevPtr, (const void*)(&Symbol));
}
template <typename T>
static inline cudaError_t cudaMemcpyToSymbol(const T& Symbol, const void *Src,
                              size_t SizeBytes, size_t Offset = 0,
                              cudaMemcpyKind Kind = cudaMemcpyHostToDevice) {
  return hipMemcpyToSymbol((const void*)(&Symbol), Src, SizeBytes, Offset, Kind);
}
template <typename T>
static inline cudaError_t cudaMemcpyToSymbolAsync(const T& Symbol, const void *Src,
                                   size_t SizeBytes, size_t Offset,
                                   cudaMemcpyKind Kind, cudaStream_t Stream = 0) {
  return hipMemcpyToSymbolAsync((const void*)(&Symbol), Src, SizeBytes, Offset, Kind, Stream);
}
template <typename T>
static inline cudaError_t cudaMemcpyFromSymbol(void *Dst, const T& Symbol, size_t SizeBytes,
                                size_t Offset = 0, cudaMemcpyKind Kind = cudaMemcpyDeviceToHost) {
  return hipMemcpyFromSymbol(Dst, (const void*)(&Symbol), SizeBytes, Offset, Kind);
}
template <typename T>
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *Dst, const T& Symbol,
                                     size_t SizeBytes, size_t Offset,
                                     cudaMemcpyKind Kind, cudaStream_t Stream = 0) {
  return hipMemcpyFromSymbolAsync(Dst, (const void*)(&Symbol), SizeBytes, Offset, Kind, Stream);
}

//##################
template <typename T>
static inline cudaError_t cudaLaunchKernel(T HostFunction, dim3 GridDim,
                            dim3 BlockDim, void **Args, size_t SharedMem,
                            cudaStream_t Stream) {
  return hipLaunchKernel((void*)HostFunction, GridDim, BlockDim, Args, SharedMem,
                         Stream);
}
template <typename T>
static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *Attr,
                                 T *HostFunction) {
  return hipFuncGetAttributes(Attr, (void*)HostFunction);
}
template <typename T>
static inline cudaError_t cudaLaunchByPtr(T *HostFunction) {
  return hipLaunchByPtr((void*)HostFunction);
}

static inline cudaError_t cudaConfigureCall(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                             cudaStream_t Stream) {
  return hipConfigureCall(GridDim, BlockDim, SharedMem, Stream);
}
static inline cudaError_t cudaSetupArgument(const void *Arg, size_t Size, size_t Offset) {
  return hipSetupArgument(Arg, Size, Offset);
}



//#################
static inline cudaError_t cudaModuleGetGlobal(hipDeviceptr_t *Dptr, size_t *Bytes,
                               hipModule_t Hmod, const char *Name) {
  return hipModuleGetGlobal(Dptr, Bytes, Hmod, Name);
}
static inline cudaError_t cudaModuleLoadData(hipModule_t *Module, const void *Image) {
  return hipModuleLoadData(Module, Image);
}
static inline cudaError_t cudaModuleLoadDataEx(hipModule_t *Module, const void *Image,
                                unsigned int NumOptions, hipJitOption *Options,
                                void **OptionValues) {
  return hipModuleLoadDataEx(Module, Image, NumOptions, Options, OptionValues);
}
static inline cudaError_t cudaModuleLoad(hipModule_t *Module, const char *FuncName) {
  return hipModuleLoad(Module, FuncName);
}
static inline cudaError_t cudaModuleUnload(hipModule_t Module) {
  return hipModuleUnload(Module);
}
static inline cudaError_t cudaModuleGetFunction(hipFunction_t *Function, hipModule_t Module,
                                 const char *Name) {
  return hipModuleGetFunction(Function, Module, Name);
}

static inline cudaError_t
cudaModuleLaunchKernel(hipFunction_t Kernel, unsigned int GridDimX,
                       unsigned int GridDimY, unsigned int GridDimZ,
                       unsigned int BlockDimX, unsigned int BlockDimY,
                       unsigned int BlockDimZ, unsigned int SharedMemBytes,
                       cudaStream_t Stream, void **KernelParams, void **Extra) {
  return hipModuleLaunchKernel(Kernel, GridDimX, GridDimY, GridDimZ, BlockDimX,
                               BlockDimY, BlockDimZ, SharedMemBytes,
                               Stream, KernelParams, Extra);
}

//#################
template <typename T>
static inline cudaError_t cudaModuleOccupancyMaxPotentialBlockSize(int *GridSize,
                                                    int *BlockSize,
                                                    T Func,
                                                    size_t DynSharedMemPerBlk,
                                                    int BlockSizeLimit) {
  return hipModuleOccupancyMaxPotentialBlockSize(
      GridSize, BlockSize, reinterpret_cast<hipFunction_t>(Func),
        DynSharedMemPerBlk, BlockSizeLimit);
}

template <typename T>
static inline cudaError_t cudaModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int *GridSize, int *BlockSize, T Func,
    size_t DynSharedMemPerBlk, int BlockSizeLimit, unsigned int Flags) {
  return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
      GridSize, BlockSize, reinterpret_cast<hipFunction_t>(Func),
        DynSharedMemPerBlk, BlockSizeLimit, Flags);
}

template <typename T>
static inline cudaError_t cudaModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int *NumBlocks, T Func, int BlockSize,
    size_t DynSharedMemPerBlk) {
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
      NumBlocks, reinterpret_cast<hipFunction_t>(Func),
        BlockSize, DynSharedMemPerBlk);
}

template <typename T>
static inline cudaError_t cudaModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, T Func, int BlockSize,
    size_t DynSharedMemPerBlk, unsigned int Flags) {
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      NumBlocks, reinterpret_cast<hipFunction_t>(Func),
        BlockSize, DynSharedMemPerBlk, Flags);
}


template <typename T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *NumBlocks,
                                                                        T Func,
                                              int BlockSize,
                                              size_t DynSharedMemPerBlk) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(
      NumBlocks, reinterpret_cast<const void*>(Func),
        BlockSize, DynSharedMemPerBlk);
}

template <typename T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, T Func, int BlockSize, size_t DynSharedMemPerBlk,
    unsigned int Flags) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      NumBlocks, reinterpret_cast<const void*>(Func),
        BlockSize, DynSharedMemPerBlk, Flags);
}

template <typename T>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int *GridSize, int *BlockSize,
                                              T Func,
                                              size_t DynSharedMemPerBlk,
                                              int BlockSizeLimit) {
  return hipOccupancyMaxPotentialBlockSize(
      GridSize, BlockSize, reinterpret_cast<const void*>(Func),
      DynSharedMemPerBlk, BlockSizeLimit);
}


//###################
/* TODO is this (texture reference API) supported by HIP ? deprecated by CUDA */

/*
static inline cudaError_t cuTexRefSetAddressMode(textureReference* texRef, int dim,
                                                   cudaTextureAddressMode am);
static inline cudaError_t cuTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags);
static inline cudaError_t cuTexRefSetFilterMode(textureReference* texRef, cudaTextureFilterMode fm);
static inline cudaError_t cuTexRefSetFlags(textureReference* texRef, unsigned int Flags);
static inline cudaError_t cuTexRefSetFormat(textureReference* texRef, hipArray_Format fmt, int NumPackedComponents);
static inline cudaError_t cuTexObjectCreate(cudaTextureObject_t* pTexObject,
                                              const HIP_RESOURCE_DESC* pResDesc,
                                              const HIP_TEXTURE_DESC* pTexDesc,
                                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
static inline cudaError_t cuTexObjectDestroy(cudaTextureObject_t texObject);
static inline cudaError_t cuTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, cudaTextureObject_t texObject);
static inline cudaError_t cuTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc, cudaTextureObject_t texObject);
static inline cudaError_t cuTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, cudaTextureObject_t texObject);
*/


//###########################

static inline cudaError_t
cudaCreateTextureObject(cudaTextureObject_t *TexObject,
                        const cudaResourceDesc *ResDesc,
                        const cudaTextureDesc *TexDesc,
                        const cudaResourceViewDesc *ResViewDesc) {
  return hipCreateTextureObject(TexObject,
                                ResDesc,
                                TexDesc,
                                ResViewDesc);
}

static inline cudaError_t cudaDestroyTextureObject(cudaTextureObject_t TextureObject) {
  return hipDestroyTextureObject(TextureObject);
}

static inline cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) {
  return hipGetChannelDesc(desc, array);
}

static inline cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *ResDesc,
                                                           cudaTextureObject_t TextureObject) {
  return hipGetTextureObjectResourceDesc(ResDesc, TextureObject);
}

static inline cudaError_t cudaGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                                              cudaTextureObject_t textureObject) {
  return hipGetTextureObjectResourceViewDesc(pResViewDesc, textureObject);
}

static inline cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc,
                                                          cudaTextureObject_t textureObject) {
  return hipGetTextureObjectTextureDesc(pTexDesc, textureObject);
}

static inline cudaError_t cudaGetTextureReference(const textureReference** texref, const void* symbol) {
  return hipGetTextureReference(texref, symbol);
}

//################ TextureD Texture Management [Deprecated]

// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cudaBindTexture(size_t* offset, const textureReference* tex, const void* devPtr,
                                          const cudaChannelFormatDesc* desc, size_t size) {
  return hipBindTexture(offset, tex, devPtr, desc, size);
}

//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t cudaBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                          const void* devPtr, size_t size = UINT_MAX) {
  return cudaBindTexture(offset, &tex, devPtr, &tex.channelDesc, size);
}

//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t
    cudaBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex, const void* devPtr,
                   const cudaChannelFormatDesc& desc, size_t size = UINT_MAX) {
  return cudaBindTexture(offset, &tex, devPtr, &desc, size);
}

// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cudaBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                                            const cudaChannelFormatDesc* desc, size_t width, size_t height,
                                            size_t pitch) {
  return hipBindTexture2D(offset, tex, devPtr, desc, width, height, pitch);
}

//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t
    cudaBindTexture2D(size_t* offset, const struct texture<T, dim, readMode>& tex,
                     const void* devPtr, size_t width, size_t height, size_t pitch) {
  return cudaBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t
    cudaBindTexture2D(size_t* offset, const struct texture<T, dim, readMode>& tex,
                     const void* devPtr, const cudaChannelFormatDesc& desc, size_t width,
                     size_t height, size_t pitch) {
  return cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}

// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cudaBindTextureToArray(const textureReference* tex, cudaArray_const_t array,
                                                 const cudaChannelFormatDesc* desc) {
  return hipBindTextureToArray(tex, array, desc);
}

//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t
    cudaBindTextureToArray(const struct texture<T, dim, readMode>& tex, cudaArray_const_t array) {
  cudaChannelFormatDesc desc;
  cudaError_t err = cudaGetChannelDesc(&desc, array);
  return (err == cudaSuccess) ? cudaBindTextureToArray(&tex, array, &desc) : err;
}
//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t
    cudaBindTextureToArray(const struct texture<T, dim, readMode>& tex, cudaArray_const_t array,
                          const cudaChannelFormatDesc& desc) {
  return cudaBindTextureToArray(&tex, array, &desc);
}

static inline cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray,
                                                     cudaMipmappedArray_const_t mipmappedArray,
                                                     unsigned int level) {
  return hipGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
}

//DEPRECATED(DEPRECATED_MSG)
static inline cudaError_t cudaBindTextureToMipmappedArray(const textureReference* tex,
                                                          cudaMipmappedArray_const_t mipmappedArray,
                                                          const cudaChannelFormatDesc* desc) {
  return hipBindTextureToMipmappedArray(tex, mipmappedArray, desc);
}
//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t cudaBindTextureToMipmappedArray(const struct texture<T, dim, readMode>& tex,
                                                        cudaMipmappedArray_const_t mipmappedArray) {
  cudaChannelFormatDesc desc;
  cudaArray_t levelArray;
  cudaError_t err = cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaGetChannelDesc(&desc, levelArray);
  return (err == cudaSuccess) ? cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc) : err;
}
//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t cudaBindTextureToMipmappedArray(const struct texture<T, dim, readMode>& tex,
                                                        cudaMipmappedArray_const_t mipmappedArray,
                                                        const cudaChannelFormatDesc& desc) {
  return cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc);
}

/* TODO where is hipCreateChannelDesc defined ? */
template<typename ...Args>
static inline cudaChannelFormatDesc cudaCreateChannelDesc(Args && ...A) {
  return hipCreateChannelDesc(std::forward<Args>(A)...);
}


// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cudaGetTextureAlignmentOffset(size_t* offset, const textureReference* texref) {
  return hipGetTextureAlignmentOffset(offset, texref);
}
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cudaUnbindTexture(const textureReference* tex) {
  return hipUnbindTexture(tex);
}
//DEPRECATED(DEPRECATED_MSG)
template <class T, int dim, cudaTextureReadMode readMode>
static inline cudaError_t cudaUnbindTexture(const struct texture<T, dim, readMode>& tex) {
  return cudaUnbindTexture(&tex);
}

/* TODO is this (texture reference API) supported by HIP ? */
/*
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetAddressMode(enum cudaTextureAddressMode* pam, const textureReference* texRef, int dim);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetFilterMode(enum cudaTextureFilterMode* pfm, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetMipmapFilterMode(enum cudaTextureFilterMode* pfm,
                                        const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefGetMipMappedArray(hipMipmappedArray_t* pArray, const textureReference* texRef);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch);
// DEPRECATED(// DEPRECATED_MSG)
static inline cudaError_t cuTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso);
*/


//#########################
// HIP: The following are not supported.
/*
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor);
hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, cudaTextureFilterMode fm);
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias);
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp);
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray, unsigned int Flags);
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels);
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level);
*/

//###################

static inline cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) {
  return hipCreateSurfaceObject(pSurfObject, pResDesc);
}

static inline cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfaceObject) {
  return hipDestroySurfaceObject(surfaceObject);
}

/* surface reference API (cudaBindSurfaceToArray) and surface reference type (template) are not supported by HIP. */
#endif
