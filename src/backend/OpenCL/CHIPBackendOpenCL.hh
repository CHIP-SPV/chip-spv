/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for CHIP. CHIPBackendOpenCL class definition with
 * inheritance from CHIPBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BACKEND_OPENCL_H
#define CHIP_BACKEND_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <CL/cl_ext_intel.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#include <CL/opencl.hpp>
#pragma GCC diagnostic pop

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"

std::string resultToString(int Status);

class CHIPContextOpenCL;
class CHIPDeviceOpenCL;
class CHIPExecItemOpenCL;
class CHIPKernelOpenCL;
class CHIPQueueOpenCL;
class CHIPEventOpenCL;
class CHIPBackendOpenCL;
class CHIPModuleOpenCL;
class CHIPTextureOpenCL;

class CHIPCallbackDataOpenCL {
private:
public:
  CHIPQueueOpenCL *ChipQueue;
  void *CallbackArgs;
  hipStreamCallback_t CallbackF;
  hipError_t Status;

  CHIPCallbackDataOpenCL(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         CHIPQueue *ChipQueue);
};

class CHIPEventMonitorOpenCL : public CHIPEventMonitor {
public:
  CHIPEventMonitorOpenCL();
  virtual void monitor() override;
};

class CHIPEventOpenCL : public CHIPEvent {
public:
  cl_event ClEvent;
  friend class CHIPEventOpenCL;

public:
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext, cl_event ClEvent,
                  CHIPEventFlags Flags = CHIPEventFlags());
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                  CHIPEventFlags Flags = CHIPEventFlags());
  virtual ~CHIPEventOpenCL() override;
  virtual void recordStream(CHIPQueue *ChipQueue) override;
  void takeOver(CHIPEvent *Other);
  virtual void decreaseRefCount(std::string Reason) override;
  virtual void increaseRefCount(std::string Reason) override;
  bool wait() override;
  float getElapsedTime(CHIPEvent *Other) override;
  virtual void hostSignal() override;
  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;
  cl_event peek();
  cl_event get();
  uint64_t getFinishTime();
  size_t getRefCount();
};

class CHIPModuleOpenCL : public CHIPModule {
protected:
  cl::Program Program_;

public:
  CHIPModuleOpenCL(std::string *ModuleStr);
  virtual ~CHIPModuleOpenCL() {}
  virtual void compile(CHIPDevice *ChipDevice) override;
  cl::Program *get();
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations_;
  cl::Context Context_;

public:
  void init(cl::Context &C) { Context_ = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&Rhs);
  void *allocate(size_t Size);
  bool free(void *P);
  bool hasPointer(const void *Ptr);
  bool pointerSize(void *Ptr, size_t *Size);
  bool pointerInfo(void *Ptr, void **Base, size_t *Size);
  int memCopy(void *Dst, const void *Src, size_t Size, cl::CommandQueue &Queue);
  int memFill(void *Dst, size_t Size, const void *Pattern, size_t PatternSize,
              cl::CommandQueue &Queue);
  void clear();
};

class CHIPContextOpenCL : public CHIPContext {
public:
  SVMemoryRegion SvmMemory;
  cl::Context *ClContext;
  CHIPContextOpenCL(cl::Context *ClContext);
  virtual ~CHIPContextOpenCL() {}
  void *allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemType,
                     CHIPHostAllocFlags Flags = CHIPHostAllocFlags()) override;

  bool isAllocatedPtrUSM(void* Ptr) override { return true; }
  virtual void freeImpl(void *Ptr) override;
  cl::Context *get();
};

class CHIPDeviceOpenCL : public CHIPDevice {
public:
  cl::Device *ClDevice;
  cl::Context *ClContext;
  CHIPDeviceOpenCL(CHIPContextOpenCL *ChipContext, cl::Device *ClDevice,
                   int Idx);
  cl::Device *get();
  virtual void populateDevicePropertiesImpl() override;
  virtual void resetImpl() override;
  virtual CHIPModuleOpenCL *addModule(std::string *ModuleStr) override;
  virtual CHIPQueue *addQueueImpl(unsigned int Flags, int Priority) override;
  virtual CHIPQueue *addQueueImpl(const uintptr_t *NativeHandles, int NumHandles) override;

  virtual CHIPTexture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override;
  virtual void destroyTexture(CHIPTexture *ChipTexture) override {
    logTrace("CHIPDeviceOpenCL::destroyTexture");
    delete ChipTexture;
  }
};

class CHIPQueueOpenCL : public CHIPQueue {
protected:
  // Any reason to make these private/protected?
  cl::CommandQueue *ClQueue_;

public:
  CHIPQueueOpenCL() = delete; // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(CHIPDevice *ChipDevice, cl_command_queue Queue = nullptr);
  ~CHIPQueueOpenCL();
  virtual CHIPEventOpenCL *getLastEvent() override;
  virtual CHIPEvent *launchImpl(CHIPExecItem *ExecItem) override;
  virtual void addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;
  virtual void finish() override;
  virtual CHIPEvent *memCopyAsyncImpl(void *Dst, const void *Src,
                                      size_t Size) override;
  cl::CommandQueue *get();
  virtual CHIPEvent *memFillAsyncImpl(void *Dst, size_t Size,
                                      const void *Pattern,
                                      size_t PatternSize) override;
  virtual CHIPEvent *memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                        const void *Src, size_t Spitch,
                                        size_t Width, size_t Height) override;
  virtual CHIPEvent *memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                        size_t Dspitch, const void *Src,
                                        size_t Spitch, size_t Sspitch,
                                        size_t Width, size_t Height,
                                        size_t Depth) override;

  virtual hipError_t getBackendHandles(uintptr_t *NativeInfo, int *NumHandles) override;
  virtual CHIPEvent *
  enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) override;
  virtual CHIPEvent *enqueueMarkerImpl() override;
  virtual CHIPEvent *memPrefetchImpl(const void *Ptr, size_t Count) override;
};

class CHIPKernelOpenCL : public CHIPKernel {
private:
  std::string Name_;
  size_t TotalArgSize_;
  cl::Kernel OclKernel_;
  size_t MaxDynamicLocalSize_;
  size_t MaxWorkGroupSize_;
  size_t StaticLocalSize_;
  size_t PrivateSize_;

  CHIPModuleOpenCL *Module;
  CHIPDeviceOpenCL *Device;

public:
  CHIPKernelOpenCL(const cl::Kernel &&ClKernel, CHIPDeviceOpenCL *Dev,
                   std::string HostFName, OCLFuncInfo *FuncInfo,
                   CHIPModuleOpenCL *Parent);
  virtual ~CHIPKernelOpenCL() {}
  OCLFuncInfo *getFuncInfo() const;
  std::string getName();
  cl::Kernel *get();
  size_t getTotalArgSize() const;

  CHIPModuleOpenCL *getModule() override { return Module; }
  const CHIPModuleOpenCL *getModule() const override { return Module; }
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;
};

class CHIPExecItemOpenCL : public CHIPExecItem {
private:
  cl::Kernel *ClKernel_;

public:
  OCLFuncInfo FuncInfo;
  int setupAllArgs(CHIPKernelOpenCL *Kernel);
  cl::Kernel *get();
};

class CHIPBackendOpenCL : public CHIPBackend {
public:
  virtual void initializeImpl(std::string CHIPPlatformStr,
                              std::string CHIPDeviceTypeStr,
                              std::string CHIPDeviceStr) override;
  virtual void initializeFromNative(const uintptr_t *NativeHandles, int NumHandles) override;

  virtual std::string getDefaultJitFlags() override;

  virtual int ReqNumHandles() override { return 4; }

  virtual CHIPQueue *createCHIPQueue(CHIPDevice *ChipDev) override;
  virtual CHIPEventOpenCL *
  createCHIPEvent(CHIPContext *ChipCtx, CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false) override;
  virtual CHIPCallbackData *createCallbackData(hipStreamCallback_t Callback,
                                               void *UserData,
                                               CHIPQueue *ChipQueue) override;
  virtual CHIPEventMonitor *createCallbackEventMonitor() override;
  virtual CHIPEventMonitor *createStaleEventMonitor() override;

  virtual hipEvent_t getHipEvent(void* NativeEvent) override;
  virtual void* getNativeEvent(hipEvent_t HipEvent) override;

};

class CHIPTextureOpenCL : public CHIPTexture {
  cl_mem Image;
  cl_sampler Sampler;

public:
  CHIPTextureOpenCL() = delete;
  CHIPTextureOpenCL(const hipResourceDesc &ResDesc, cl_mem TheImage,
                    cl_sampler TheSampler)
      : CHIPTexture(ResDesc), Image(TheImage), Sampler(TheSampler) {}

  virtual ~CHIPTextureOpenCL() {
    cl_int Status;
    Status = clReleaseMemObject(Image);
    assert(Status == CL_SUCCESS && "Invalid image handler?");
    Status = clReleaseSampler(Sampler);
    assert(Status == CL_SUCCESS && "Invalid sampler handler?");
    (void)Status;
  }

  cl_mem getImage() const { return Image; }
  cl_sampler getSampler() const { return Sampler; }
};

#endif
