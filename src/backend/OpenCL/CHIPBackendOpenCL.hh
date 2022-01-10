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

#include <CL/opencl.hpp>

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"

std::string resultToString(int status);

class CHIPContextOpenCL;
class CHIPDeviceOpenCL;
class CHIPExecItemOpenCL;
class CHIPKernelOpenCL;
class CHIPQueueOpenCL;
class CHIPEventOpenCL;
class CHIPBackendOpenCL;
class CHIPModuleOpenCL;

class CHIPCallbackDataOpenCL : public CHIPCallbackData {
 private:
 public:
  CHIPCallbackDataOpenCL(hipStreamCallback_t callback_f_, void *callback_args_,
                         CHIPQueue *chip_queue_);
  virtual void setup() override;
};

class CHIPEventMonitorOpenCL : public CHIPEventMonitor {
 public:
  CHIPEventMonitorOpenCL();
  // virtual void monitor() override;
};

class CHIPEventOpenCL : public CHIPEvent {
 public:
  cl_event ev;

 public:
  CHIPEventOpenCL(CHIPContextOpenCL *chip_ctx_, cl_event ev_,
                  CHIPEventType event_type_ = CHIPEventType::Default)
      : CHIPEvent((CHIPContext *)(chip_ctx_), event_type_), ev(ev_) {
    event_status = EVENT_STATUS_RECORDING;
    clRetainEvent(ev);
  }

  CHIPEventOpenCL(CHIPContextOpenCL *chip_ctx_,
                  CHIPEventType event_type_ = CHIPEventType::Default)
      : CHIPEvent((CHIPContext *)(chip_ctx_), event_type_), ev(nullptr) {}

  virtual ~CHIPEventOpenCL() override;
  virtual void deinit() override;
  virtual void decreaseRefCount() override;
  virtual void increaseRefCount() override;
  void recordStream(CHIPQueue *chip_queue_) override;
  bool wait() override;
  float getElapsedTime(CHIPEvent *other) override;

  virtual void barrier(CHIPQueue *chip_queue_) override;

  virtual void hostSignal() override;

  virtual bool updateFinishStatus() override;

  cl_event peek() { return ev; }
  cl_event get() {
    increaseRefCount();
    return ev;
  }

  uint64_t getFinishTime() {
    std::lock_guard<std::mutex> Lock(mtx);
    int status;
    uint64_t ret;
    status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(ret),
                                     &ret, NULL);

    CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd,
                                "Failed to query event for profiling info.");
    return ret;
  }

  size_t *getRefCount() {
    cl_uint refcount;
    int status = ::clGetEventInfo(this->peek(), CL_EVENT_REFERENCE_COUNT, 4,
                                  &refcount, NULL);
    CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd);
    // logDebug("CHIPEventOpenCL::getRefCount() CHIP refc: {} OCL refc: {}",
    // refc,
    //         refcount);
    return refc;
  }
};

class CHIPModuleOpenCL : public CHIPModule {
 protected:
  cl::Program program;

 public:
  CHIPModuleOpenCL(std::string *module_str) : CHIPModule(module_str){};
  virtual void compile(CHIPDevice *chip_dev) override;
  cl::Program &get() { return program; }

  virtual bool registerVar(const char *var_name_) override {
    UNIMPLEMENTED(false);
  }
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations;
  cl::Context Context;

 public:
  void init(cl::Context &C) { Context = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&rhs) {
    SvmAllocations = std::move(rhs.SvmAllocations);
    Context = std::move(rhs.Context);
    return *this;
  }

  void *allocate(cl::Context ctx, size_t size);
  bool free(void *p, size_t *size);
  bool hasPointer(const void *p);
  bool pointerSize(void *ptr, size_t *size);
  bool pointerInfo(void *ptr, void **pbase, size_t *psize);
  int memCopy(void *dst, const void *src, size_t size, cl::CommandQueue &queue);
  int memFill(void *dst, size_t size, const void *pattern, size_t patt_size,
              cl::CommandQueue &queue);
  void clear();
};

class CHIPContextOpenCL : public CHIPContext {
 public:
  SVMemoryRegion svm_memory;
  cl::Context *cl_ctx;
  CHIPContextOpenCL(cl::Context *ctx_in);

  void *allocate_(size_t size, size_t alignment,
                  CHIPMemoryType mem_type) override;

  void free_(void *ptr) override{};
  virtual hipError_t memCopy(void *dst, const void *src, size_t size,
                             hipStream_t stream) override;
  cl::Context *get() { return cl_ctx; }
  virtual CHIPEvent *createEvent(unsigned flags) override;
};

class CHIPDeviceOpenCL : public CHIPDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  CHIPDeviceOpenCL(CHIPContextOpenCL *chip_ctx, cl::Device *dev_in, int idx);

  cl::Device *get() { return cl_dev; }

  virtual void populateDeviceProperties_() override;

  virtual void reset() override;

  virtual CHIPModuleOpenCL *addModule(std::string *module_str) override {
    CHIPModuleOpenCL *mod = new CHIPModuleOpenCL(module_str);
    chip_modules.push_back(mod);
    return mod;
  }

  virtual CHIPQueue *addQueue_(unsigned int flags, int priority) override;
  virtual CHIPTexture *createTexture(
      const hipResourceDesc *pResDesc, const hipTextureDesc *pTexDesc,
      const struct hipResourceViewDesc *pResViewDesc) override {
    UNIMPLEMENTED(nullptr);
  }

  virtual void destroyTexture(CHIPTexture *textureObject) override {
    UNIMPLEMENTED();
  };
};

class CHIPQueueOpenCL : public CHIPQueue {
 protected:
  // Any reason to make these private/protected?
  cl::Context *cl_ctx;
  cl::Device *cl_dev;
  cl::CommandQueue *cl_q;

 public:
  CHIPQueueOpenCL() = delete;  // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(CHIPDevice *chip_device);
  ~CHIPQueueOpenCL();

  virtual void updateLastEvent(CHIPEvent *ev) override;

  virtual CHIPEventOpenCL *getLastEvent() override;

  virtual CHIPEvent *launchImpl(CHIPExecItem *exec_item) override;
  virtual void finish() override;

  virtual CHIPEvent *memCopyAsyncImpl(void *dst, const void *src,
                                      size_t size) override;
  cl::CommandQueue *get() { return cl_q; }

  virtual CHIPEvent *memFillAsyncImpl(void *dst, size_t size,
                                      const void *pattern,
                                      size_t pattern_size) override;

  virtual CHIPEvent *memCopy2DAsyncImpl(void *dst, size_t dpitch,
                                        const void *src, size_t spitch,
                                        size_t width, size_t height) override;

  virtual CHIPEvent *memCopy3DAsyncImpl(void *dst, size_t dpitch,
                                        size_t dspitch, const void *src,
                                        size_t spitch, size_t sspitch,
                                        size_t width, size_t height,
                                        size_t depth) override;

  // Memory copy to texture object, i.e. image
  virtual CHIPEvent *memCopyToTextureImpl(CHIPTexture *texObj,
                                          void *src) override;

  virtual void getBackendHandles(unsigned long *nativeInfo,
                                 int *size) override {}  // TODO

  virtual CHIPEvent *enqueueBarrierImpl(
      std::vector<CHIPEvent *> *eventsToWaitFor) override {
    UNIMPLEMENTED(nullptr);
  }

  virtual CHIPEvent *enqueueMarkerImpl() override;
  virtual CHIPEvent *memPrefetchImpl(const void *ptr, size_t count) override {
    UNIMPLEMENTED(nullptr);
  }
};

class CHIPKernelOpenCL : public CHIPKernel {
 private:
  std::string name;
  size_t TotalArgSize;
  cl::Kernel ocl_kernel;

 public:
  CHIPKernelOpenCL(const cl::Kernel &&cl_kernel, std::string host_f_name_,
                   OCLFuncInfo *func_info_);

  OCLFuncInfo *get_func_info() const { return func_info; }
  std::string get_name() { return name; }
  cl::Kernel get() const { return ocl_kernel; }
  size_t getTotalArgSize() const { return TotalArgSize; };
};

class CHIPExecItemOpenCL : public CHIPExecItem {
 private:
  cl::Kernel *cl_kernel;

 public:
  OCLFuncInfo FuncInfo;
  int setupAllArgs(CHIPKernelOpenCL *kernel);
  cl::Kernel *get() { return cl_kernel; }
};

class CHIPBackendOpenCL : public CHIPBackend {
 public:
  void initialize_(std::string CHIPPlatformStr, std::string CHIPDeviceTypeStr,
                   std::string CHIPDeviceStr) override;

  void uninitialize() override;

  virtual std::string getDefaultJitFlags() override;

  virtual CHIPTexture *createCHIPTexture(intptr_t image_,
                                         intptr_t sampler_) override {
    UNIMPLEMENTED(nullptr);
    // return new CHIPTextureOpenCL();
  }

  virtual CHIPQueue *createCHIPQueue(CHIPDevice *chip_dev) override {
    CHIPDeviceOpenCL *chip_dev_cl = (CHIPDeviceOpenCL *)chip_dev;
    return new CHIPQueueOpenCL(chip_dev_cl);
  }

  // virtual CHIPDevice *createCHIPDevice() override {
  //   return new CHIPDeviceOpenCL();
  // }

  // virtual CHIPContext *createCHIPContext() override {
  //   return new CHIPContextOpenCL();
  // }

  virtual CHIPEvent *createCHIPEvent(CHIPContext *chip_ctx_,
                                     CHIPEventType event_type_) override;

  virtual CHIPCallbackData *createCallbackData(
      hipStreamCallback_t callback, void *userData,
      CHIPQueue *chip_queue_) override {
    UNIMPLEMENTED(nullptr);
  }

  virtual CHIPEventMonitor *createEventMonitor() override {
    return new CHIPEventMonitorOpenCL();
  }
};

#endif