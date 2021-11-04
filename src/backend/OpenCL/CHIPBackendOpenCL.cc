#include "CHIPBackendOpenCL.hh"

// CHIPDeviceOpenCL
// ************************************************************************
CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *chip_ctx_,
                                   cl::Device *dev_in_, int idx_)
    : CHIPDevice(chip_ctx_), cl_dev(dev_in_), cl_ctx(chip_ctx_->get()) {
  logDebug(
      "CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
      "pointer");

  chip_ctx_->addDevice(this);
}

void CHIPDeviceOpenCL::populateDeviceProperties_() {
  logTrace("CHIPDeviceOpenCL->populate_device_properties()");
  cl_int err;
  std::string Temp;

  assert(cl_dev != nullptr);
  Temp = cl_dev->getInfo<CL_DEVICE_NAME>();
  strncpy(hip_device_props.name, Temp.c_str(), 255);
  hip_device_props.name[255] = 0;

  hip_device_props.totalGlobalMem =
      cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);

  hip_device_props.sharedMemPerBlock =
      cl_dev->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  hip_device_props.maxThreadsPerBlock =
      cl_dev->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  std::vector<size_t> wi = cl_dev->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  hip_device_props.maxThreadsDim[0] = wi[0];
  hip_device_props.maxThreadsDim[1] = wi[1];
  hip_device_props.maxThreadsDim[2] = wi[2];

  // Maximum configured clock frequency of the device in MHz.
  hip_device_props.clockRate =
      1000 * cl_dev->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  hip_device_props.multiProcessorCount =
      cl_dev->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  hip_device_props.l2CacheSize =
      cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  hip_device_props.totalConstMem =
      cl_dev->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  hip_device_props.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (cl_dev->getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> sg = cl_dev->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (sg.begin() != sg.end())
      hip_device_props.warpSize = *std::min_element(sg.begin(), sg.end());
  }
  hip_device_props.maxGridSize[0] = hip_device_props.maxGridSize[1] =
      hip_device_props.maxGridSize[2] = 65536;
  hip_device_props.memoryClockRate = 1000;
  hip_device_props.memoryBusWidth = 256;
  hip_device_props.major = 2;
  hip_device_props.minor = 0;

  hip_device_props.maxThreadsPerMultiProcessor = 10;

  hip_device_props.computeMode = 0;
  hip_device_props.arch = {};

  Temp = cl_dev->getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    hip_device_props.arch.hasGlobalInt32Atomics = 1;
  else
    hip_device_props.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    hip_device_props.arch.hasSharedInt32Atomics = 1;
  else
    hip_device_props.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    hip_device_props.arch.hasGlobalInt64Atomics = 1;
    hip_device_props.arch.hasSharedInt64Atomics = 1;
  } else {
    hip_device_props.arch.hasGlobalInt64Atomics = 1;
    hip_device_props.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos)
    hip_device_props.arch.hasDoubles = 1;
  else
    hip_device_props.arch.hasDoubles = 0;

  hip_device_props.clockInstructionRate = 2465;
  hip_device_props.concurrentKernels = 1;
  hip_device_props.pciDomainID = 0;
  hip_device_props.pciBusID = 0x10;
  hip_device_props.pciDeviceID = 0x40 + idx;
  hip_device_props.isMultiGpuBoard = 0;
  hip_device_props.canMapHostMemory = 1;
  hip_device_props.gcnArch = 0;
  hip_device_props.integrated = 0;
  hip_device_props.maxSharedMemoryPerMultiProcessor = 0;
}

void CHIPDeviceOpenCL::reset() { UNIMPLEMENTED(); }
// CHIPEventOpenCL
// ************************************************************************
// CHIPModuleOpenCL
//*************************************************************************
void CHIPModuleOpenCL::compile(CHIPDevice *chip_dev_) {
  CHIPDeviceOpenCL *chip_dev_ocl = (CHIPDeviceOpenCL *)chip_dev_;
  CHIPContextOpenCL *chip_ctx_ocl =
      (CHIPContextOpenCL *)(chip_dev_ocl->getContext());

  int err;
  std::vector<char> binary_vec(src.begin(), src.end());
  auto Program = cl::Program(*(chip_ctx_ocl->get()), binary_vec, false, &err);
  if (err != CL_SUCCESS)
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError);

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string name = chip_dev_ocl->getName();
  int build_failed = Program.build("-x spir -cl-kernel-arg-info");

  std::string log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*chip_dev_ocl->cl_dev, &err);
  if (err != CL_SUCCESS)
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError);
  logDebug("Program BUILD LOG for device #{}:{}:\n{}\n",
           chip_dev_ocl->getDeviceId(), name, log);
  if (build_failed != CL_SUCCESS)
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError);

  std::vector<cl::Kernel> kernels;
  err = Program.createKernels(&kernels);
  if (err != CL_SUCCESS)
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError);
  logDebug("Kernels in CHIPModuleOpenCL: {} \n", kernels.size());
  for (int kernel_idx = 0; kernel_idx < kernels.size(); kernel_idx++) {
    auto kernel = kernels[kernel_idx];
    std::string host_f_name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError,
                                "Failed to fetch OpenCL kernel name");
    auto found_func_info = func_infos.find(host_f_name);
    if (found_func_info == func_infos.end())
      CHIPERR_LOG_AND_THROW("Failed to find kernel in OpenCLFunctionInfoMap",
                            hipErrorInitializationError);
    auto func_info = *(func_infos[host_f_name]);
    CHIPKernelOpenCL *chip_kernel =
        new CHIPKernelOpenCL(std::move(kernel), host_f_name, func_info);
    chip_kernels.push_back(chip_kernel);
  }
}

void CHIPDeviceOpenCL::addQueue(unsigned int flags, int priority) {
  chip_queues.push_back(new CHIPQueueOpenCL(this));
}

// CHIPKernelOpenCL
//*************************************************************************
CHIPKernelOpenCL::CHIPKernelOpenCL(const cl::Kernel &&cl_kernel_,
                                   std::string host_f_name_,
                                   OCLFuncInfo func_info_)
    : CHIPKernel(host_f_name_, func_info_), ocl_kernel(cl_kernel_) {
  int err = 0;
  // TODO attributes
  cl_uint NumArgs = ocl_kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
  CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorTbd,
                              "Failed to get num args for kernel");

  assert(func_info->ArgTypeInfo.size() == NumArgs);

  if (NumArgs > 0) {
    logDebug("Kernel {} numArgs: {} \n", name, NumArgs);
    logDebug("  RET_TYPE: {} {} {}\n", func_info->retTypeInfo.size,
             (unsigned)func_info->retTypeInfo.space,
             (unsigned)func_info->retTypeInfo.type);
    for (auto &argty : func_info->ArgTypeInfo) {
      logDebug("  ARG: SIZE {} SPACE {} TYPE {}\n", argty.size,
               (unsigned)argty.space, (unsigned)argty.type);
      TotalArgSize += argty.size;
    }
  }
}
// CHIPExecItemOpenCL
//*************************************************************************
// CHIPContextOpenCL
//*************************************************************************
CHIPContextOpenCL::CHIPContextOpenCL(cl::Context *ctx_in) {
  logDebug("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  cl_ctx = ctx_in;
}

void *CHIPContextOpenCL::allocate_(size_t size, size_t alignment,
                                   CHIPMemoryType mem_type) {
  void *retval;

  retval = svm_memory.allocate(*cl_ctx, size);
  return retval;
}

hipError_t CHIPContextOpenCL::memCopy(void *dst, const void *src, size_t size,
                                      hipStream_t stream) {
  logWarn("CHIPContextOpenCL::memCopy not implemented");
  // FIND_QUEUE_LOCKED(stream);
  std::lock_guard<std::mutex> Lock(mtx);
  CHIPQueue *Queue = findQueue(stream);
  if (Queue == nullptr) return hipErrorInvalidResourceHandle;

  if (svm_memory.hasPointer(dst) || svm_memory.hasPointer(src))
    return Queue->memCopy(dst, src, size);
  else
    return hipErrorInvalidDevicePointer;
}

// CHIPQueueOpenCL
//*************************************************************************
hipError_t CHIPQueueOpenCL::launch(CHIPExecItem *exec_item) {
  // std::lock_guard<std::mutex> Lock(mtx);
  logTrace("CHIPQueueOpenCL->launch()");
  CHIPExecItemOpenCL *chip_ocl_exec_item = (CHIPExecItemOpenCL *)exec_item;
  CHIPKernelOpenCL *kernel =
      (CHIPKernelOpenCL *)chip_ocl_exec_item->getKernel();
  assert(kernel != nullptr);
  logTrace("Launching Kernel {}", kernel->get_name());

  chip_ocl_exec_item->setup_all_args(kernel);

  dim3 GridDim = chip_ocl_exec_item->getGrid();
  dim3 BlockDim = chip_ocl_exec_item->getBlock();

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = cl_q->enqueueNDRangeKernel(kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorTbd);
  hipError_t retval = hipSuccess;

  // TODO
  // cl_event LastEvent;
  // if (retval == hipSuccess) {
  //   if (LastEvent != nullptr) {
  //     logDebug("Launch: LastEvent == {}, will be: {}", (void *)LastEvent,
  //              (void *)ev.get());
  //     clReleaseEvent(LastEvent);
  //   } else
  //     logDebug("launch: LastEvent == NULL, will be: {}\n", (void *)ev.get());
  //   LastEvent = ev.get();
  //   clRetainEvent(LastEvent);
  // }

  // TODO remove this
  // delete chip_ocl_exec_item;
  return retval;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(CHIPDevice *chip_device_)
    : CHIPQueue(chip_device_) {
  cl_ctx = ((CHIPContextOpenCL *)chip_context)->get();
  cl_dev = ((CHIPDeviceOpenCL *)chip_device)->get();

  cl_q = new cl::CommandQueue(*cl_ctx, *cl_dev);

  chip_device_->addQueue(this);
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {
  delete cl_ctx;
  delete cl_dev;
}

hipError_t CHIPQueueOpenCL::memCopy(void *dst, const void *src, size_t size) {
  std::lock_guard<std::mutex> Lock(mtx);
  logDebug("clSVMmemcpy {} -> {} / {} B\n", src, dst, size);
  cl_event ev = nullptr;
  auto LastEvent = ev;
  int retval = ::clEnqueueSVMMemcpy(cl_q->get(), CL_FALSE, dst, src, size, 0,
                                    nullptr, &ev);
  CHIPERR_CHECK_LOG_AND_THROW(retval, CL_SUCCESS, hipErrorRuntimeMemory);
  if (LastEvent != nullptr) {
    logDebug("memCopy: LastEvent == {}, will be: {}", (void *)LastEvent,
             (void *)ev);
    clReleaseEvent(LastEvent);
  } else {
    logDebug("memCopy: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  }
  return hipSuccess;
}

hipError_t CHIPQueueOpenCL::memCopyAsync(void *dst, const void *src,
                                         size_t size) {
  UNIMPLEMENTED(hipErrorUnknown);
}

void CHIPQueueOpenCL::finish() { UNIMPLEMENTED(); }

static int setLocalSize(size_t shared, OCLFuncInfo *FuncInfo,
                        cl_kernel kernel) {
  logWarn("setLocalSize");
  int err = CL_SUCCESS;

  if (shared > 0) {
    logDebug("setLocalMemSize to {}\n", shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn(
          "Can't set the dynamic local size, "
          "because the kernel doesn't use any local memory.\n");
    } else {
      err = ::clSetKernelArg(kernel, LastArgIdx, shared, nullptr);
      CHIPERR_CHECK_LOG_AND_THROW(
          err, CL_SUCCESS, hipErrorTbd,
          "clSetKernelArg() failed to set dynamic local size");
    }
  }

  return err;
}

int CHIPExecItemOpenCL::setup_all_args(CHIPKernelOpenCL *kernel) {
  OCLFuncInfo *FuncInfo = kernel->get_func_info();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  if ((offset_sizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
    CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
  }

  if (offset_sizes.size() == 0) return CL_SUCCESS;

  std::sort(offset_sizes.begin(), offset_sizes.end());
  if ((std::get<0>(offset_sizes[0]) != 0) ||
      (std::get<1>(offset_sizes[0]) == 0)) {
    CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
  }

  // check args are set
  if (offset_sizes.size() > 1) {
    for (size_t i = 1; i < offset_sizes.size(); ++i) {
      if ((std::get<0>(offset_sizes[i]) == 0) ||
          (std::get<1>(offset_sizes[i]) == 0) ||
          ((std::get<0>(offset_sizes[i - 1]) +
            std::get<1>(offset_sizes[i - 1])) > std::get<0>(offset_sizes[i]))) {
        CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
      }
    }
  }

  const unsigned char *start = arg_data.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < offset_sizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(offset_sizes[i]), std::get<1>(offset_sizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(offset_sizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(offset_sizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      size_t size = std::get<1>(offset_sizes[i]);
      size_t offs = std::get<0>(offset_sizes[i]);
      void *value = (void *)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      err = ::clSetKernelArg(kernel->get().get(), i, size, value);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(shared_mem, FuncInfo, kernel->get().get());
}
// CHIPBackendOpenCL
//*************************************************************************
void CHIPBackendOpenCL::initialize_(std::string CHIPPlatformStr,
                                    std::string CHIPDeviceTypeStr,
                                    std::string CHIPDeviceStr) {
  logDebug("CHIPBackendOpenCL Initialize");
  std::vector<cl::Platform> Platforms;
  cl_int err = cl::Platform::get(&Platforms);
  if (err != CL_SUCCESS)
    CHIPERR_CHECK_LOG_AND_THROW(err, CL_SUCCESS, hipErrorInitializationError);
  std::cout << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
  for (int i = 0; i < Platforms.size(); i++) {
    std::cout << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
  }

  std::vector<cl::Device> enabled_devices;
  std::vector<cl::Device> Devices;
  int selected_platform;
  int selected_device;
  cl_bitfield selected_dev_type = 0;

  try {
    if (!CHIPDeviceStr.compare("all")) {  // Use all devices that match type
      selected_device = -1;
    } else {
      selected_device = std::stoi(CHIPDeviceStr);
    }

    // Platform index in range?
    selected_platform = std::stoi(CHIPPlatformStr);
    if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
      throw InvalidPlatformOrDeviceNumber(
          "CHIP_PLATFORM: platform number out of range");
    std::cout << "Selected Platform: " << selected_platform << ". "
              << Platforms[selected_platform].getInfo<CL_PLATFORM_NAME>()
              << "\n";

    // Device  index in range?
    err =  // Get All devices and print
        Platforms[selected_platform].getDevices(CL_DEVICE_TYPE_ALL, &Devices);
    for (int i = 0; i < Devices.size(); i++) {
      std::cout << i << ". " << Devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }
    if (selected_device >= Devices.size())
      throw InvalidPlatformOrDeviceNumber(
          "CHIP_DEVICE: device number out of range");
    if (selected_device == -1) {  // All devices enabled
      enabled_devices = Devices;
      logDebug("All Devices enabled\n", "");
    } else {
      enabled_devices.push_back(Devices[selected_device]);
      std::cout << "\nEnabled Devices:\n";
      std::cout << selected_device << ". "
                << enabled_devices[0].getInfo<CL_DEVICE_NAME>() << "\n";
    }

    if (err != CL_SUCCESS)
      throw InvalidPlatformOrDeviceNumber(
          "CHIP_DEVICE: can't get devices for platform");

    std::transform(CHIPDeviceTypeStr.begin(), CHIPDeviceTypeStr.end(),
                   CHIPDeviceTypeStr.begin(), ::tolower);
    if (CHIPDeviceTypeStr == "all")
      selected_dev_type = CL_DEVICE_TYPE_ALL;
    else if (CHIPDeviceTypeStr == "cpu")
      selected_dev_type = CL_DEVICE_TYPE_CPU;
    else if (CHIPDeviceTypeStr == "gpu")
      selected_dev_type = CL_DEVICE_TYPE_GPU;
    else if (CHIPDeviceTypeStr == "default")
      selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
    else if (CHIPDeviceTypeStr == "accel")
      selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
    else
      throw InvalidDeviceType("Unknown value provided for CHIP_DEVICE_TYPE\n");
    std::cout << "Using Devices of type " << CHIPDeviceTypeStr << "\n";

  } catch (const InvalidDeviceType &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const InvalidPlatformOrDeviceNumber &e) {
    logCritical("{}\n", e.what());
    return;
  } catch (const std::invalid_argument &e) {
    logCritical("Could not convert CHIP_PLATFORM or CHIP_DEVICES to a number");
    return;
  } catch (const std::out_of_range &e) {
    logCritical("CHIP_PLATFORM or CHIP_DEVICES is out of range", "");
    return;
  }

  std::vector<cl::Device> spirv_enabled_devices;
  for (cl::Device dev : enabled_devices) {
    std::string ver = dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
    if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
      spirv_enabled_devices.push_back(dev);
    }
  }

  // TODO uncomment this once testing on SPIR-V Enabled OpenCL HW
  // std::cout << "SPIR-V Enabled Devices: " << spirv_enabled_devices.size()
  //          << "\n";
  // for (int i = 0; i < spirv_enabled_devices.size(); i++) {
  //  std::cout << i << ". "
  //            << spirv_enabled_devices[i].getInfo<CL_DEVICE_NAME>() <<
  //            "\n";
  //}

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  // TODO Change this to spirv_enabled_devices
  cl::Context *ctx = new cl::Context(enabled_devices);
  CHIPContextOpenCL *chip_context = new CHIPContextOpenCL(ctx);
  Backend->addContext(chip_context);
  for (int i = 0; i < enabled_devices.size(); i++) {
    cl::Device *dev = new cl::Device(enabled_devices[i]);
    CHIPDeviceOpenCL *chip_dev = new CHIPDeviceOpenCL(chip_context, dev, i);
    logDebug("CHIPDeviceOpenCL {}",
             chip_dev->cl_dev->getInfo<CL_DEVICE_NAME>());
    chip_dev->populateDeviceProperties();
    // Backend->addDevice(chip_dev);
    CHIPQueueOpenCL *queue = new CHIPQueueOpenCL(chip_dev);
    // chip_dev->addQueue(queue);
    Backend->addQueue(queue);
  }
  std::cout << "OpenCL Context Initialized.\n";
};

void CHIPBackendOpenCL::uninitialize() { UNIMPLEMENTED(); }

// Other
//*************************************************************************

std::string resultToString(int status) {
  switch (status) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_RESOURCES:
      return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_DEVICE_NOT_FOUND";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_MEM_COPY_OVERLAP:
      return "CL_DEVICE_NOT_FOUND";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_DEVICE_NOT_FOUND";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_DEVICE_NOT_FOUND";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_MAP_FAILURE:
      return "CL_DEVICE_NOT_FOUND";
#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_DEVICE_NOT_FOUND";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_DEVICE_NOT_FOUND";
#endif
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_NOT_FOUND";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_FOUND";
#endif
    case (CL_INVALID_VALUE):
      return "CL_INVALID_VALUE";
    case (CL_INVALID_DEVICE_TYPE):
      return "CL_INVALID_DEVICE_TYPE";
    case (CL_INVALID_PLATFORM):
      return "CL_INVALID_PLATFORM";
    case (CL_INVALID_DEVICE):
      return "CL_INVALID_DEVICE";
    case (CL_INVALID_CONTEXT):
      return "CL_INVALID_CONTEXT";
    case (CL_INVALID_QUEUE_PROPERTIES):
      return "CL_INVALID_QUEUE_PROPERTIES";
    case (CL_INVALID_COMMAND_QUEUE):
      return "CL_INVALID_COMMAND_QUEUE";
    case (CL_INVALID_HOST_PTR):
      return "CL_INVALID_HOST_PTR";
    case (CL_INVALID_MEM_OBJECT):
      return "CL_INVALID_MEM_OBJECT";
    case (CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case (CL_INVALID_IMAGE_SIZE):
      return "CL_INVALID_IMAGE_SIZE";
    case (CL_INVALID_SAMPLER):
      return "CL_INVALID_SAMPLER";
    case (CL_INVALID_BINARY):
      return "CL_INVALID_BINARY";
    case (CL_INVALID_BUILD_OPTIONS):
      return "CL_INVALID_BUILD_OPTIONS";
    case (CL_INVALID_PROGRAM):
      return "CL_INVALID_PROGRAM";
    case (CL_INVALID_PROGRAM_EXECUTABLE):
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case (CL_INVALID_KERNEL_NAME):
      return "CL_INVALID_KERNEL_NAME";
    case (CL_INVALID_KERNEL_DEFINITION):
      return "CL_INVALID_KERNEL_DEFINITION";
    case (CL_INVALID_KERNEL):
      return "CL_INVALID_KERNEL";
    case (CL_INVALID_ARG_INDEX):
      return "CL_INVALID_ARG_INDEX";
    case (CL_INVALID_ARG_VALUE):
      return "CL_INVALID_ARG_VALUE";
    case (CL_INVALID_ARG_SIZE):
      return "CL_INVALID_ARG_SIZE";
    case (CL_INVALID_KERNEL_ARGS):
      return "CL_INVALID_KERNEL_ARGS";
    case (CL_INVALID_WORK_DIMENSION):
      return "CL_INVALID_WORK_DIMENSION";
    case (CL_INVALID_WORK_GROUP_SIZE):
      return "CL_INVALID_WORK_GROUP_SIZE";
    case (CL_INVALID_WORK_ITEM_SIZE):
      return "CL_INVALID_WORK_ITEM_SIZE";
    case (CL_INVALID_GLOBAL_OFFSET):
      return "CL_INVALID_GLOBAL_OFFSET";
    case (CL_INVALID_EVENT_WAIT_LIST):
      return "CL_INVALID_EVENT_WAIT_LIST";
    case (CL_INVALID_EVENT):
      return "CL_INVALID_EVENT";
    case (CL_INVALID_OPERATION):
      return "CL_INVALID_OPERATION";
    case (CL_INVALID_GL_OBJECT):
      return "CL_INVALID_GL_OBJECT";
    case (CL_INVALID_BUFFER_SIZE):
      return "CL_INVALID_BUFFER_SIZE";
    case (CL_INVALID_MIP_LEVEL):
      return "CL_INVALID_MIP_LEVEL";
    case (CL_INVALID_GLOBAL_WORK_SIZE):
      return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
    case (CL_INVALID_PROPERTY):
      return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
    case (CL_INVALID_IMAGE_DESCRIPTOR):
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case (CL_INVALID_COMPILER_OPTIONS):
      return "CL_INVALID_COMPILER_OPTIONS";
    case (CL_INVALID_LINKER_OPTIONS):
      return "CL_INVALID_LINKER_OPTIONS";
    case (CL_INVALID_DEVICE_PARTITION_COUNT):
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
    case (CL_INVALID_PIPE_SIZE):
      return "CL_INVALID_PIPE_SIZE";
    case (CL_INVALID_DEVICE_QUEUE):
      return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
    case (CL_INVALID_SPEC_ID):
      return "CL_INVALID_SPEC_ID";
    case (CL_MAX_SIZE_RESTRICTION_EXCEEDED):
      return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
    default:
      return "CL_UNKNOWN_ERROR";
  }
}
