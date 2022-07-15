#include "CHIPBackendOpenCL.hh"
#include "Utils.hh"

#include <sstream>

#include "Utils.hh"

static cl_sampler createSampler(cl_context Ctx, const hipResourceDesc &ResDesc,
                                const hipTextureDesc &TexDesc) {

  bool IsNormalized = TexDesc.normalizedCoords != 0;

  // Identify the address mode
  cl_addressing_mode AddressMode = CL_ADDRESS_NONE;
  if (ResDesc.resType == hipResourceTypeLinear)
    // "This [address mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    // Effectively out-of-bound references are undefined.
    AddressMode = CL_ADDRESS_NONE;
  else if (TexDesc.addressMode[0] == hipAddressModeClamp)
    AddressMode = CL_ADDRESS_CLAMP_TO_EDGE;
  else if (TexDesc.addressMode[0] == hipAddressModeBorder)
    AddressMode = CL_ADDRESS_CLAMP;
  else if (!IsNormalized)
    // "... if cudaTextureDesc::normalizedCoords is set to zero,
    // cudaAddressModeWrap and cudaAddressModeMirror won't be
    // supported and will be switched to cudaAddressModeClamp."
    // - CUDA 11.6.1/CUDA Runtime API.
    AddressMode = CL_ADDRESS_CLAMP_TO_EDGE;
  else if (TexDesc.addressMode[0] == hipAddressModeWrap)
    AddressMode = CL_ADDRESS_REPEAT;
  else if (TexDesc.addressMode[0] == hipAddressModeMirror)
    AddressMode = CL_ADDRESS_MIRRORED_REPEAT;

  // Identify the filter mode
  cl_filter_mode FilterMode = CL_FILTER_NEAREST;
  if (ResDesc.resType == hipResourceTypeLinear)
    // "This [filter mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    FilterMode = CL_FILTER_NEAREST;
  else if (TexDesc.filterMode == hipFilterModePoint)
    FilterMode = CL_FILTER_NEAREST;
  else if (TexDesc.filterMode == hipFilterModeLinear)
    FilterMode = CL_FILTER_LINEAR;

  cl_sampler_properties SamplerProps[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                          TexDesc.normalizedCoords != 0,
                                          CL_SAMPLER_ADDRESSING_MODE,
                                          AddressMode,
                                          CL_SAMPLER_FILTER_MODE,
                                          FilterMode,
                                          0};
  cl_int Status = CL_SUCCESS;
  auto Sampler = clCreateSamplerWithProperties(Ctx, SamplerProps, &Status);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return Sampler;
}

static cl_mem_object_type getImageType(unsigned HipTextureID) {
  switch (HipTextureID) {
  default:
  case hipTextureTypeCubemap:
  case hipTextureTypeCubemapLayered:
    break;
  case hipTextureType1D:
    return CL_MEM_OBJECT_IMAGE1D;
  case hipTextureType2D:
    return CL_MEM_OBJECT_IMAGE2D;
  case hipTextureType3D:
    return CL_MEM_OBJECT_IMAGE3D;
  case hipTextureType1DLayered:
    return CL_MEM_OBJECT_IMAGE1D_ARRAY;
  case hipTextureType2DLayered:
    return CL_MEM_OBJECT_IMAGE2D_ARRAY;
  }
  CHIPASSERT(false && "Unknown or unsupported HIP texture type.");
  return CL_MEM_OBJECT_IMAGE2D;
}

static const std::map<std::tuple<int, hipChannelFormatKind, bool>,
                      cl_channel_type>
    IntChannelTypeMap = {
        {{8, hipChannelFormatKindSigned, false}, CL_SIGNED_INT8},
        {{16, hipChannelFormatKindSigned, false}, CL_SIGNED_INT16},
        {{32, hipChannelFormatKindSigned, false}, CL_SIGNED_INT32},
        {{8, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT8},
        {{16, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT16},
        {{32, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT32},
        {{8, hipChannelFormatKindSigned, true}, CL_SNORM_INT8},
        {{16, hipChannelFormatKindSigned, true}, CL_SNORM_INT16},
        {{8, hipChannelFormatKindUnsigned, true}, CL_UNORM_INT8},
        {{16, hipChannelFormatKindUnsigned, true}, CL_UNORM_INT16},
};

static cl_image_format getImageFormat(hipChannelFormatDesc Desc,
                                      bool NormalizedFloat) {

  cl_image_format ImageFormat;
  switch (Desc.f) {
  default:
    CHIPERR_LOG_AND_THROW("Unsupported texel type kind.", hipErrorTbd);
  case hipChannelFormatKindUnsigned:
  case hipChannelFormatKindSigned: {
    if (Desc.x > 16)
      // "Note that this [cudaTextureReadMode] applies only to 8-bit and 16-bit
      // integer formats. 32-bit integer format would not be promoted,
      // regardless of whether or not this cudaTextureDesc::readMode is set
      // cudaReadModeNormalizedFloat is specified."
      //
      // - CUDA 11.6.1/CUDA Runtime API.
      NormalizedFloat = false;

    auto I = IntChannelTypeMap.find(
        std::make_tuple(Desc.x, Desc.f, NormalizedFloat));
    if (I == IntChannelTypeMap.end())
      CHIPERR_LOG_AND_THROW("Unsupported integer texel size.", hipErrorTbd);
    ImageFormat.image_channel_data_type = I->second;
    break;
  }
  case hipChannelFormatKindFloat: {
    if (Desc.x != 32)
      CHIPERR_LOG_AND_THROW("Unsupported float texel size.", hipErrorTbd);
    ImageFormat.image_channel_data_type = CL_FLOAT;
    break;
  }
  }

  // Check the layout is one of: [X, 0, 0, 0], [X, X, 0, 0] or [X, X, X, X].
  if (!((Desc.y == 0 && Desc.z == 0 && Desc.w == 0) ||
        (Desc.y == Desc.x && Desc.z == 0 && Desc.w == 0) ||
        (Desc.y == Desc.x && Desc.z == Desc.x && Desc.w == Desc.x)))
    CHIPERR_LOG_AND_THROW("Unsupported channel layout.", hipErrorTbd);

  unsigned NumChannels = 1u + (Desc.y != 0) + (Desc.z != 0) + (Desc.w != 0);
  constexpr cl_channel_order ChannelOrders[4] = {CL_R, CL_RG, CL_RGB, CL_RGBA};
  ImageFormat.image_channel_order = ChannelOrders[NumChannels - 1];

  return ImageFormat;
}

static cl_image_desc getImageDescription(unsigned HipTextureTypeID,
                                         size_t Width, size_t Height = 0,
                                         size_t Depth = 0) {
  cl_image_desc ImageDesc;
  memset(&ImageDesc, 0, sizeof(cl_image_desc));
  ImageDesc.image_type = getImageType(HipTextureTypeID);
  ImageDesc.image_width = Width;
  ImageDesc.image_height = Height;
  ImageDesc.image_depth = Depth;
  ImageDesc.image_array_size = Depth;
  ImageDesc.image_row_pitch = 0;
  ImageDesc.image_slice_pitch = 0;
  ImageDesc.num_mip_levels = 0;
  ImageDesc.num_samples = 0;
  return ImageDesc;
}

static cl_mem createImage(cl_context Ctx, unsigned int TextureType,
                          hipChannelFormatDesc Format, bool NormalizedFloat,
                          size_t Width, size_t Height = 0, size_t Depth = 0) {

  cl_image_format ImageFormat = getImageFormat(Format, NormalizedFloat);
  cl_image_desc ImageDesc =
      getImageDescription(TextureType, Width, Height, Depth);
  cl_int Status;
  // These must be zero when host_ptr argument is NULL.
  CHIPASSERT(ImageDesc.image_row_pitch == 0);
  CHIPASSERT(ImageDesc.image_slice_pitch == 0);
  cl_mem Image = clCreateImage(Ctx, CL_MEM_READ_ONLY, &ImageFormat, &ImageDesc,
                               nullptr, &Status);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  return Image;
}

static void memCopyToImage(cl_command_queue CmdQ, cl_mem Image,
                           const void *HostSrc, const CHIPRegionDesc &SrcRegion,
                           bool BlockingCopy = true) {

  size_t InputRowPitch = SrcRegion.isPitched() ? SrcRegion.Pitch[0] : 0;
  size_t InputSlicePitch = 0;
  if (SrcRegion.isPitched() && SrcRegion.getNumDims() > 2)
    // The slice pitch must be zero for non-arrayed 1D and 2D images
    // (OpenCL v2.2/5.3.3).
    InputSlicePitch = SrcRegion.Pitch[1];

  const size_t *DstOrigin = SrcRegion.Offset;
  const size_t *DstRegion = SrcRegion.Size;
  cl_int Status = clEnqueueWriteImage(CmdQ, Image, BlockingCopy, DstOrigin,
                                      DstRegion, InputRowPitch, InputSlicePitch,
                                      HostSrc, 0, nullptr, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

// CHIPCallbackDataLevel0
// ************************************************************************

CHIPCallbackDataOpenCL::CHIPCallbackDataOpenCL(hipStreamCallback_t TheCallback,
                                               void *TheCallbackArgs,
                                               CHIPQueue *ChipQueue)
    : ChipQueue((CHIPQueueOpenCL *)ChipQueue) {
  if (TheCallbackArgs != nullptr)
    CallbackArgs = TheCallbackArgs;
  if (TheCallback == nullptr)
    CHIPERR_LOG_AND_THROW("", hipErrorTbd);
  CallbackF = TheCallback;
}

// CHIPEventMonitorOpenCL
// ************************************************************************
CHIPEventMonitorOpenCL::CHIPEventMonitorOpenCL() : CHIPEventMonitor(){};

void CHIPEventMonitorOpenCL::monitor() {
  logTrace("CHIPEventMonitorOpenCL::monitor()");
  CHIPEventMonitor::monitor();
}

// CHIPDeviceOpenCL
// ************************************************************************

cl::Device *CHIPDeviceOpenCL::get() { return ClDevice; }
CHIPModuleOpenCL *CHIPDeviceOpenCL::addModule(std::string *ModuleStr) {
  CHIPModuleOpenCL *Module = new CHIPModuleOpenCL(ModuleStr);
  ChipModules.insert(std::make_pair(ModuleStr, Module));
  return Module;
}
CHIPTexture *
CHIPDeviceOpenCL::createTexture(const hipResourceDesc *ResDesc,
                                const hipTextureDesc *TexDesc,
                                const struct hipResourceViewDesc *ResViewDesc) {
  logTrace("CHIPDeviceOpenCL::createTexture");

  bool NormalizedFloat = TexDesc->readMode == hipReadModeNormalizedFloat;
  auto *Q = (CHIPQueueOpenCL *)getActiveQueue();

  cl_context CLCtx = ((CHIPContextOpenCL *)getContext())->get()->get();
  cl_sampler Sampler = createSampler(CLCtx, *ResDesc, *TexDesc);

  if (ResDesc->resType == hipResourceTypeArray) {
    hipArray *Array = ResDesc->res.array.array;
    // Checked in CHIPBindings already.
    CHIPASSERT(Array->data && "Invalid hipArray.");
    CHIPASSERT(!Array->isDrv && "Not supported/implemented yet.");
    size_t Width = Array->width;
    size_t Height = Array->height;
    size_t Depth = Array->depth;

    cl_mem Image = createImage(CLCtx, Array->textureType, Array->desc,
                               NormalizedFloat, Width, Height, Depth);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    CHIPRegionDesc SrcRegion = CHIPRegionDesc::from(*Array);
    memCopyToImage(Q->get()->get(), Image, Array->data, SrcRegion);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypeLinear) {
    auto &Res = ResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    cl_mem Image =
        createImage(CLCtx, hipTextureType1D, Res.desc, NormalizedFloat, Width);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::get1DRegion(Width, TexelByteSize);
    memCopyToImage(Q->get()->get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = ResDesc->res.pitch2D;
    assert(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    cl_mem Image = createImage(CLCtx, hipTextureType2D, Res.desc,
                               NormalizedFloat, Res.width, Res.height);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::from(*ResDesc);
    memCopyToImage(Q->get()->get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  CHIPASSERT(false && "Unsupported/unimplemented texture resource type.");
  return nullptr;
}

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *ChipCtx,
                                   cl::Device *DevIn, int Idx)
    : CHIPDevice(ChipCtx, Idx), ClDevice(DevIn), ClContext(ChipCtx->get()) {
  logTrace("CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
           "pointer");
}

void CHIPDeviceOpenCL::populateDevicePropertiesImpl() {
  logTrace("CHIPDeviceOpenCL->populate_device_properties()");
  cl_int Err;
  std::string Temp;

  assert(ClDevice != nullptr);
  Temp = ClDevice->getInfo<CL_DEVICE_NAME>();
  strncpy(HipDeviceProps_.name, Temp.c_str(), 255);
  HipDeviceProps_.name[255] = 0;

  HipDeviceProps_.totalGlobalMem =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&Err);

  HipDeviceProps_.sharedMemPerBlock =
      ClDevice->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&Err);

  HipDeviceProps_.maxThreadsPerBlock =
      ClDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&Err);

  std::vector<size_t> Wi = ClDevice->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  HipDeviceProps_.maxThreadsDim[0] = Wi[0];
  HipDeviceProps_.maxThreadsDim[1] = Wi[1];
  HipDeviceProps_.maxThreadsDim[2] = Wi[2];

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ClDevice->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ClDevice->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem =
      ClDevice->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  HipDeviceProps_.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (ClDevice->getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> Sg = ClDevice->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (Sg.begin() != Sg.end())
      HipDeviceProps_.warpSize = *std::min_element(Sg.begin(), Sg.end());
  }
  HipDeviceProps_.maxGridSize[0] = HipDeviceProps_.maxGridSize[1] =
      HipDeviceProps_.maxGridSize[2] = 65536;
  HipDeviceProps_.memoryClockRate = 1000;
  HipDeviceProps_.memoryBusWidth = 256;
  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor = 10;

  HipDeviceProps_.computeMode = 0;
  HipDeviceProps_.arch = {};

  Temp = ClDevice->getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasSharedInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  } else {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos)
    HipDeviceProps_.arch.hasDoubles = 1;
  else
    HipDeviceProps_.arch.hasDoubles = 0;

  HipDeviceProps_.clockInstructionRate = 2465;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.gcnArch = 0;
  HipDeviceProps_.integrated = 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor = 0;

  auto Max1D2DWidth = ClDevice->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  auto Max2DHeight = ClDevice->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  auto Max3DWidth = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
  auto Max3DHeight = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
  auto Max3DDepth = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

  // Clamp texture dimensions to [0, INT_MAX] because the return value
  // of hipDeviceGetAttribute() is int type.
  HipDeviceProps_.maxTexture1DLinear = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture1D = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture2D[0] = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture2D[1] = clampToInt(Max2DHeight);
  HipDeviceProps_.maxTexture3D[0] = clampToInt(Max3DWidth);
  HipDeviceProps_.maxTexture3D[1] = clampToInt(Max3DHeight);
  HipDeviceProps_.maxTexture3D[2] = clampToInt(Max3DDepth);

  // OpenCL does not have alignment requirements for images that
  // clients should follow.
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;
}

void CHIPDeviceOpenCL::resetImpl() { UNIMPLEMENTED(); }
// CHIPEventOpenCL
// ************************************************************************

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 cl_event ClEvent, CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipContext), Flags), ClEvent(ClEvent) {
  clRetainEvent(ClEvent);
}

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipContext), Flags), ClEvent(nullptr) {}

cl_event CHIPEventOpenCL::peek() { return ClEvent; }
cl_event CHIPEventOpenCL::get() {
  increaseRefCount("get()");
  return ClEvent;
}

uint64_t CHIPEventOpenCL::getFinishTime() {
  int Status;
  uint64_t Ret;
  Status = clGetEventProfilingInfo(ClEvent, CL_PROFILING_COMMAND_END,
                                   sizeof(Ret), &Ret, NULL);

  if (Status != CL_SUCCESS) {
    auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                 sizeof(int), &EventStatus_, NULL);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  }
  // CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd,
  //                             "Failed to query event for profiling info.");
  return Ret;
}

size_t CHIPEventOpenCL::getRefCount() {
  cl_uint RefCount;
  if (ClEvent == nullptr)
    return 0;
  int Status = ::clGetEventInfo(this->peek(), CL_EVENT_REFERENCE_COUNT, 4,
                                &RefCount, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return RefCount;
}

CHIPEventOpenCL::~CHIPEventOpenCL() { ClEvent = nullptr; }

void CHIPEventOpenCL::decreaseRefCount(std::string Reason) {
  std::lock_guard<std::mutex> Lock(Mtx);

  logTrace("CHIPEventOpenCL::decreaseRefCount() msg={}", Msg.c_str());
  if (!ClEvent) {
    logTrace("CHIPEventOpenCL::decreaseRefCount() ClEvent is null. Likely "
             "never recorded. Skipping...");
    return;
  }
  size_t R = getRefCount();
  logTrace("CHIP Refc: {}->{} OpenCL Refc: {}->{} REASON: {}", *Refc_,
           *Refc_ - 1, R, R - 1, Reason);
  (*Refc_)--;
  // Destructor to be called by event monitor once backend is done using it
  // if (*refc == 1) delete this;
}

void CHIPEventOpenCL::increaseRefCount(std::string Reason) {
  std::lock_guard<std::mutex> Lock(Mtx);

  logTrace("CHIPEventOpenCL::increaseRefCount() msg={}", Msg.c_str());
  size_t R = getRefCount();
  logTrace("CHIP Refc: {}->{} OpenCL Refc: {}->{} REASON: {}", *Refc_,
           *Refc_ + 1, R, R + 1, Reason);
  (*Refc_)++;
}

CHIPEventOpenCL *CHIPBackendOpenCL::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  CHIPEventOpenCL *Event =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx, Flags);
  if (UserEvent) {
    Event->increaseRefCount("hipEventCreate");
    Event->track();
  }
  return Event;
}

void CHIPEventOpenCL::recordStream(CHIPQueue *ChipQueue) {
  std::lock_guard<std::mutex> Lock(Mtx);

  logDebug("CHIPEvent::recordStream()");
  assert(ChipQueue->getLastEvent() != nullptr);
  this->takeOver(ChipQueue->getLastEvent());
  EventStatus_ = EVENT_STATUS_RECORDING;
}

void CHIPEventOpenCL::takeOver(CHIPEvent *OtherIn) {
  if (*Refc_ > 1)
    decreaseRefCount("takeOver");
  auto *Other = (CHIPEventOpenCL *)OtherIn;
  this->ClEvent = Other->ClEvent;
  this->Refc_ = Other->Refc_;
  this->Msg = Other->Msg;
}
// void CHIPEventOpenCL::recordStream(CHIPQueue *chip_queue_) {
//   logTrace("CHIPEventOpenCL::recordStream()");
//   /**
//    * each CHIPQueue keeps track of the status of the last enqueue command.
//    This
//    * is done by creating a CHIPEvent and associating it with the newly
//    submitted
//    * command. Each CHIPQueue has a LastEvent field.
//    *
//    * Recording is done by taking ownership of the target queues' LastEvent,
//    * incrementing that event's refcount.
//    */
//
//   auto chip_queue = (CHIPQueueOpenCL *)chip_queue_;
//   auto last_chip_event = (CHIPEventOpenCL *)chip_queue->getLastEvent();

//   // If this event was used previously, clear it
//   // can be >1 because recordEvent can be called >1 on the same event
//   bool fresh_event = true;
//   if (ev != nullptr) {
//     fresh_event = false;
//     decreaseRefCount();
//   }

//   // if no previous event, create a marker event - we always need 2 events to
//   // measure differences
//   assert(chip_queue->getLastEvent() != nullptr);

//   // Take over target queues event
//   this->ev = chip_queue->getLastEvent()->get();
//   this->refc = chip_queue->getLastEvent()->getRefCount();
//   this->msg = chip_queue->getLastEvent()->msg;
//   // if (fresh_event) assert(this->refc  3);

//   event_status = EVENT_STATUS_RECORDING;

//   /**
//    * There's nothing preventing you from calling hipRecordStream multiple
//    times
//    * in a row on the same event. In such case, after the first call, this
//    events
//    * clEvent field is no longer null and the event's refcount has been
//    * incremented.
//    *
//    * From HIP API: If hipEventRecord() has been previously called on this
//    * event, then this call will overwrite any existing state in event.
//    *
//    * hipEventCreate(myEvent); < clEvent is nullptr
//    * hipMemCopy(..., Q1)
//    * Q1.LastEvent = Q1_MemCopyEvent_0.refcount = 1
//    *
//    * hipStreamRecord(myEvent, Q1);
//    * clEvent== Q1_MemCopyEvent_0, refcount 1->2
//    *
//    * hipMemCopy(..., Q1)
//    * Q1.LastEvent = Q1_MemCopyEvent_1.refcount = 1
//    * Q1_MemCopyEvent_0.refcount 2->1
//    *
//    * hipStreamRecord(myEvent, Q1);
//    * Q1_MemCopyEvent_0.refcount 1->0
//    * clEvent==Q1_MemCopyEvent_1, refcount 1->2
//    */
// }

bool CHIPEventOpenCL::wait() {
  logTrace("CHIPEventOpenCL::wait()");

  if (EventStatus_ != EVENT_STATUS_RECORDING) {
    logWarn("Called wait() on an event that isn't active.");
    return false;
  }

  auto Status = clWaitForEvents(1, &ClEvent);

  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return true;
}

bool CHIPEventOpenCL::updateFinishStatus(bool ThrowErrorIfNotReady) {
  logTrace("CHIPEventOpenCL::updateFinishStatus()");
  if (EventStatus_ != EVENT_STATUS_RECORDING)
    return false;

  int UpdatedStatus;
  auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(int), &UpdatedStatus, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  if (UpdatedStatus <= CL_COMPLETE)
    EventStatus_ = EVENT_STATUS_RECORDED;

  return true;
}

float CHIPEventOpenCL::getElapsedTime(CHIPEvent *OtherIn) {
  // Why do I need to lock the context mutex?
  // Can I lock the mutex of this and the other event?
  //

  CHIPEventOpenCL *Other = (CHIPEventOpenCL *)OtherIn;

  if (this->getContext() != Other->getContext())
    CHIPERR_LOG_AND_THROW(
        "Attempted to get elapsed time between two events that are not part of "
        "the same context",
        hipErrorTbd);

  this->updateFinishStatus();
  Other->updateFinishStatus();

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    CHIPERR_LOG_AND_THROW("one of the events isn't/hasn't recorded",
                          hipErrorInvalidHandle);

  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("one of the events hasn't finished",
                          hipErrorNotReady);

  uint64_t Started = this->getFinishTime();
  uint64_t Finished = Other->getFinishTime();

  logTrace("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)this, Started, (void *)Other, Finished);

  // apparently fails for Intel NEO, god knows why
  // assert(Finished >= Started);
  int64_t Elapsed;
  const int64_t NANOSECS = 1000000000;
  if (Finished < Started)
    logWarn("Finished < Started\n");
  Elapsed = Finished - Started;
  int64_t MS = (Elapsed / NANOSECS) * 1000;
  int64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  return (float)MS + FractInMS;
}

void CHIPEventOpenCL::hostSignal() { UNIMPLEMENTED(); }

// CHIPModuleOpenCL
//*************************************************************************

CHIPModuleOpenCL::CHIPModuleOpenCL(std::string *ModuleStr)
    : CHIPModule(ModuleStr){};

cl::Program *CHIPModuleOpenCL::get() { return &Program_; }

void CHIPModuleOpenCL::compile(CHIPDevice *ChipDev) {

  // TODO make compile_ which calls consumeSPIRV()
  logTrace("CHIPModuleOpenCL::compile()");
  consumeSPIRV();
  CHIPDeviceOpenCL *ChipDevOcl = (CHIPDeviceOpenCL *)ChipDev;
  CHIPContextOpenCL *ChipCtxOcl =
      (CHIPContextOpenCL *)(ChipDevOcl->getContext());

  int Err;
  std::vector<char> BinaryVec(Src_.begin(), Src_.end());
  auto Program = cl::Program(*(ChipCtxOcl->get()), BinaryVec, false, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string Name = ChipDevOcl->getName();
  Err = Program.build(Backend->getJitFlags().c_str());
  auto ErrBuild = Err;

  std::string Log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*ChipDevOcl->ClDevice, &Err);
  logTrace("Program BUILD LOG for device #{}:{}:\n{}\n",
           ChipDevOcl->getDeviceId(), Name, Log);
  CHIPERR_CHECK_LOG_AND_THROW(ErrBuild, CL_SUCCESS,
                              hipErrorInitializationError);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Program BUILD LOG for device #{}:{}:\n{}\n",
           ChipDevOcl->getDeviceId(), Name, Log);

  std::vector<cl::Kernel> Kernels;
  Err = Program.createKernels(&Kernels);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Kernels in CHIPModuleOpenCL: {} \n", Kernels.size());
  for (int KernelIdx = 0; KernelIdx < Kernels.size(); KernelIdx++) {
    auto Kernel = Kernels[KernelIdx];
    std::string HostFName = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&Err);
    CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError,
                                "Failed to fetch OpenCL kernel name");
    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      continue; // TODO
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }
    CHIPKernelOpenCL *ChipKernel = new CHIPKernelOpenCL(
        std::move(Kernel), ChipDevOcl, HostFName, FuncInfo, this);
    addKernel(ChipKernel);
  }
}

CHIPQueue *CHIPDeviceOpenCL::addQueueImpl(unsigned int Flags, int Priority) {
  CHIPQueueOpenCL *NewQ = new CHIPQueueOpenCL(this);
  return NewQ;
}

CHIPQueue *CHIPDeviceOpenCL::addQueueImpl(const uintptr_t *NativeHandles,
                                          int NumHandles) {
  cl_command_queue CmdQ = (cl_command_queue)NativeHandles[3];
  CHIPQueueOpenCL *NewQ = new CHIPQueueOpenCL(this, CmdQ);
  return NewQ;
}

// CHIPKernelOpenCL
//*************************************************************************

OCLFuncInfo *CHIPKernelOpenCL::getFuncInfo() const { return FuncInfo_; }
std::string CHIPKernelOpenCL::getName() { return Name_; }
cl::Kernel *CHIPKernelOpenCL::get() { return &OclKernel_; }
size_t CHIPKernelOpenCL::getTotalArgSize() const { return TotalArgSize_; };

hipError_t CHIPKernelOpenCL::getAttributes(hipFuncAttributes *Attr) {

  Attr->binaryVersion = 10;
  Attr->cacheModeCA = 0;

  Attr->constSizeBytes = 0; // TODO
  Attr->localSizeBytes = PrivateSize_;

  Attr->maxThreadsPerBlock = MaxWorkGroupSize_;
  Attr->sharedSizeBytes = StaticLocalSize_;
  Attr->maxDynamicSharedSizeBytes = MaxDynamicLocalSize_;

  Attr->numRegs = 0;
  Attr->preferredShmemCarveout = 0;
  Attr->ptxVersion = 10;

  return hipSuccess;
}

CHIPKernelOpenCL::CHIPKernelOpenCL(const cl::Kernel &&ClKernel,
                                   CHIPDeviceOpenCL *Dev, std::string HostFName,
                                   OCLFuncInfo *FuncInfo,
                                   CHIPModuleOpenCL *Parent)
    : CHIPKernel(HostFName, FuncInfo), Module(Parent), TotalArgSize_(0),
      Device(Dev) {

  OclKernel_ = ClKernel;
  int Err = 0;
  // TODO attributes
  cl_uint NumArgs = OclKernel_.getInfo<CL_KERNEL_NUM_ARGS>(&Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                              "Failed to get num args for kernel");
  assert(FuncInfo_->ArgTypeInfo.size() == NumArgs);

  MaxWorkGroupSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*Device->get());
  StaticLocalSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(*Device->get());
  MaxDynamicLocalSize_ =
      (size_t)Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  PrivateSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(*Device->get());

  Name_ = OclKernel_.getInfo<CL_KERNEL_FUNCTION_NAME>();

  if (NumArgs > 0) {
    logTrace("Kernel {} numArgs: {} \n", Name_, NumArgs);
    logTrace("  RET_TYPE: {} {} {}\n", FuncInfo_->RetTypeInfo.Size,
             (unsigned)FuncInfo_->RetTypeInfo.Space,
             (unsigned)FuncInfo_->RetTypeInfo.Type);
    for (auto &Argty : FuncInfo_->ArgTypeInfo) {
      logTrace("  ARG: SIZE {} SPACE {} TYPE {}\n", Argty.Size,
               (unsigned)Argty.Space, (unsigned)Argty.Type);
      TotalArgSize_ += Argty.Size;
    }
  }
}

// CHIPContextOpenCL
//*************************************************************************

void CHIPContextOpenCL::freeImpl(void *Ptr) { SvmMemory.free(Ptr); }

cl::Context *CHIPContextOpenCL::get() { return ClContext; }
CHIPContextOpenCL::CHIPContextOpenCL(cl::Context *CtxIn) {
  logTrace("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  ClContext = CtxIn;
  SvmMemory.init(*CtxIn);
}

void *CHIPContextOpenCL::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemType,
                                      CHIPHostAllocFlags Flags) {
  void *Retval;

  Retval = SvmMemory.allocate(Size);
  return Retval;
}

// CHIPQueueOpenCL
//*************************************************************************
struct HipStreamCallbackData {
  hipStream_t Stream;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
};

void CL_CALLBACK pfn_notify(cl_event Event, cl_int CommandExecStatus,
                            void *UserData) {
  HipStreamCallbackData *Cbo = (HipStreamCallbackData *)(UserData);
  if (Cbo == nullptr)
    return;
  if (Cbo->Callback == nullptr)
    return;
  Cbo->Callback(Cbo->Stream, Cbo->Status, Cbo->UserData);
  delete Cbo;
}

cl::CommandQueue *CHIPQueueOpenCL::get() { return ClQueue_; }

void CHIPQueueOpenCL::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  logTrace("CHIPQueueOpenCL::addCallback()");
  auto Ev = getLastEvent();
  if (Ev == nullptr) {
    Callback(this, hipSuccess, UserData);
    return;
  }

  HipStreamCallbackData *Cb =
      new HipStreamCallbackData{this, hipSuccess, UserData, Callback};

  auto Status = clSetEventCallback(Ev->peek(), CL_COMPLETE, pfn_notify, Cb);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  // enqueue barrier with no dependencies (all further enqueues will wait for
  // this one to finish)

  enqueueBarrier(nullptr);
  return;
};

CHIPEvent *CHIPQueueOpenCL::enqueueMarkerImpl() {
  cl::Event MarkerEvent;
  auto Status = this->get()->enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  CHIPEventOpenCL *CHIPMarkerEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, MarkerEvent.get());
  CHIPMarkerEvent->Msg = "marker";
  return CHIPMarkerEvent;
}

CHIPEventOpenCL *CHIPQueueOpenCL::getLastEvent() {
  return (CHIPEventOpenCL *)LastEvent_;
}

CHIPEvent *CHIPQueueOpenCL::launchImpl(CHIPExecItem *ExecItem) {
  //
  logTrace("CHIPQueueOpenCL->launch()");
  CHIPExecItemOpenCL *ChipOclExecItem = (CHIPExecItemOpenCL *)ExecItem;
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)ChipOclExecItem->getKernel();
  assert(Kernel != nullptr);
  logTrace("Launching Kernel {}", Kernel->getName());

  ChipOclExecItem->setupAllArgs(Kernel);

  dim3 GridDim = ChipOclExecItem->getGrid();
  dim3 BlockDim = ChipOclExecItem->getBlock();

  const cl::NDRange Global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange Local(BlockDim.x, BlockDim.y, BlockDim.z);

  logTrace("Launch GLOBAL: {} {} {}", Global.get()[0], Global.get()[1],
           Global.get()[2]);

  logTrace("Launch LOCAL: {} {} {}", Local.get()[0], Local.get()[1],
           Local.get()[2]);

  cl::Event Ev;
  int Err = ClQueue_->enqueueNDRangeKernel(*Kernel->get(), cl::NullRange,
                                           Global, Local, nullptr, &Ev);

  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);

  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev.get());
  // clRetainEvent(ev.get());
  // updateLastEvent(e);
  return E;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(CHIPDevice *ChipDevice, cl_command_queue Queue)
    : CHIPQueue(ChipDevice) {

  if (Queue)
    ClQueue_ = new cl::CommandQueue(Queue);
  else {
    cl::Context *ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
    cl::Device *ClDevice_ = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
    cl_int Status;
    ClQueue_ = new cl::CommandQueue(*ClContext_, *ClDevice_,
                                    CL_QUEUE_PROFILING_ENABLE, &Status);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS,
                                hipErrorInitializationError);
  }
  /**
   * queues should always have lastEvent. Can't do this in the constuctor
   * because enqueueMarker is virtual and calling overriden virtual methods from
   * constructors is undefined behavior.
   *
   * Also, must call implementation method enqueueMarker_ as opposed to wrapped
   * one (enqueueMarker) because the wrapped method enforces queue semantics
   * which require LastEvent to be initialized.
   *
   */
  updateLastEvent(enqueueMarkerImpl());
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {}

CHIPEvent *CHIPQueueOpenCL::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  logTrace("clSVMmemcpy {} -> {} / {} B\n", Src, Dst, Size);
  cl_event Ev = nullptr;
  int Retval = ::clEnqueueSVMMemcpy(ClQueue_->get(), CL_FALSE, Dst, Src, Size,
                                    0, nullptr, &Ev);
  CHIPERR_CHECK_LOG_AND_THROW(Retval, CL_SUCCESS, hipErrorRuntimeMemory);
  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev);
  return E;
}

void CHIPQueueOpenCL::finish() {
  auto Status = ClQueue_->finish();
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

CHIPEvent *CHIPQueueOpenCL::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  logTrace("clSVMmemfill {} / {} B\n", Dst, Size);
  cl_event Ev = nullptr;
  int Retval = ::clEnqueueSVMMemFill(ClQueue_->get(), Dst, Pattern, PatternSize,
                                     Size, 0, nullptr, &Ev);
  CHIPERR_CHECK_LOG_AND_THROW(Retval, CL_SUCCESS, hipErrorRuntimeMemory);
  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev);
  return E;
};

CHIPEvent *CHIPQueueOpenCL::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  UNIMPLEMENTED(nullptr);
};

CHIPEvent *CHIPQueueOpenCL::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  UNIMPLEMENTED(nullptr);
};

hipError_t CHIPQueueOpenCL::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueOpenCL::getBackendHandles");
  if (*NumHandles < 4) {
    logError("getBackendHandles requires space for 4 handles");
    return hipErrorInvalidValue;
  }
  *NumHandles = 4;

  // Get queue handler
  NativeInfo[3] = (uintptr_t)ClQueue_->get();

  // Get context handler
  cl::Context *Ctx = ((CHIPContextOpenCL *)ChipContext_)->get();
  NativeInfo[2] = (uintptr_t)Ctx->get();

  // Get device handler
  cl::Device *Dev = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
  NativeInfo[1] = (uintptr_t)Dev->get();

  // Get platform handler
  cl_platform_id Plat = Dev->getInfo<CL_DEVICE_PLATFORM>();
  NativeInfo[0] = (uintptr_t)Plat;
  return hipSuccess;
}

CHIPEvent *CHIPQueueOpenCL::memPrefetchImpl(const void *Ptr, size_t Count) {
  UNIMPLEMENTED(nullptr);
}

CHIPEvent *
CHIPQueueOpenCL::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
  cl::Event MarkerEvent;
  int status = ClQueue_->enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd);

  cl::vector<cl::Event> Events = {};
  // if (EventsToWaitFor)
  //   for (auto E : *EventsToWaitFor) {
  //     auto Ee = (CHIPEventOpenCL *)E;
  //     Events.push_back(cl::Event(Ee->peek()));
  //   }
  Events.push_back(MarkerEvent);

  cl::Event Barrier;
  auto Status = ClQueue_->enqueueBarrierWithWaitList(&Events, &Barrier);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  CHIPEventOpenCL *NewEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Barrier.get());

  return NewEvent;
}

static int setLocalSize(size_t Shared, OCLFuncInfo *FuncInfo,
                        cl_kernel Kernel) {
  logTrace("setLocalSize");
  int Err = CL_SUCCESS;

  if (Shared > 0) {
    logTrace("setLocalMemSize to {}\n", Shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].Space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn("Can't set the dynamic local size, "
              "because the kernel doesn't use any local memory.\n");
    } else {
      Err = ::clSetKernelArg(Kernel, LastArgIdx, Shared, nullptr);
      CHIPERR_CHECK_LOG_AND_THROW(
          Err, CL_SUCCESS, hipErrorTbd,
          "clSetKernelArg() failed to set dynamic local size");
    }
  }

  return Err;
}

// CHIPExecItemOpenCL
//*************************************************************************

cl::Kernel *CHIPExecItemOpenCL::get() { return ClKernel_; }

int CHIPExecItemOpenCL::setupAllArgs(CHIPKernelOpenCL *Kernel) {
  OCLFuncInfo *FuncInfo = Kernel->getFuncInfo();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].Space == OCLSpace::Local)
      ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  CHIPASSERT(NumLocals <= 1);
  int Err = 0;

  if (ArgsPointer_) {
    logTrace("Setting up arguments NEW HIP API");
    for (size_t InArgIdx = 0, OutArgIdx = 0;
         OutArgIdx < FuncInfo->ArgTypeInfo.size(); ++OutArgIdx, ++InArgIdx) {
      OCLArgTypeInfo &Ai = FuncInfo->ArgTypeInfo[OutArgIdx];
      if (Ai.Type == OCLType::Pointer) {
        if (Ai.Space == OCLSpace::Local)
          continue;
        logTrace("clSetKernelArgSVMPointer {} SIZE {} to {}\n", OutArgIdx,
                 Ai.Size, ArgsPointer_[InArgIdx]);
        CHIPASSERT(Ai.Size == sizeof(void *));
        const void *Argval = *(void **)ArgsPointer_[InArgIdx];
        Err =
            ::clSetKernelArgSVMPointer(Kernel->get()->get(), OutArgIdx, Argval);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArgSVMPointer failed");
      } else if (Ai.Type == OCLType::Image) {
        auto *TexObj = *(CHIPTextureOpenCL **)ArgsPointer_[InArgIdx];

        // Set image argument.
        cl_mem Image = TexObj->getImage();
        logTrace("set image arg {} for tex {}\n", OutArgIdx, (void *)TexObj);
        Err = ::clSetKernelArg(Kernel->get()->get(), OutArgIdx, sizeof(cl_mem),
                               &Image);
        CHIPERR_CHECK_LOG_AND_THROW(
            Err, CL_SUCCESS, hipErrorTbd,
            "clSetKernelArg failed for image argument.");

        // Set sampler argument.
        OutArgIdx++;
        cl_sampler Sampler = TexObj->getSampler();
        logTrace("set sampler arg {} for tex {}\n", OutArgIdx, (void *)TexObj);
        Err = ::clSetKernelArg(Kernel->get()->get(), OutArgIdx,
                               sizeof(cl_sampler), &Sampler);
        CHIPERR_CHECK_LOG_AND_THROW(
            Err, CL_SUCCESS, hipErrorTbd,
            "clSetKernelArg failed for sampler argument.");
      } else {
        logTrace("clSetKernelArg {} SIZE {} to {}\n", OutArgIdx, Ai.Size,
                 ArgsPointer_[InArgIdx]);
        Err = ::clSetKernelArg(Kernel->get()->get(), OutArgIdx, Ai.Size,
                               ArgsPointer_[InArgIdx]);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArg failed");
      }
    }
  } else {
    logTrace("Setting up arguments OLD HIP API");

    if ((OffsetSizes_.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
    }

    if (OffsetSizes_.size() == 0)
      return CL_SUCCESS;

    std::sort(OffsetSizes_.begin(), OffsetSizes_.end());
    if ((std::get<0>(OffsetSizes_[0]) != 0) ||
        (std::get<1>(OffsetSizes_[0]) == 0)) {
      CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
    }

    // check args are set
    if (OffsetSizes_.size() > 1) {
      for (size_t i = 1; i < OffsetSizes_.size(); ++i) {
        if ((std::get<0>(OffsetSizes_[i]) == 0) ||
            (std::get<1>(OffsetSizes_[i]) == 0) ||
            ((std::get<0>(OffsetSizes_[i - 1]) +
              std::get<1>(OffsetSizes_[i - 1])) >
             std::get<0>(OffsetSizes_[i]))) {
          CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
        }
      }
    }

    const unsigned char *Start = ArgData_.data();
    void *P;
    int Err;
    for (cl_uint i = 0; i < OffsetSizes_.size(); ++i) {
      OCLArgTypeInfo &Ai = FuncInfo->ArgTypeInfo[i];
      logTrace("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetSizes_[i]), std::get<1>(OffsetSizes_[i]),
               (unsigned)Ai.Type, (unsigned)Ai.Space, Ai.Size);

      if (Ai.Type == OCLType::Pointer) {
        // TODO other than global AS ?
        assert(Ai.Size == sizeof(void *));
        assert(std::get<1>(OffsetSizes_[i]) == Ai.Size);
        P = *(void **)(Start + std::get<0>(OffsetSizes_[i]));
        logTrace("setArg SVM {} to {}\n", i, P);
        Err = ::clSetKernelArgSVMPointer(Kernel->get()->get(), i, P);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArgSVMPointer failed");
      } else if (Ai.Type == OCLType::Image) {
        CHIPASSERT(false && "UNIMPLMENTED: Texture argument handling for the "
                            "old HIP kernel ABI.");
      } else {
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offs = std::get<0>(OffsetSizes_[i]);
        void *Value = (void *)(Start + Offs);
        logTrace("setArg {} size {} offs {}\n", i, Size, Offs);
        Err = ::clSetKernelArg(Kernel->get()->get(), i, Size, Value);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArg failed");
      }
    }
  }

  return setLocalSize(SharedMem_, FuncInfo, Kernel->get()->get());
}

// CHIPBackendOpenCL
//*************************************************************************

CHIPQueue *CHIPBackendOpenCL::createCHIPQueue(CHIPDevice *ChipDev) {
  CHIPDeviceOpenCL *ChipDevCl = (CHIPDeviceOpenCL *)ChipDev;
  return new CHIPQueueOpenCL(ChipDevCl);
}

CHIPCallbackData *
CHIPBackendOpenCL::createCallbackData(hipStreamCallback_t Callback,
                                      void *UserData, CHIPQueue *ChipQueue) {
  UNIMPLEMENTED(nullptr);
}

CHIPEventMonitor *CHIPBackendOpenCL::createCallbackEventMonitor() {
  auto Evm = new CHIPEventMonitorOpenCL();
  Evm->start();
  return Evm;
}

CHIPEventMonitor *CHIPBackendOpenCL::createStaleEventMonitor() {
  UNIMPLEMENTED(nullptr);
}

std::string CHIPBackendOpenCL::getDefaultJitFlags() {
  return std::string("-x spir -cl-kernel-arg-info");
}

void CHIPBackendOpenCL::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendOpenCL Initialize");

  // transform device type string into CL
  cl_bitfield SelectedDevType = 0;
  if (CHIPDeviceTypeStr == "all")
    SelectedDevType = CL_DEVICE_TYPE_ALL;
  else if (CHIPDeviceTypeStr == "cpu")
    SelectedDevType = CL_DEVICE_TYPE_CPU;
  else if (CHIPDeviceTypeStr == "gpu")
    SelectedDevType = CL_DEVICE_TYPE_GPU;
  else if (CHIPDeviceTypeStr == "default")
    SelectedDevType = CL_DEVICE_TYPE_DEFAULT;
  else if (CHIPDeviceTypeStr == "accel")
    SelectedDevType = CL_DEVICE_TYPE_ACCELERATOR;
  else
    throw InvalidDeviceType("Unknown value provided for CHIP_DEVICE_TYPE\n");
  logDebug("Using Devices of type {}", CHIPDeviceTypeStr);

  std::vector<cl::Platform> Platforms;
  cl_int Err = cl::Platform::get(&Platforms);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);
  std::stringstream StrStream;
  StrStream << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
  for (int i = 0; i < Platforms.size(); i++) {
    StrStream << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
  }
  logDebug("{}", StrStream.str());
  StrStream.str("");

  StrStream << "OpenCL Devices of type " << CHIPDeviceTypeStr
            << " with SPIR-V_1 support:\n";
  std::vector<cl::Device> Devices;
  for (auto Plat : Platforms) {
    std::vector<cl::Device> Dev;
    Err = Plat.getDevices(SelectedDevType, &Dev);
    CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);
    for (auto D : Dev) {
      std::string Ver = D.getInfo<CL_DEVICE_IL_VERSION>(&Err);
      if ((Err == CL_SUCCESS) && (Ver.rfind("SPIR-V_1.", 0) == 0)) {
        std::string DeviceName = D.getInfo<CL_DEVICE_NAME>();
        StrStream << DeviceName << "\n";
        Devices.push_back(D);
      }
    }
  }
  logDebug("{}", StrStream.str());

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  // TODO Change this to spirv_enabled_devices
  cl::Context *Ctx = new cl::Context(Devices);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx);
  Backend->addContext(ChipContext);
  for (int i = 0; i < Devices.size(); i++) {
    cl::Device *Dev = new cl::Device(Devices[i]);
    CHIPDeviceOpenCL *ChipDev = new CHIPDeviceOpenCL(ChipContext, Dev, i);
    logTrace("CHIPDeviceOpenCL {}",
             ChipDev->ClDevice->getInfo<CL_DEVICE_NAME>());
    ChipDev->populateDeviceProperties();

    // Add device to context & backend
    ChipContext->addDevice(ChipDev);
    Backend->addDevice(ChipDev);

    // Create and add queue queue to Device and Backend
    ChipDev->createQueueAndRegister((int)0, (int)0);
  }
  logDebug("OpenCL Context Initialized.");
};

void CHIPBackendOpenCL::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendOpenCL InitializeNative");

  // cl_platform_id PlatId = (cl_platform_id)NativeHandles[0];
  cl_device_id DevId = (cl_device_id)NativeHandles[1];
  cl_context CtxId = (cl_context)NativeHandles[2];

  cl::Context *Ctx = new cl::Context(CtxId);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx);
  addContext(ChipContext);

  cl::Device *Dev = new cl::Device(DevId);
  CHIPDeviceOpenCL *ChipDev = new CHIPDeviceOpenCL(ChipContext, Dev, 0);
  logTrace("CHIPDeviceOpenCL {}", ChipDev->ClDevice->getInfo<CL_DEVICE_NAME>());
  ChipDev->populateDeviceProperties();

  // Add device to context & backend
  ChipContext->addDevice(ChipDev);
  addDevice(ChipDev);

  ChipDev->createQueueAndRegister(NativeHandles, NumHandles);

  setActiveDevice(ChipDev);

  logDebug("OpenCL Context Initialized.");
}

hipEvent_t CHIPBackendOpenCL::getHipEvent(void *NativeEvent) {
  cl_event E = (cl_event)NativeEvent;
  // this retains cl_event
  CHIPEventOpenCL *NewEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ActiveCtx_, E);
  NewEvent->Msg = "fromHipEvent";
  return NewEvent;
}

void *CHIPBackendOpenCL::getNativeEvent(hipEvent_t HipEvent) {
  CHIPEventOpenCL *E = (CHIPEventOpenCL *)HipEvent;
  if (!E->isRecordingOrRecorded())
    return nullptr;
  // TODO should we retain here?
  // E->increaseRefCount();
  return (void *)E->ClEvent;
}

// Other
//*************************************************************************

std::string resultToString(int Status) {
  switch (Status) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
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
