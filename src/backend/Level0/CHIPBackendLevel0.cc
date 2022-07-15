#include "CHIPBackendLevel0.hh"
#include "Utils.hh"

static ze_image_type_t getImageType(unsigned HipTextureID) {
  switch (HipTextureID) {
  default:
  case hipTextureTypeCubemap:
  case hipTextureTypeCubemapLayered:
    break;
  case hipTextureType1D:
    return ZE_IMAGE_TYPE_1D;
  case hipTextureType2D:
    return ZE_IMAGE_TYPE_2D;
  case hipTextureType3D:
    return ZE_IMAGE_TYPE_3D;
  case hipTextureType1DLayered:
    return ZE_IMAGE_TYPE_1DARRAY;
  case hipTextureType2DLayered:
    return ZE_IMAGE_TYPE_2DARRAY;
  }
  CHIPASSERT(false && "Unknown or unsupported HIP texture type.");
  return ZE_IMAGE_TYPE_2D;
}

#define LAYOUT_KEY(_X, _Y, _Z, _W) (_W << 24 | _Z << 16 | _Y << 8 | _X)

#define LAYOUT_KEY_FROM_FORMAT_DESC(_DESC)                                     \
  LAYOUT_KEY(_DESC.x, _DESC.y, _DESC.z, _DESC.w)

#define DEF_LAYOUT_MAP(_X, _Y, _Z, _W, _LAYOUT)                                \
  case LAYOUT_KEY(_X, _Y, _Z, _W):                                             \
    Result.layout = _LAYOUT;                                                   \
    break

static ze_image_format_t getImageFormat(hipChannelFormatDesc FormatDesc,
                                        bool NormalizedFloat) {
  bool Supported = FormatDesc.f == hipChannelFormatKindUnsigned ||
                   FormatDesc.f == hipChannelFormatKindSigned ||
                   FormatDesc.f == hipChannelFormatKindFloat;
  if (!Supported)
    CHIPERR_LOG_AND_THROW("Unsupported channel description.", hipErrorTbd);

  CHIPASSERT(FormatDesc.x < (1 << 8) && FormatDesc.y < (1 << 8) &&
             FormatDesc.z < (1 << 8) && FormatDesc.w < (1 << 8) && "Overlap");

  ze_image_format_t Result{};
  switch (LAYOUT_KEY_FROM_FORMAT_DESC(FormatDesc)) {
  default:
    CHIPERR_LOG_AND_THROW("Unsupported channel description.", hipErrorTbd);
    break;
    DEF_LAYOUT_MAP(8, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_8);
    DEF_LAYOUT_MAP(8, 8, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_8_8);
    DEF_LAYOUT_MAP(8, 8, 8, 8, ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8);
    DEF_LAYOUT_MAP(16, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_16);
    DEF_LAYOUT_MAP(16, 16, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_16_16);
    DEF_LAYOUT_MAP(16, 16, 16, 16, ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16);
    DEF_LAYOUT_MAP(32, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_32);
    DEF_LAYOUT_MAP(32, 32, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_32_32);
    DEF_LAYOUT_MAP(32, 32, 32, 32, ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32);
  }

  if (FormatDesc.x > 16 && (FormatDesc.f == hipChannelFormatKindUnsigned ||
                            FormatDesc.f == hipChannelFormatKindSigned)) {
    // "Note that this [cudaTextureReadMode] applies only to 8-bit and 16-bit
    // integer formats. 32-bit integer format would not be promoted, regardless
    // of whether or not this cudaTextureDesc::readMode is set
    // cudaReadModeNormalizedFloat is specified."
    // - CUDA 11.6.1/CUDA Runtime API.
    NormalizedFloat = false;
  }

  switch (FormatDesc.f) {
  default:
    CHIPASSERT(false && "Unsupported/unimplemented format type.");
    return Result;
  case hipChannelFormatKindSigned:
    Result.type = NormalizedFloat ? ZE_IMAGE_FORMAT_TYPE_SNORM
                                  : ZE_IMAGE_FORMAT_TYPE_SINT;
    break;
  case hipChannelFormatKindUnsigned:
    Result.type = NormalizedFloat ? ZE_IMAGE_FORMAT_TYPE_UNORM
                                  : ZE_IMAGE_FORMAT_TYPE_UINT;
    break;
  case hipChannelFormatKindFloat:
    Result.type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    break;
  }

  // These fields are for swizzle descriptions.
  Result.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
  Result.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
  Result.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
  Result.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
  return Result;
}

#undef LAYOUT_KEY
#undef LAYOUT_KEY_FROM_FORMAT_DESC
#undef DEF_LAYOUT_MAP

static ze_image_desc_t getImageDescription(unsigned int TextureType,
                                           hipChannelFormatDesc Format,
                                           bool NormalizedFloat, size_t Width,
                                           size_t Height = 0,
                                           size_t Depth = 0) {
  ze_image_desc_t Result{};
  Result.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
  Result.pNext = nullptr;
  Result.flags = 0; // Read-only
  Result.type = getImageType(TextureType);
  Result.format = getImageFormat(Format, NormalizedFloat);
  Result.width = Width;
  Result.height = Height;
  Result.depth = Depth;       // L0 spec: Ignored for non-ZE_IMAGE_TYPE_3D;
  Result.arraylevels = Depth; // L0 spec: Ignored for non-array types.
  Result.miplevels = 0;
  return Result;
}

static ze_sampler_handle_t
createSampler(CHIPDeviceLevel0 *ChipDev, const hipResourceDesc *PResDesc,
              const hipTextureDesc *PTexDesc,
              const struct hipResourceViewDesc *PResViewDesc) {
  logTrace("CHIPTextureLevel0::createSampler");

  // Identify the normalization
  ze_bool_t IsNormalized = PTexDesc->normalizedCoords != 0;

  // Identify the address mode
  ze_sampler_address_mode_t AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  if (PResDesc->resType == hipResourceTypeLinear)
    // "This [address mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    // Effectively out-of-bound references are undefined.
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  else if (PTexDesc->addressMode[0] == hipAddressModeClamp)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (PTexDesc->addressMode[0] == hipAddressModeBorder)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  else if (!IsNormalized)
    // "... if cudaTextureDesc::normalizedCoords is set to zero,
    // cudaAddressModeWrap and cudaAddressModeMirror won't be
    // supported and will be switched to cudaAddressModeClamp."
    // - CUDA 11.6.1/CUDA Runtime API.
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (PTexDesc->addressMode[0] == hipAddressModeWrap)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
  else if (PTexDesc->addressMode[0] == hipAddressModeMirror)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
  else
    CHIPASSERT(false && "Unknown address mode!");

  // Identify the filter mode
  ze_sampler_filter_mode_t FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  if (PResDesc->resType == hipResourceTypeLinear)
    // "This [filter mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (PTexDesc->filterMode == hipFilterModePoint)
    FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (PTexDesc->filterMode == hipFilterModeLinear)
    FilterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
  else
    CHIPASSERT(false && "Unknown filter mode!");

  ze_sampler_desc_t SamplerDesc = {ZE_STRUCTURE_TYPE_SAMPLER_DESC, nullptr,
                                   AddressMode, FilterMode, IsNormalized};

  // Create LZ samler handle
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipDev->getContext();
  ze_sampler_handle_t Sampler{};
  ze_result_t Status =
      zeSamplerCreate(ChipCtxLz->get(), ChipDev->get(), &SamplerDesc, &Sampler);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return Sampler;
}

// CHIPEventLevel0
// ***********************************************************************

ze_event_handle_t CHIPEventLevel0::peek() { return Event_; }

ze_event_handle_t CHIPEventLevel0::get() {
  increaseRefCount("get()");
  return Event_;
}

CHIPEventLevel0::~CHIPEventLevel0() {
  logDebug("chipEventLevel0 DEST {}", (void *)this);
  if (Event_) {
    auto Status = zeEventDestroy(Event_);
    // '~CHIPEventLevel0' has a non-throwing exception specification
    assert(Status == ZE_RESULT_SUCCESS);
  }
  Event_ = nullptr;
  EventPoolHandle_ = nullptr;
  EventPool = nullptr;
  Timestamp_ = 0;
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 LZEventPool *TheEventPool,
                                 unsigned int ThePoolIndex,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr), Timestamp_(0) {
  EventPool = TheEventPool;
  EventPoolIndex = ThePoolIndex;
  EventPoolHandle_ = TheEventPool->get();

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      EventPoolIndex,               // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signaauto l
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event_ completes
  };

  auto Status = zeEventCreate(EventPoolHandle_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr), Timestamp_(0), EventPoolIndex(0),
      EventPool(0) {
  CHIPContextLevel0 *ZeCtx = (CHIPContextLevel0 *)ChipContext_;

  unsigned int PoolFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  // if (!flags.isDisableTiming())
  //   pool_flags = pool_flags | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      1                                  // count
  };

  ze_result_t Status = zeEventPoolCreate(ZeCtx->get(), &EventPoolDesc, 0,
                                         nullptr, &EventPoolHandle_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ pool creation fail! ");

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      0,                            // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signal
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event_ completes
  };

  Status = zeEventCreate(EventPoolHandle_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 ze_event_handle_t NativeEvent)
    : CHIPEvent((CHIPContext *)(ChipCtx)), Event_(NativeEvent),
      EventPoolHandle_(nullptr), Timestamp_(0), EventPoolIndex(0),
      EventPool(nullptr) {}

// Must use this for now - Level Zero hangs when events are host visible +
// kernel timings are enabled
void CHIPEventLevel0::recordStream(CHIPQueue *ChipQueue) {
  ze_result_t Status;

  {
    std::lock_guard<std::mutex> Lock(Mtx);
    if (EventStatus_ == EVENT_STATUS_RECORDED) {
      logTrace("Event {}: EVENT_STATUS_RECORDED ... Resetting event.",
               (void *)this);
      ze_result_t Status = zeEventHostReset(Event_);
      EventStatus_ = EVENT_STATUS_INIT;
      CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    }
  }

  if (ChipQueue == nullptr)
    CHIPERR_LOG_AND_THROW("Queue passed in is null", hipErrorTbd);

  CHIPQueueLevel0 *Q = (CHIPQueueLevel0 *)ChipQueue;

  auto CommandList = Q->getCmdListCompute();
  Status = zeCommandListAppendBarrier(CommandList, nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendWriteGlobalTimestamp(
      CommandList, (uint64_t *)(Q->getSharedBufffer()), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendBarrier(CommandList, nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendMemoryCopy(CommandList, &Timestamp_,
                                         Q->getSharedBufffer(),
                                         sizeof(uint64_t), Event_, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  auto DestoyCommandListEvent =
      ((CHIPBackendLevel0 *)Backend)->createCHIPEvent(this->ChipContext_);
  DestoyCommandListEvent->Msg = "recordStreamComplete";
  Status = zeCommandListAppendBarrier(
      CommandList, DestoyCommandListEvent->get(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Q->executeCommandList(CommandList);
  DestoyCommandListEvent->track();

  std::lock_guard<std::mutex> Lock(Mtx);
  EventStatus_ = EVENT_STATUS_RECORDING;
  Msg = "recordStream";
}

bool CHIPEventLevel0::wait() {
  logTrace("CHIPEventLevel0::wait() msg={}", Msg);

  ze_result_t Status = zeEventHostSynchronize(Event_, UINT64_MAX);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  std::lock_guard<std::mutex> Lock(Mtx);
  EventStatus_ = EVENT_STATUS_RECORDED;
  return true;
}

bool CHIPEventLevel0::updateFinishStatus(bool ThrowErrorIfNotReady) {
  std::lock_guard<std::mutex> Lock(Mtx);

  auto EventStatusOld = getEventStatusStr();

  ze_result_t Status = zeEventQueryStatus(Event_);
  if (Status == ZE_RESULT_NOT_READY && ThrowErrorIfNotReady) {
    CHIPERR_LOG_AND_THROW("Event Not Ready", hipErrorNotReady);
  }
  if (Status == ZE_RESULT_SUCCESS)
    EventStatus_ = EVENT_STATUS_RECORDED;

  auto EventStatusNew = getEventStatusStr();
  if (EventStatusNew != EventStatusOld) {
    logTrace("CHIPEventLevel0::updateFinishStatus() {}: {} -> {}", Msg,
             EventStatusOld, EventStatusNew);
    return true;
  }

  return false;
}

/** This Doesn't work right now due to Level Zero Backend hanging?
  unsinged long CHIPEventLevel0::getFinishTime() {

    ze_kernel_timestamp_result_t res{};
    auto Status = zeEventQueryKernelTimestamp(Event_, &res);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

    CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_context;
    CHIPDeviceLevel0* chip_dev_lz =
        (CHIPDeviceLevel0*)chip_ctx_lz->getDevices()[0];

    auto props = chip_dev_lz->getDeviceProps();

    uint64_t timerResolution = props->timerResolution;
    uint32_t timestampValidBits = props->timestampValidBits;

    return res.context.kernelEnd * timerResolution;
  }
  */

unsigned long CHIPEventLevel0::getFinishTime() {
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevices()[0];
  auto Props = ChipDevLz->getDeviceProps();

  uint64_t TimerResolution = Props->timerResolution;
  uint32_t TimestampValidBits = Props->timestampValidBits;

  uint32_t T = (Timestamp_ & (((uint64_t)1 << TimestampValidBits) - 1));
  T = T * TimerResolution;

  return T;
}

float CHIPEventLevel0::getElapsedTime(CHIPEvent *OtherIn) {
  /**
   * Modified HIPLZ Implementation
   * https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
   */
  logTrace("CHIPEventLevel0::getElapsedTime()");
  CHIPEventLevel0 *Other = (CHIPEventLevel0 *)OtherIn;

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  this->updateFinishStatus();
  Other->updateFinishStatus();
  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("One of the events for getElapsedTime() was done yet",
                          hipErrorNotReady);

  unsigned long Started = this->getFinishTime();
  unsigned long Finished = Other->getFinishTime();

  /**
   *
   * Kernel timestamps execute along a device timeline but because of limited
   * range may wrap unexpectedly. Because of this, the temporal order of two
   * kernel timestamps shouldn’t be inferred despite coincidental START/END
   * values.
   * https://spec.oneapi.io/level-zero/latest/core/PROG.html#kernel-timestamp-events
   */
  uint64_t Elapsed = std::fabs(Finished - Started);

#define NANOSECS 1000000000
  uint64_t MS = (Elapsed / NANOSECS) * 1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  auto Ms = (float)MS + FractInMS;

  return Ms;
}

void CHIPEventLevel0::hostSignal() {
  logTrace("CHIPEventLevel0::hostSignal()");
  auto Status = zeEventHostSignal(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  std::lock_guard<std::mutex> Lock(Mtx);
  EventStatus_ = EVENT_STATUS_RECORDED;
}

// End CHIPEventLevel0

// CHIPCallbackDataLevel0
// ***********************************************************************

CHIPCallbackDataLevel0::CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF,
                                               void *CallbackArgs,
                                               CHIPQueue *ChipQueue)
    : CHIPCallbackData(CallbackF, CallbackArgs, ChipQueue) {
  std::lock_guard<std::mutex> Lock(Mtx);
  CHIPContext *Ctx = ChipQueue->getContext();

  CpuCallbackComplete = (CHIPEventLevel0 *)Backend->createCHIPEvent(Ctx);
  CpuCallbackComplete->Msg = "CpuCallbackComplete";

  GpuReady = ChipQueue->enqueueBarrierImpl(nullptr);
  GpuReady->Msg = "GpuReady";

  std::vector<CHIPEvent *> ChipEvs = {CpuCallbackComplete};
  ChipQueue->enqueueBarrierImpl(&ChipEvs);

  GpuAck = ChipQueue->enqueueMarkerImpl();
  GpuAck->Msg = "GpuAck";
}

// End CHIPCallbackDataLevel0

// CHIPEventMonitorLevel0
// ***********************************************************************

void CHIPCallbackEventMonitorLevel0::monitor() {
  logTrace("CHIPEventMonitorLevel0::monitor()");
  while (Backend->CallbackQueue.size()) {
    std::lock_guard<std::mutex> Lock(Backend->CallbackQueueMtx);

    // get the callback item
    CHIPCallbackDataLevel0 *CallbackData =
        (CHIPCallbackDataLevel0 *)Backend->CallbackQueue.front();

    // Lock the item and members
    std::lock_guard<std::mutex> LockCallbackData(CallbackData->Mtx);
    Backend->CallbackQueue.pop();

    // Update Status
    CallbackData->GpuReady->updateFinishStatus(false);
    if (CallbackData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED) {
      // if not ready, push to the back
      Backend->CallbackQueue.push(CallbackData);
      pthread_yield();
      continue;
    }

    CallbackData->execute(hipSuccess);
    CallbackData->CpuCallbackComplete->hostSignal();
    CallbackData->GpuAck->wait();

    delete CallbackData;
    pthread_yield();
  }
  logTrace("CHIPCallbackEventMonitorLevel0 out of callbacks. Exiting thread");
  pthread_exit(0);
}

void CHIPStaleEventMonitorLevel0::monitor() {
  logTrace("CHIPStaleEventMonitorLevel0::monitor()");
  // Stop is false and I have more events

  while (!Stop) {
    usleep(20000);
    auto LzBackend = (CHIPBackendLevel0 *)Backend;
    logTrace("num Events {} num queues() {}", Backend->Events.size(),
             LzBackend->EventCommandListMap.size());
    std::vector<CHIPEvent *> EventsToDelete;
    std::vector<ze_command_list_handle_t> CommandListsToDelete;

    std::lock_guard<std::mutex> AllEventsLock(Backend->EventsMtx);
    std::lock_guard<std::mutex> AllCommandListsLock(
        ((CHIPBackendLevel0 *)Backend)->CommandListsMtx);

    auto EventCommandListMap =
        &((CHIPBackendLevel0 *)Backend)->EventCommandListMap;

    for (int i = 0; i < Backend->Events.size(); i++) {
      CHIPEvent *ChipEvent = Backend->Events[i];

      assert(ChipEvent);
      auto E = (CHIPEventLevel0 *)ChipEvent;

      // do not change refcount for user events
      if (E->updateFinishStatus(false) && E->EventPool) {
        E->decreaseRefCount("Event became ready");
        E->releaseDependencies();
      }

      // delete the event if refcount reached 0
      if (E->getCHIPRefc() == 0) {
        auto Found =
            std::find(Backend->Events.begin(), Backend->Events.end(), E);
        if (Found == Backend->Events.end())
          CHIPERR_LOG_AND_THROW("StaleEventMonitor is trying to destroy an "
                                "event which is already "
                                "removed from backend event list",
                                hipErrorTbd);
        Backend->Events.erase(Found);

        if (E->EventPool)
          E->EventPool->returnSlot(E->EventPoolIndex);

        // Check if this event is associated with a CommandList
        bool CommandListFound = EventCommandListMap->count(E);
        if (CommandListFound) {
          auto CommandList = (*EventCommandListMap)[E];
          EventCommandListMap->erase(E);
          auto Status = zeCommandListDestroy(CommandList);
          CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
        }

        // Add the most course-grain lock here in case event destructor is not
        // thread-safe
        std::lock_guard Lock(Backend->Mtx);
        delete E;
      }

    } // done collecting events to delete

    if (Stop && !Backend->Events.size() && !EventCommandListMap->size()) {
      logTrace(
          "CHIPStaleEventMonitorLevel0 stop was called and all events have "
          "been cleared");
      pthread_exit(0);
    }

  } // endless loop
}
// End CHIPEventMonitorLevel0

// CHIPKernelLevelZero
// ***********************************************************************

ze_kernel_handle_t CHIPKernelLevel0::get() { return ZeKernel_; }

CHIPKernelLevel0::CHIPKernelLevel0(ze_kernel_handle_t ZeKernel,
                                   CHIPDeviceLevel0 *Dev, std::string HostFName,
                                   OCLFuncInfo *FuncInfo,
                                   CHIPModuleLevel0 *Parent)
    : CHIPKernel(HostFName, FuncInfo), ZeKernel_(ZeKernel), Module(Parent),
      Device(Dev), Name_(HostFName) {
  logTrace("CHIPKernelLevel0 constructor via ze_kernel_handle");

  ze_kernel_properties_t Props = {ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, 0};

  ze_result_t Status = zeKernelGetProperties(ZeKernel, &Props);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  PrivateSize_ = Props.privateMemSize;
  StaticLocalSize_ = Props.localMemSize;

  // TODO there doesn't seem to exist a way to get these from L0 API
  MaxDynamicLocalSize_ =
      Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  MaxWorkGroupSize_ = Device->getAttr(hipDeviceAttributeMaxThreadsPerBlock);
}
// End CHIPKernelLevelZero

hipError_t CHIPKernelLevel0::getAttributes(hipFuncAttributes *Attr) {

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
}

// CHIPQueueLevelZero
// ***********************************************************************

void CHIPQueueLevel0::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  CHIPCallbackData *Callbackdata =
      Backend->createCallbackData(Callback, UserData, this);

  {
    std::lock_guard<std::mutex> Lock(Backend->CallbackQueueMtx);
    Backend->CallbackQueue.push(Callbackdata);
  }

  // Setup event handling on the CPU side
  {
    std::lock_guard<std::mutex> Lock(Mtx);
    auto Monitor = ((CHIPBackendLevel0 *)Backend)->CallbackEventMonitor;
    if (!Monitor) {
      auto Evm = new CHIPCallbackEventMonitorLevel0();
      Evm->start();
      Monitor = Evm;
    }
  }

  return;
}

CHIPEventLevel0 *CHIPQueueLevel0::getLastEvent() {
  return (CHIPEventLevel0 *)LastEvent_;
}

ze_command_list_handle_t CHIPQueueLevel0::getCmdListCopy() {
#ifdef L0_IMM_QUEUES
  return ZeCmdListCopyImm_;
#else
  ze_command_list_handle_t CommandList;
  auto Status = zeCommandListCreate(ZeCtx_, ZeDev_, &CommandListMemoryDesc_,
                                    &CommandList);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return CommandList;
#endif
}

ze_command_list_handle_t CHIPQueueLevel0::getCmdListCompute() {
#ifdef L0_IMM_QUEUES
  return ZeCmdListComputeImm_;
#else
  ze_command_list_handle_t ZeCmdList;
  auto Status =
      zeCommandListCreate(ZeCtx_, ZeDev_, &CommandListComputeDesc_, &ZeCmdList);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return ZeCmdList;
#endif
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags)
    : CHIPQueueLevel0(ChipDev, Flags, 0) {}
CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev)
    : CHIPQueueLevel0(ChipDev, 0, 0) {}

void CHIPQueueLevel0::initializeCopyListImm() {
  ze_command_queue_desc_t CommandQueueCopyDesc = getNextComputeQueueDesc();

  // Create an immediate command list for copy engine
  auto Status = zeCommandListCreateImmediate(
      ZeCtx_, ZeDev_, &CommandQueueCopyDesc, &ZeCmdListCopyImm_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  logTrace("Created an immediate copy list");
}
void CHIPQueueLevel0::initializeComputeListImm() {

  ze_command_queue_desc_t CommandQueueComputeDesc = getNextComputeQueueDesc();

  // Create an immediate command list
  auto Status = zeCommandListCreateImmediate(
      ZeCtx_, ZeDev_, &CommandQueueComputeDesc, &ZeCmdListComputeImm_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  logTrace("Created an immediate compute list");
}

void CHIPQueueLevel0::initializeQueueGroupProperties() {

  auto ChipContextLz = (CHIPContextLevel0 *)ChipContext_;
  SharedBuf_ =
      ChipContextLz->allocateImpl(32, 8, hipMemoryType::hipMemoryTypeUnified);

  // Initialize the uint64_t part as 0
  *(uint64_t *)this->SharedBuf_ = 0;

  // Discover the number of command queues
  uint32_t CmdqueueGroupCount = 0;
  auto Status = zeDeviceGetCommandQueueGroupProperties(
      ZeDev_, &CmdqueueGroupCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("CommandGroups found: {}", CmdqueueGroupCount);

  // Create a vector of command queue properties, fill it
  ze_command_queue_group_properties_t *CmdqueueGroupProperties =
      (ze_command_queue_group_properties_t *)malloc(
          CmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
  for (int i = 0; i < CmdqueueGroupCount; i++) {
    CmdqueueGroupProperties[i] = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, // stype
        nullptr,                                          // pNext
        0,                                                // flags
        0, // maxMemoryFillPatternSize
        0  // numQueues
    };
  }
  Status = zeDeviceGetCommandQueueGroupProperties(ZeDev_, &CmdqueueGroupCount,
                                                  CmdqueueGroupProperties);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  this->MaxMemoryFillPatternSize =
      CmdqueueGroupProperties[0].maxMemoryFillPatternSize;

  // Find a command queue type that support compute
  for (uint32_t i = 0; i < CmdqueueGroupCount; ++i) {
    if (CmdqueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ComputeQueueGroupOrdinal_ = i;
      ComputeQueueProperties_ = CmdqueueGroupProperties[i];
      logTrace("Found compute command group");
      break;
    }
  }

  // Find a command queue type that support copy
  for (uint32_t i = 0; i < CmdqueueGroupCount; ++i) {
    if (CmdqueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
      CopyQueueGroupOrdinal_ = i;
      CopyQueueProperties_ = CmdqueueGroupProperties[i];
      logTrace("Found memory command group");
      break;
    }
  }

  // initialize compute and copy list descriptors
  CommandListComputeDesc_ = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
      nullptr,
      ComputeQueueGroupOrdinal_,
      0 /* CommandListFlags */,
  };

  CommandListMemoryDesc_ = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
      nullptr,
      CopyQueueGroupOrdinal_,
      0 /* CommandListFlags */,
  };
}

ze_command_queue_desc_t CHIPQueueLevel0::getNextComputeQueueDesc() {
  ze_command_queue_desc_t CommandQueueComputeDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      ComputeQueueGroupOrdinal_,
      NextComputeQueueIndex_, // index
      0,                      // flags
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  auto MaxQueues = ComputeQueueProperties_.numQueues;
  NextCopyQueueIndex_ = (NextCopyQueueIndex_ + 1) % MaxQueues;

  return CommandQueueComputeDesc;
}

ze_command_queue_desc_t CHIPQueueLevel0::getNextCopyQueueDesc() {
  ze_command_queue_desc_t CommandQueueCopyDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      CopyQueueGroupOrdinal_,
      NextComputeQueueIndex_, // index
      0,                      // flags
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  auto MaxQueues = CopyQueueProperties_.numQueues;
  NextCopyQueueIndex_ = (NextCopyQueueIndex_ + 1) % MaxQueues;

  return CommandQueueCopyDesc;
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags,
                                 int Priority)
    : CHIPQueue(ChipDev, Flags, Priority) {
  ze_result_t Status;
  auto ChipDevLz = ChipDev;
  auto Ctx = ChipDevLz->getContext();
  auto ChipContextLz = (CHIPContextLevel0 *)Ctx;

  ZeCtx_ = ChipContextLz->get();
  ZeDev_ = ChipDevLz->get();

  logTrace("CHIPQueueLevel0 constructor called via Flags and Priority");
  initializeQueueGroupProperties();

  ze_command_queue_desc_t QueueDescriptor = getNextComputeQueueDesc();
  Status = zeCommandQueueCreate(ZeCtx_, ZeDev_, &QueueDescriptor, &ZeCmdQ_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

#ifdef L0_IMM_QUEUES
  initializeComputeListImm();
  initializeCopyListImm();
#endif
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 ze_command_queue_handle_t ZeCmdQ)
    : CHIPQueue(ChipDev, 0, 0) {
  auto ChipDevLz = ChipDev;
  auto Ctx = ChipDevLz->getContext();
  auto ChipContextLz = (CHIPContextLevel0 *)Ctx;

  ZeCtx_ = ChipContextLz->get();
  ZeDev_ = ChipDevLz->get();

  initializeQueueGroupProperties();

  ZeCmdQ_ = ZeCmdQ;

#ifdef L0_IMM_QUEUES
  initializeComputeListImm();
  initializeCopyListImm();
#endif
}

CHIPEvent *CHIPQueueLevel0::launchImpl(CHIPExecItem *ExecItem) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *LaunchEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  LaunchEvent->Msg = "launch";

  CHIPKernelLevel0 *ChipKernel = (CHIPKernelLevel0 *)ExecItem->getKernel();
  ze_kernel_handle_t KernelZe = ChipKernel->get();
  logTrace("Launching Kernel {}", ChipKernel->getName());

  ze_result_t Status =
      zeKernelSetGroupSize(KernelZe, ExecItem->getBlock().x,
                           ExecItem->getBlock().y, ExecItem->getBlock().z);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  ExecItem->setupAllArgs();
  auto X = ExecItem->getGrid().x;
  auto Y = ExecItem->getGrid().y;
  auto Z = ExecItem->getGrid().z;
  ze_group_count_t LaunchArgs = {X, Y, Z};
  auto CommandList = getCmdListCompute();
  Status = zeCommandListAppendLaunchKernel(CommandList, KernelZe, &LaunchArgs,
                                           LaunchEvent->get(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  auto StatusReadyCheck = zeEventQueryStatus(LaunchEvent->peek());
  if (StatusReadyCheck != ZE_RESULT_NOT_READY) {
    logCritical("KernelLaunch event immediately ready!");
  }
  executeCommandList(CommandList);

  return LaunchEvent;
}

CHIPEvent *CHIPQueueLevel0::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memFill";

  if (PatternSize >= MaxMemoryFillPatternSize) {
    logCritical("PatternSize: {} Max: {}", PatternSize,
                MaxMemoryFillPatternSize);
    CHIPERR_LOG_AND_THROW("MemFill PatternSize exceeds the max for this queue",
                          hipErrorTbd);
  }

  if (std::ceil(log2(PatternSize)) != std::floor(log2(PatternSize))) {
    logCritical("PatternSize: {} Max: {}", PatternSize,
                MaxMemoryFillPatternSize);
    CHIPERR_LOG_AND_THROW("MemFill PatternSize is not a power of 2",
                          hipErrorTbd);
  }
  auto CommandList = getCmdListCopy();
  ze_result_t Status = zeCommandListAppendMemoryFill(
      CommandList, Dst, Pattern, PatternSize, Size, Ev->get(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return Ev;
};

CHIPEvent *CHIPQueueLevel0::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  return memCopy3DAsyncImpl(Dst, Dpitch, 0, Src, Spitch, 0, Width, Height, 0);
};

CHIPEvent *CHIPQueueLevel0::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memCopy3DAsync";

  ze_copy_region_t DstRegion;
  DstRegion.originX = 0;
  DstRegion.originY = 0;
  DstRegion.originZ = 0;
  DstRegion.width = Width;
  DstRegion.height = Height;
  DstRegion.depth = Depth;
  ze_copy_region_t SrcRegion;
  SrcRegion.originX = 0;
  SrcRegion.originY = 0;
  SrcRegion.originZ = 0;
  SrcRegion.width = Width;
  SrcRegion.height = Height;
  SrcRegion.depth = Depth;
  auto CommandList = getCmdListCopy();
  ze_result_t Status = zeCommandListAppendMemoryCopyRegion(
      CommandList, Dst, &DstRegion, Dpitch, Dspitch, Src, &SrcRegion, Spitch,
      Sspitch, Ev->get(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return Ev;
};

// Memory copy to texture object, i.e. image
CHIPEvent *CHIPQueueLevel0::memCopyToImage(ze_image_handle_t Image,
                                           const void *Src,
                                           const CHIPRegionDesc &SrcRegion) {
  logTrace("CHIPQueueLevel0::memCopyToImage");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memCopyToImage";

  if (!SrcRegion.isPitched()) {
    auto CommandList = getCmdListCopy();
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, Src, 0, Ev->get(), 0, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    executeCommandList(CommandList);

    return Ev;
  }

  // Copy image data row by row since level zero does not have pitched copy.
  CHIPASSERT(SrcRegion.getNumDims() == 2 &&
             "UNIMPLEMENTED: 3D pitched image copy.");
  const char *SrcRow = (const char *)Src;
  for (size_t Row = 0; Row < SrcRegion.Size[1]; Row++) {
    bool LastRow = Row == SrcRegion.Size[1] - 1;
    ze_image_region_t DstZeRegion{};
    DstZeRegion.originX = 0;
    DstZeRegion.originY = Row;
    DstZeRegion.originZ = 0;
    DstZeRegion.width = SrcRegion.Size[0];
    DstZeRegion.height = 1;
    DstZeRegion.depth = 1;
    auto CommandList = getCmdListCopy();
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, SrcRow, &DstZeRegion, LastRow ? Ev->get() : nullptr,
        0, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    executeCommandList(CommandList);
    SrcRow += SrcRegion.Pitch[0];
  }
  return Ev;
};

hipError_t CHIPQueueLevel0::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueLevel0::getBackendHandles");
  if (*NumHandles < 4) {
    logError("getBackendHandles requires space for 4 handles");
    return hipErrorInvalidValue;
  }
  *NumHandles = 4;

  // Get queue handler
  NativeInfo[3] = (uintptr_t)ZeCmdQ_;

  // Get context handler
  CHIPContextLevel0 *Ctx = (CHIPContextLevel0 *)ChipContext_;
  NativeInfo[2] = (uintptr_t)Ctx->get();

  // Get device handler
  CHIPDeviceLevel0 *Dev = (CHIPDeviceLevel0 *)ChipDevice_;
  NativeInfo[1] = (uintptr_t)Dev->get();

  // Get driver handler
  NativeInfo[0] = (uintptr_t)Ctx->ZeDriver;
  return hipSuccess;
}

CHIPEvent *CHIPQueueLevel0::enqueueMarkerImpl() {
  CHIPEventLevel0 *MarkerEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);

  MarkerEvent->Msg = "marker";
  auto CommandList = getCmdListCompute();
  auto Status = zeCommandListAppendSignalEvent(CommandList, MarkerEvent->get());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return MarkerEvent;
}

CHIPEvent *
CHIPQueueLevel0::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
  // Create an event, refc=2, add it to EventList
  CHIPEventLevel0 *EventToSignal =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);
  EventToSignal->Msg = "barrier";
  size_t NumEventsToWaitFor = 0;

  NumEventsToWaitFor = EventsToWaitFor ? EventsToWaitFor->size() : 0;

  ze_event_handle_t *EventHandles = nullptr;
  ze_event_handle_t SignalEventHandle = nullptr;

  SignalEventHandle = ((CHIPEventLevel0 *)(EventToSignal))->get();

  if (NumEventsToWaitFor > 0) {
    EventHandles = new ze_event_handle_t[NumEventsToWaitFor];
    for (int i = 0; i < NumEventsToWaitFor; i++) {
      CHIPEventLevel0 *ChipEventLz = (CHIPEventLevel0 *)(*EventsToWaitFor)[i];
      CHIPASSERT(ChipEventLz);
      EventHandles[i] = ChipEventLz->get();
      EventToSignal->addDependency(ChipEventLz);
    }
  } // done gather Event_ handles to wait on

  // TODO Should this be memory or compute?
  auto CommandList = getCmdListCompute();
  auto Status = zeCommandListAppendBarrier(CommandList, SignalEventHandle,
                                           NumEventsToWaitFor, EventHandles);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  if (EventHandles)
    delete[] EventHandles;

  return EventToSignal;
}

CHIPEvent *CHIPQueueLevel0::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  logTrace("CHIPQueueLevel0::memCopyAsync");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *MemCopyEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);

  ze_result_t Status;
  CHIPASSERT(MemCopyEvent->peek());
  auto CommandList = getCmdListCopy();
  Status = zeCommandListAppendMemoryCopy(CommandList, Dst, Src, Size,
                                         MemCopyEvent->get(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  executeCommandList(CommandList);

  return MemCopyEvent;
}

void CHIPQueueLevel0::finish() {
  // The finish Event_ that denotes the finish of current command list items
  pthread_yield();
  // Using zeCommandQueueSynchronize() for ensuring the device printf
  // buffers get flushed.
  zeCommandQueueSynchronize(ZeCmdQ_, UINT64_MAX);

  return;
}

void CHIPQueueLevel0::executeCommandList(ze_command_list_handle_t CommandList) {
#ifdef L0_IMM_QUEUES
#else

  auto LastCmdListEvent =
      ((CHIPBackendLevel0 *)Backend)->createCHIPEvent(ChipContext_);
  LastCmdListEvent->Msg = "CmdListFinishTracker";

  ze_result_t Status;

  {
    std::lock_guard<std::mutex> Lock(
        ((CHIPBackendLevel0 *)Backend)->CommandListsMtx);

    // Associate this event with the command list. Once the events are signaled,
    // CHIPEventMonitorLevel0 will destroy the command list

    ((CHIPBackendLevel0 *)Backend)
        ->EventCommandListMap[(CHIPEventLevel0 *)LastCmdListEvent] =
        CommandList;

    Status = zeCommandListAppendBarrier(CommandList, LastCmdListEvent->get(), 0,
                                        nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

    Status = zeCommandListClose(CommandList);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    Status =
        zeCommandQueueExecuteCommandLists(ZeCmdQ_, 1, &CommandList, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  LastCmdListEvent->track();
#endif
};

// End CHIPQueueLevelZero

// EventPool
// ***********************************************************************
LZEventPool::LZEventPool(CHIPContextLevel0 *Ctx, unsigned int Size)
    : Ctx_(Ctx), Size_(Size) {

  unsigned int PoolFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  // if (!flags.isDisableTiming())
  //   pool_flags = pool_flags | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      Size_                              // count
  };

  ze_result_t Status =
      zeEventPoolCreate(Ctx_->get(), &EventPoolDesc, 0, nullptr, &EventPool_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ pool creation fail! ");

  for (int i = 0; i < Size_; i++) {
    CHIPEventFlags Flags;
    Events_.push_back(new CHIPEventLevel0(Ctx_, this, i, Flags));
    FreeSlots_.push(i);
  }
};

LZEventPool::~LZEventPool() {
  for (int i = 0; i < Size_; i++) {
    delete Events_[i];
  }

  auto Status = zeEventPoolDestroy(EventPool_);
  // '~CHIPEventLevel0' has a non-throwing exception specification
  assert(Status == ZE_RESULT_SUCCESS);
};

CHIPEventLevel0 *LZEventPool::getEvent() {
  int PoolIndex = getFreeSlot();
  if (PoolIndex == -1)
    return nullptr;
  auto Event = Events_[PoolIndex];

  // reset event
  auto Status = zeEventHostReset(Event->get());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return Event;
};

int LZEventPool::getFreeSlot() {
  if (FreeSlots_.size() == 0)
    return -1;

  auto Slot = FreeSlots_.top();
  FreeSlots_.pop();

  return Slot;
}

void LZEventPool::returnSlot(int Slot) {
  FreeSlots_.push(Slot);
  return;
}

// End EventPool

// CHIPBackendLevel0
// ***********************************************************************

CHIPEventLevel0 *CHIPBackendLevel0::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  std::lock_guard Lock(Backend->Mtx);
  CHIPEventLevel0 *Event;
  if (UserEvent) {
    Event = new CHIPEventLevel0((CHIPContextLevel0 *)ChipCtx, Flags);
    Event->increaseRefCount("hipEventCreate");
    Event->track();
  } else {
    auto ZeCtx = (CHIPContextLevel0 *)ChipCtx;
    Event = ZeCtx->getEventFromPool();
  }

  return Event;
}

void CHIPBackendLevel0::uninitialize() {
  logDebug("CHIPBackend::uninitialize()");

  for (auto Q : Backend->getQueues())
    Q->updateLastEvent(nullptr);

  if (CallbackEventMonitor)
    CallbackEventMonitor->join();

  StaleEventMonitor->Stop = true;
  StaleEventMonitor->join();

  logWarn("Remaining {} events that haven't been collected:",
          Backend->Events.size());
  for (auto E : Backend->Events)
    logWarn("{} status= {} refc={}", E->Msg, E->getEventStatusStr(),
            E->getCHIPRefc());

  logWarn("Remaining {} command lists that haven't been collected:",
          ((CHIPBackendLevel0 *)Backend)->EventCommandListMap.size());
}
std::string CHIPBackendLevel0::getDefaultJitFlags() {
  return std::string(
      "-cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi");
}

void CHIPBackendLevel0::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendLevel0 Initialize");
  ze_result_t Status;
  Status = zeInit(0);
  if (Status != ZE_RESULT_SUCCESS) {
    logError("zeInit failed ");
    std::abort();
  }

  bool AnyDeviceType = false;
  ze_device_type_t ZeDeviceType;
  if (!CHIPDeviceTypeStr.compare("gpu")) {
    ZeDeviceType = ZE_DEVICE_TYPE_GPU;
  } else if (!CHIPDeviceTypeStr.compare("fpga")) {
    ZeDeviceType = ZE_DEVICE_TYPE_FPGA;
  } else if (!CHIPDeviceTypeStr.compare("default")) {
    // For 'default' pick all devices of any type.
    AnyDeviceType = true;
  } else {
    CHIPERR_LOG_AND_THROW("CHIP_DEVICE_TYPE must be either gpu or fpga",
                          hipErrorInitializationError);
  }
  int PlatformIdx = std::atoi(CHIPPlatformStr.c_str());
  std::vector<ze_driver_handle_t> ZeDrivers;
  std::vector<ze_device_handle_t> ZeDevices;

  // Get number of drivers
  uint32_t DriverCount = 0, DeviceCount = 0;
  Status = zeDriverGet(&DriverCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("Found Level0 Drivers: {}", DriverCount);
  // Resize and fill ZeDriver vector with drivers
  ZeDrivers.resize(DriverCount);
  Status = zeDriverGet(&DriverCount, ZeDrivers.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  // TODO Allow for multilpe platforms(drivers)
  // TODO Check platform ID is not the same as OpenCL. You can have
  // two OCL platforms but only one level0 driver
  ze_driver_handle_t ZeDriver = ZeDrivers[PlatformIdx];

  assert(ZeDriver != nullptr);
  // Load devices to device vector
  Status = zeDeviceGet(ZeDriver, &DeviceCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  ZeDevices.resize(DeviceCount);
  Status = zeDeviceGet(ZeDriver, &DeviceCount, ZeDevices.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  const ze_context_desc_t CtxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};

  ze_context_handle_t ZeCtx;
  Status = zeContextCreateEx(ZeDriver, &CtxDesc, DeviceCount, ZeDevices.data(),
                             &ZeCtx);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  CHIPContextLevel0 *ChipL0Ctx = new CHIPContextLevel0(ZeDriver, ZeCtx);
  Backend->addContext(ChipL0Ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of CHIPDevice
  for (int i = 0; i < DeviceCount; i++) {
    auto Dev = ZeDevices[i];
    ze_device_properties_t DeviceProperties{};
    DeviceProperties.pNext = nullptr;
    DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    auto Status = zeDeviceGetProperties(Dev, &DeviceProperties);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    if (AnyDeviceType || ZeDeviceType == DeviceProperties.type) {
      CHIPDeviceLevel0 *ChipL0Dev =
          new CHIPDeviceLevel0(std::move(Dev), ChipL0Ctx, i);
      ChipL0Dev->populateDeviceProperties();
      ChipL0Ctx->addDevice(ChipL0Dev);

      ChipL0Dev->createQueueAndRegister((int)0, (int)0);

      Backend->addDevice(ChipL0Dev);
      break; // For now don't add more than one device
    }
  } // End adding CHIPDevices

  StaleEventMonitor =
      (CHIPStaleEventMonitorLevel0 *)Backend->createStaleEventMonitor();
}

void CHIPBackendLevel0::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendLevel0 InitializeNative");

  ze_driver_handle_t Drv = (ze_driver_handle_t)NativeHandles[0];
  ze_device_handle_t Dev = (ze_device_handle_t)NativeHandles[1];
  ze_context_handle_t Ctx = (ze_context_handle_t)NativeHandles[2];

  CHIPContextLevel0 *ChipCtx = new CHIPContextLevel0(Drv, Ctx);
  addContext(ChipCtx);

  CHIPDeviceLevel0 *ChipDev = new CHIPDeviceLevel0(&Dev, ChipCtx, 0);
  ChipCtx->addDevice(ChipDev);
  addDevice(ChipDev);

  ChipDev->createQueueAndRegister(NativeHandles, NumHandles);

  StaleEventMonitor =
      (CHIPStaleEventMonitorLevel0 *)Backend->createStaleEventMonitor();

  setActiveDevice(ChipDev);
}

hipEvent_t CHIPBackendLevel0::getHipEvent(void *NativeEvent) {
  ze_event_handle_t E = (ze_event_handle_t)NativeEvent;
  CHIPEventLevel0 *NewEvent =
      new CHIPEventLevel0((CHIPContextLevel0 *)ActiveCtx_, E);
  NewEvent->increaseRefCount("getHipEvent");
  return NewEvent;
}

void *CHIPBackendLevel0::getNativeEvent(hipEvent_t HipEvent) {
  CHIPEventLevel0 *E = (CHIPEventLevel0 *)HipEvent;
  if (!E->isRecordingOrRecorded())
    return nullptr;
  // TODO should we retain here?
  return (void *)E->get();
}

// CHIPContextLevelZero
// ***********************************************************************

void *CHIPContextLevel0::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemTy,
                                      CHIPHostAllocFlags Flags) {

  void *Ptr = 0;
  logWarn("Ignoring alignment. Using hardcoded value 0x1000");
  Alignment = 0x1000; // TODO Where/why

  if (MemTy == hipMemoryType::hipMemoryTypeUnified) {
    logWarn("Usigned zeMallocHost instead of zeMallocShared due to outstanding "
            "bug");
    MemTy = hipMemoryType::hipMemoryTypeHost;
  }

  ze_device_mem_alloc_flags_t DeviceFlags =
      ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;

  ze_device_mem_alloc_desc_t DmaDesc{
      /* DmaDesc.stype   = */ ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
      /* DmaDesc.pNext   = */ nullptr,
      /* DmaDesc.flags   = */ DeviceFlags,
      /* DmaDesc.ordinal = */ 0,
  };
  ze_host_mem_alloc_flags_t HostFlags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  if (Flags.isWriteCombined())
    HostFlags += ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
  ze_host_mem_alloc_desc_t HmaDesc{
      /* HmaDesc.stype = */ ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
      /* HmaDesc.pNext = */ nullptr,
      /* HmaDesc.flags = */ HostFlags,
  };
  if (MemTy == hipMemoryType::hipMemoryTypeUnified) {

    // TODO Check if devices support cross-device sharing?
    // ze_device_handle_t ZeDev = ((CHIPDeviceLevel0
    // *)getDevices()[0])->get();
    ze_device_handle_t ZeDev = nullptr; // Do not associate allocation

    ze_result_t Status = zeMemAllocShared(ZeCtx, &DmaDesc, &HmaDesc, Size,
                                          Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", Status);

    return Ptr;
  } else if (MemTy == hipMemoryType::hipMemoryTypeDevice) {
    auto ChipDev = (CHIPDeviceLevel0 *)Backend->getActiveDevice();
    ze_device_handle_t ZeDev = ChipDev->get();

    ze_result_t Status =
        zeMemAllocDevice(ZeCtx, &DmaDesc, Size, Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    return Ptr;
  } else if (MemTy == hipMemoryType::hipMemoryTypeHost) {
    // TODO Check if devices support cross-device sharing?
    ze_result_t Status = zeMemAllocHost(ZeCtx, &HmaDesc, Size, Alignment, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", Status);

    return Ptr;
  }
  CHIPERR_LOG_AND_THROW("Failed to allocate memory", hipErrorMemoryAllocation);
}

// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t *ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : CHIPDevice(ChipCtx, Idx), ZeDev_(*ZeDev), ZeCtx_(ChipCtx->get()),
      ZeDeviceProps_() {
  ZeDeviceProps_.pNext = nullptr;
  assert(Ctx_ != nullptr);
}
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t &&ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : CHIPDevice(ChipCtx, Idx), ZeDev_(ZeDev), ZeCtx_(ChipCtx->get()),
      ZeDeviceProps_() {
  ZeDeviceProps_.pNext = nullptr;
  assert(Ctx_ != nullptr);
}

void CHIPDeviceLevel0::resetImpl() { UNIMPLEMENTED(); }

void CHIPDeviceLevel0::populateDevicePropertiesImpl() {
  ze_result_t Status = ZE_RESULT_SUCCESS;

  // Initialize members used as input for zeDeviceGet*Properties() calls.
  ZeDeviceProps_.pNext = nullptr;
  ZeDeviceProps_.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ze_device_memory_properties_t DeviceMemProps;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  DeviceMemProps.pNext = nullptr;
  ze_device_compute_properties_t DeviceComputeProps;
  DeviceComputeProps.pNext = nullptr;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ze_device_cache_properties_t DeviceCacheProps;
  DeviceCacheProps.pNext = nullptr;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
  ze_device_module_properties_t DeviceModuleProps;
  DeviceModuleProps.pNext = nullptr;
  DeviceModuleProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  ze_device_image_properties_t DeviceImageProps;
  DeviceImageProps.pNext = nullptr;
  DeviceImageProps.stype = ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;

  // Query device properties
  Status = zeDeviceGetProperties(ZeDev_, &ZeDeviceProps_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  this->MaxMallocSize_ = ZeDeviceProps_.maxMemAllocSize;

  // Query device memory properties
  uint32_t Count = 1;
  Status = zeDeviceGetMemoryProperties(ZeDev_, &Count, &DeviceMemProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device computation properties
  Status = zeDeviceGetComputeProperties(ZeDev_, &DeviceComputeProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device cache properties
  Count = 1;
  Status = zeDeviceGetCacheProperties(ZeDev_, &Count, &DeviceCacheProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device module properties
  Status = zeDeviceGetModuleProperties(ZeDev_, &DeviceModuleProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device image properties
  Status = zeDeviceGetImageProperties(ZeDev_, &DeviceImageProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Copy device name
  if (255 < ZE_MAX_DEVICE_NAME) {
    strncpy(HipDeviceProps_.name, HipDeviceProps_.name, 255);
    HipDeviceProps_.name[255] = 0;
  } else {
    strncpy(HipDeviceProps_.name, HipDeviceProps_.name, ZE_MAX_DEVICE_NAME);
    HipDeviceProps_.name[ZE_MAX_DEVICE_NAME - 1] = 0;
  }

  // Get total device memory
  HipDeviceProps_.totalGlobalMem = DeviceMemProps.totalSize;

  HipDeviceProps_.sharedMemPerBlock = DeviceComputeProps.maxSharedLocalMemory;
  //??? Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  HipDeviceProps_.maxThreadsPerBlock = DeviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  HipDeviceProps_.maxThreadsDim[0] = DeviceComputeProps.maxGroupSizeX;
  HipDeviceProps_.maxThreadsDim[1] = DeviceComputeProps.maxGroupSizeY;
  HipDeviceProps_.maxThreadsDim[2] = DeviceComputeProps.maxGroupSizeZ;

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ZeDeviceProps_.coreClockRate; // deviceMemoryProps.maxClockRate;
  // Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ZeDeviceProps_.numEUsPerSubslice *
      ZeDeviceProps_.numSlices; // DeviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize = DeviceCacheProps.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem = DeviceMemProps.totalSize;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // as per gen architecture doc
  HipDeviceProps_.regsPerBlock = 4096;

  HipDeviceProps_.warpSize =
      DeviceComputeProps.subGroupSizes[DeviceComputeProps.numSubGroupSizes - 1];

  // Replicate from OpenCL implementation

  // HIP and LZ uses int and uint32_t, respectively, for storing the
  // group count. Clamp the group count to INT_MAX to avoid 2^31+ size
  // being interpreted as negative number.
  constexpr unsigned IntMax = std::numeric_limits<int>::max();
  HipDeviceProps_.maxGridSize[0] =
      std::min(DeviceComputeProps.maxGroupCountX, IntMax);
  HipDeviceProps_.maxGridSize[1] =
      std::min(DeviceComputeProps.maxGroupCountY, IntMax);
  HipDeviceProps_.maxGridSize[2] =
      std::min(DeviceComputeProps.maxGroupCountZ, IntMax);
  HipDeviceProps_.memoryClockRate = DeviceMemProps.maxClockRate;
  HipDeviceProps_.memoryBusWidth = DeviceMemProps.maxBusWidth;
  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor =
      ZeDeviceProps_.numEUsPerSubslice * ZeDeviceProps_.numThreadsPerEU; //  10;

  HipDeviceProps_.computeMode = hipComputeModeDefault;
  HipDeviceProps_.arch = {};

  HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  HipDeviceProps_.arch.hasSharedInt32Atomics = 1;

  HipDeviceProps_.arch.hasGlobalInt64Atomics =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;
  HipDeviceProps_.arch.hasSharedInt64Atomics =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;

  HipDeviceProps_.arch.hasDoubles =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_FP64) ? 1 : 0;

  HipDeviceProps_.clockInstructionRate = ZeDeviceProps_.coreClockRate;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.gcnArch = 0;
  HipDeviceProps_.integrated =
      (ZeDeviceProps_.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor =
      DeviceComputeProps.maxSharedLocalMemory;

  // Clamp texture dimensions to [0, INT_MAX] because the return value
  // of hipDeviceGetAttribute() is int type.
  auto MaxDim0 = clampToInt(DeviceImageProps.maxImageDims1D);
  auto MaxDim1 = clampToInt(DeviceImageProps.maxImageDims2D);
  auto MaxDim2 = clampToInt(DeviceImageProps.maxImageDims3D);

  HipDeviceProps_.maxTexture1DLinear = MaxDim0;
  HipDeviceProps_.maxTexture1D = MaxDim0;
  HipDeviceProps_.maxTexture2D[0] = MaxDim0;
  HipDeviceProps_.maxTexture2D[1] = MaxDim1;
  HipDeviceProps_.maxTexture3D[0] = MaxDim0;
  HipDeviceProps_.maxTexture3D[1] = MaxDim1;
  HipDeviceProps_.maxTexture3D[2] = MaxDim2;

  // Level0 does not have alignment requirements for images that
  // clients should follow.
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;
}

CHIPQueue *CHIPDeviceLevel0::addQueueImpl(unsigned int Flags, int Priority) {
  CHIPQueueLevel0 *NewQ = new CHIPQueueLevel0(this, Flags, Priority);
  return NewQ;
}

CHIPQueue *CHIPDeviceLevel0::addQueueImpl(const uintptr_t *NativeHandles,
                                          int NumHandles) {
  ze_command_queue_handle_t CmdQ = (ze_command_queue_handle_t)NativeHandles[3];
  CHIPQueueLevel0 *NewQ;
  if (!CmdQ) {
    logWarn("initializeFromNative: native queue pointer is null. Creating a "
            "new queue");
    NewQ = new CHIPQueueLevel0(this, 0, 0);
  } else {
    NewQ = new CHIPQueueLevel0(this, CmdQ);
  }

  return NewQ;
}

ze_image_handle_t CHIPDeviceLevel0::allocateImage(unsigned int TextureType,
                                                  hipChannelFormatDesc Format,
                                                  bool NormalizedFloat,
                                                  size_t Width, size_t Height,
                                                  size_t Depth) {
  logTrace("CHIPContextLevel0::allocateImage()");
  auto ImageDesc = getImageDescription(TextureType, Format, NormalizedFloat,
                                       Width, Height, Depth);
  ze_image_handle_t ImageHandle{};

  ze_result_t Status = zeImageCreate(ZeCtx_, ZeDev_, &ImageDesc, &ImageHandle);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorMemoryAllocation);
  return ImageHandle;
}

CHIPTexture *CHIPDeviceLevel0::createTexture(
    const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
    const struct hipResourceViewDesc *PResViewDesc) {
  logTrace("CHIPDeviceLevel0::createTexture");

  bool NormalizedFloat = PTexDesc->readMode == hipReadModeNormalizedFloat;
  auto *Q = (CHIPQueueLevel0 *)getActiveQueue();

  ze_sampler_handle_t SamplerHandle =
      createSampler(this, PResDesc, PTexDesc, PResViewDesc);

  if (PResDesc->resType == hipResourceTypeArray) {
    hipArray *Array = PResDesc->res.array.array;
    // Checked in CHIPBindings already.
    CHIPASSERT(Array->data && "Invalid hipArray.");
    CHIPASSERT(!Array->isDrv && "Not supported/implemented yet.");
    size_t Width = Array->width;
    size_t Height = Array->height;
    size_t Depth = Array->depth;

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(Array->textureType, Array->desc, NormalizedFloat, Width,
                      Height, Depth));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    CHIPRegionDesc SrcRegion = CHIPRegionDesc::from(*Array);
    Q->memCopyToImage(ImageHandle, Array->data, SrcRegion);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  if (PResDesc->resType == hipResourceTypeLinear) {
    auto &Res = PResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(hipTextureType1D, Res.desc, NormalizedFloat, Width));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::get1DRegion(Width, TexelByteSize);
    Q->memCopyToImage(ImageHandle, Res.devPtr, SrcDesc);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  if (PResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = PResDesc->res.pitch2D;

    CHIPASSERT(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(hipTextureType2D, Res.desc, NormalizedFloat, Res.width,
                      Res.height));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::from(*PResDesc);
    Q->memCopyToImage(ImageHandle, Res.devPtr, SrcDesc);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  CHIPASSERT(false && "Unsupported/unimplemented texture resource type.");
  return nullptr;
}

// Other
// ***********************************************************************
std::string resultToString(ze_result_t Status) {
  switch (Status) {
  case ZE_RESULT_SUCCESS:
    return "ZE_RESULT_SUCCESS";
  case ZE_RESULT_NOT_READY:
    return "ZE_RESULT_NOT_READY";
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  case ZE_RESULT_ERROR_UNKNOWN:
    return "ZE_RESULT_ERROR_UNKNOWN";
  default:
    return "Unknown Error Code";
  }
}

// CHIPModuleLevel0
// ***********************************************************************
void CHIPModuleLevel0::compile(CHIPDevice *ChipDev) {
  logTrace("CHIPModuleLevel0.compile()");
  consumeSPIRV();
  ze_result_t Status;

  // Create module with global address aware
  std::string CompilerOptions = Backend->getJitFlags();
  ze_module_desc_t ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 IlSize_,
                                 FuncIL_,
                                 CompilerOptions.c_str(),
                                 nullptr};

  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)(ChipDev->getContext());
  CHIPDeviceLevel0 *LzDev = (CHIPDeviceLevel0 *)ChipDev;

  ze_device_handle_t ZeDev = LzDev->get();
  ze_context_handle_t ZeCtx = ChipCtxLz->get();

  ze_module_build_log_handle_t Log;
  auto BuildStatus =
      zeModuleCreate(ZeCtx, ZeDev, &ModuleDesc, &ZeModule_, &Log);
  if (BuildStatus != ZE_RESULT_SUCCESS) {
    size_t LogSize;
    Status = zeModuleBuildLogGetString(Log, &LogSize, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    char LogStr[LogSize];
    Status = zeModuleBuildLogGetString(Log, &LogSize, LogStr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logError("ZE Build Log: {}", std::string(LogStr).c_str());
  }
  CHIPERR_CHECK_LOG_AND_THROW(BuildStatus, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("LZ CREATE MODULE via calling zeModuleCreate {} ",
           resultToString(BuildStatus));
  // if (Status == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
  //  CHIPERR_LOG_AND_THROW("Module failed to JIT: " + std::string(log_str),
  //                        hipErrorUnknown);
  //}

  uint32_t KernelCount = 0;
  Status = zeModuleGetKernelNames(ZeModule_, &KernelCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("Found {} kernels in this module.", KernelCount);

  const char *KernelNames[KernelCount];
  Status = zeModuleGetKernelNames(ZeModule_, &KernelCount, KernelNames);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  for (auto &Kernel : KernelNames)
    logTrace("Kernel {}", Kernel);
  for (int i = 0; i < KernelCount; i++) {
    std::string HostFName = KernelNames[i];
    logTrace("Registering kernel {}", HostFName);

    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      // TODO: __syncthreads() gets turned into
      // Intel_Symbol_Table_Void_Program This is a call to OCML so it
      // shouldn't be turned into a CHIPKernel
      continue;
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in
      // OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }

    // Create kernel
    ze_kernel_handle_t ZeKernel;
    ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                   0, // flags
                                   HostFName.c_str()};
    Status = zeKernelCreate(ZeModule_, &KernelDesc, &ZeKernel);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logTrace("LZ KERNEL CREATION via calling zeKernelCreate {} ", Status);
    CHIPKernelLevel0 *ChipZeKernel =
        new CHIPKernelLevel0(ZeKernel, LzDev, HostFName, FuncInfo, this);
    addKernel(ChipZeKernel);
  }
}

void CHIPExecItem::setupAllArgs() {
  CHIPKernelLevel0 *Kernel = (CHIPKernelLevel0 *)ChipKernel_;

  OCLFuncInfo *FuncInfo = ChipKernel_->getFuncInfo();

  size_t NumLocals = 0;

  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].Space == OCLSpace::Local) {
      ++NumLocals;
    }
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  // Argument processing for the new HIP launch API.
  if (ArgsPointer_) {
    for (size_t InArgIdx = 0, OutArgIdx = 0;
         OutArgIdx < FuncInfo->ArgTypeInfo.size(); ++OutArgIdx, ++InArgIdx) {
      OCLArgTypeInfo &ArgTypeInfo = FuncInfo->ArgTypeInfo[OutArgIdx];

      // Handle direct texture object passing. When we see an image
      // type we know it's derived from a texture object argument
      if (ArgTypeInfo.Type == OCLType::Image) {
        auto *TexObj = *(CHIPTextureLevel0 **)ArgsPointer_[InArgIdx];

        // Set image argument.
        ze_image_handle_t ImageHandle = TexObj->getImage();
        logTrace("setImageArg {} size {}\n", OutArgIdx,
                 sizeof(ze_image_handle_t));
        ze_result_t Status = zeKernelSetArgumentValue(
            Kernel->get(), OutArgIdx, sizeof(ze_image_handle_t), &ImageHandle);
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

        // Set sampler argument.
        OutArgIdx++;
        ze_sampler_handle_t SamplerHandle = TexObj->getSampler();
        logTrace("setSamplerArg {} size {}\n", OutArgIdx,
                 sizeof(ze_sampler_handle_t));
        Status = zeKernelSetArgumentValue(Kernel->get(), OutArgIdx,
                                          sizeof(ze_sampler_handle_t),
                                          &SamplerHandle);
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
      } else {
        logTrace("setArg {} size {} addr {}\n", OutArgIdx, ArgTypeInfo.Size,
                 ArgsPointer_[InArgIdx]);
        ze_result_t Status = zeKernelSetArgumentValue(
            Kernel->get(), OutArgIdx, ArgTypeInfo.Size, ArgsPointer_[InArgIdx]);
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");
      }
    }
  } else {
    // Argument processing for the old HIP launch API.
    if ((OffsetSizes_.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
    }

    if (OffsetSizes_.size() == 0)
      return;

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
    for (size_t i = 0; i < OffsetSizes_.size(); ++i) {
      OCLArgTypeInfo &ArgTypeInfo = FuncInfo->ArgTypeInfo[i];
      logTrace("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetSizes_[i]), std::get<1>(OffsetSizes_[i]),
               (unsigned)ArgTypeInfo.Type, (unsigned)ArgTypeInfo.Space,
               ArgTypeInfo.Size);

      CHIPASSERT(ArgTypeInfo.Type != OCLType::Image &&
                 "UNIMPLEMENTED: texture object arguments for old HIP kernel "
                 "launch API.");

      if (ArgTypeInfo.Type == OCLType::Pointer) {
        // TODO: sync with ExecItem's solution
        assert(ArgTypeInfo.Size == sizeof(void *));
        assert(std::get<1>(OffsetSizes_[i]) == ArgTypeInfo.Size);
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offset = std::get<0>(OffsetSizes_[i]);
        const void *Value = (void *)(Start + Offset);
        logTrace("setArg SVM {} to {}\n", i, (void *)Value);
        ze_result_t Status =
            zeKernelSetArgumentValue(Kernel->get(), i, Size, Value);

        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
                 "{} ",
                 Status);
      } else {
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offset = std::get<0>(OffsetSizes_[i]);
        const void *Value = (void *)(Start + Offset);
        logTrace("setArg {} size {} offs {}\n", i, Size, Offset);
        ze_result_t Status =
            zeKernelSetArgumentValue(Kernel->get(), i, Size, Value);

        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
                 "{} ",
                 Status);
      }
    }
  }

  // Setup the kernel argument's value related to dynamically sized share
  // memory
  if (NumLocals == 1) {
    ze_result_t Status = zeKernelSetArgumentValue(
        Kernel->get(), FuncInfo->ArgTypeInfo.size() - 1, SharedMem_, nullptr);
    logTrace("LZ set dynamically sized share memory related argument via "
             "calling "
             "zeKernelSetArgumentValue {} ",
             Status);
  }

  return;
}
