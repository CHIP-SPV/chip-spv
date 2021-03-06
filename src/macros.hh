#ifndef MACROS_HH
#define MACROS_HH

#include "logging.hh"
#include "iostream"
#include "CHIPException.hh"

#ifdef CHIP_ERROR_ON_UNIMPL
#define UNIMPLEMENTED(x)                                                       \
  CHIPERR_LOG_AND_THROW("Called a function which is not implemented",          \
                        hipErrorNotSupported);
#else
#define UNIMPLEMENTED(x)                                                       \
  do {                                                                         \
    logWarn("{}: Called a function which is not implemented", __FUNCTION__);   \
    return x;                                                                  \
  } while (0)
#endif

#define RETURN(x)                                                              \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    Backend->TlsLastError = err;                                               \
    return err;                                                                \
  } while (0)

#define ERROR_IF(cond, err)                                                    \
  if (cond)                                                                    \
    do {                                                                       \
      logError("Error {} at {}:{} code {}", err, __FILE__, __LINE__, #cond);   \
      Backend->TlsLastError = err;                                             \
      return err;                                                              \
  } while (0)

#define ERROR_CHECK_DEVNUM(device)                                             \
  ERROR_IF(((device < 0) || ((size_t)device >= Backend->getNumDevices())),     \
           hipErrorInvalidDevice)

#define ERROR_CHECK_DEVHANDLE(device)                                          \
  auto I = std::find(Backend->getDevices().begin(),                            \
                     Backend->getDevices().end(), device);                     \
  ERROR_IF(I == Backend->getDevices().end(), hipErrorInvalidDevice)

#endif