/**
 * @file CHIPDriver.cc
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Definitions of extern declared functions and objects in CHIPDriver.hh
 * Initializing the CHIP runtime with backend selection through CHIP_BE
 * environment variable.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "CHIPDriver.hh"

#include <string>

#include "backend/backends.hh"

std::once_flag initialized;
std::once_flag uninitialized;
CHIPBackend* Backend;

std::string read_env_var(std::string ENV_VAR) {
  const char* ENV_VAR_IN = std::getenv(ENV_VAR.c_str());
  if (ENV_VAR_IN == nullptr) {
    return std::string();
  }
  std::string var = std::string(ENV_VAR_IN);
  std::transform(var.begin(), var.end(), var.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  return var;
};

std::string read_backend_selection();

void read_env_vars(std::string& CHIPPlatformStr, std::string& CHIPDeviceTypeStr,
                   std::string& CHIPDeviceStr) {
  CHIPPlatformStr = read_env_var("CHIP_PLATFORM");
  if (CHIPPlatformStr.size() == 0) CHIPPlatformStr = "0";

  CHIPDeviceTypeStr = read_env_var("CHIP_DEVICE_TYPE");
  if (CHIPDeviceTypeStr.size() == 0) CHIPDeviceTypeStr = "default";

  CHIPDeviceStr = read_env_var("CHIP_DEVICE");
  if (CHIPDeviceStr.size() == 0) CHIPDeviceStr = "0";

  logDebug("CHIP_PLATFORM={}", CHIPPlatformStr.c_str());
  logDebug("CHIP_DEVICE_TYPE={}", CHIPDeviceTypeStr.c_str());
  logDebug("CHIP_DEVICE={}", CHIPDeviceStr.c_str());
};

void CHIPInitializeCallOnce(std::string BE) {
  std::string CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr;
  read_env_vars(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
  logDebug("CHIPDriver Initialize");
  // Get the current Backend Env Var

  // If no BE is passed to init explicitly, read env var
  std::string CHIP_BE;
  if (BE.size() == 0) {
    CHIP_BE = read_env_var("CHIP_BE");
  } else {
    CHIP_BE = BE;
  }

  // TODO Check configuration for what backends are configured
  if (!CHIP_BE.compare("opencl")) {
    logTrace("CHIPBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
  } else if (!CHIP_BE.compare("level0")) {
    logTrace("CHIPBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
  } else if (!CHIP_BE.compare("")) {
    logWarn("CHIP_BE was not set. Defaulting to OPENCL");
    Backend = new CHIPBackendOpenCL();
  } else {
    CHIPERR_LOG_AND_THROW(
        "Invalid CHIP-SPV Backend Selected. Accepted values : level0, opencl.",
        hipErrorInitializationError);
  }
  Backend->initialize(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
}

extern void CHIPInitialize(std::string BE) {
  std::call_once(initialized, &CHIPInitializeCallOnce, BE);
};

void CHIPUninitializeCallOnce() {
  logTrace("Uninitializing CHIP...");
  Backend->uninitialize();
}

extern void CHIPUninitialize() {
  std::call_once(uninitialized, &CHIPUninitializeCallOnce);
}
