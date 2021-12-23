#include "logging.hh"
#include <iostream>
#include <mutex>

std::once_flag SpdlogWasSetup;

void setupSpdlog() { std::call_once(SpdlogWasSetup, &_setupSpdlog); }

void _setupSpdlog() {
  spdlog::set_default_logger(spdlog::stderr_color_mt("CHIP"));
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("%n %^%l%$ [TID %t] [%E.%F] : %v");

  spdlog::level::level_enum spd_loglevel;

  const char *loglevel = getenv("CHIP_LOGLEVEL");
  if (loglevel) {
    // std::cout << "CHIP_LOGLEVEL=" << loglevel << "\n";
    std::string level(loglevel);
    if (level == "debug") spd_loglevel = spdlog::level::debug;
    if (level == "info") spd_loglevel = spdlog::level::info;
    if (level == "warn") spd_loglevel = spdlog::level::warn;
    if (level == "err") spd_loglevel = spdlog::level::err;
    if (level == "crit") spd_loglevel = spdlog::level::critical;
    if (level == "off") spd_loglevel = spdlog::level::off;
  } else {
    // std::cout << "CHIP_LOGLEVEL was not set. Default to trace\n";
    spd_loglevel = spdlog::level::trace;
  }

  spdlog::set_level(spd_loglevel);
}
