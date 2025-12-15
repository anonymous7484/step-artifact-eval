#include "base/logging.h"


namespace Ramulator {

Logger_t Logging::create_logger(std::string name, std::string pattern) {
  // Check if logger already exists first
  auto existing_logger = spdlog::get("Ramulator::" + name);
  if (existing_logger) {
    return existing_logger;
  }

  // Create new logger only if it doesn't exist
  auto logger = spdlog::stdout_color_st("Ramulator::" + name);

  if (!logger) {
    throw InitializationError("Error creating logger {}!", name);
  }

  logger->set_pattern(pattern);
  logger->set_level(spdlog::level::debug);
  return logger;
}

Logger_t Logging::get(std::string name) {
  auto logger = spdlog::get("Ramulator::" + name);
  if (logger) {
    return logger;
  } else {
    throw std::runtime_error(
      fmt::format(
        "Logger {} does not exist!",
        name
      )
    );
  }
}

bool Logging::_create_base_logger() {
  try {
    auto logger = create_logger("Base");
    if (logger) {
      return true;
    } else {
      throw InitializationError("Error creating the base logger!");
    }
  } catch (const spdlog::spdlog_ex& e) {
    // Logger already exists, this is fine
    return true;
  } catch (const InitializationError& e) {
    // Re-throw initialization errors
    throw;
  }
  return false;
}

}        // namespace Ramulator
