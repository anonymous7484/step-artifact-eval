// ramulator2-rs/ramulator_wrapper.cpp
#include "ramulator_wrapper.h"

// C++ standard library includes
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>


// ======  Logic to make Ramulator not crash when imported as a library multiple times  ====== 

// Disable spdlog completely before any includes
#define SPDLOG_DISABLE
#include <spdlog/spdlog.h>
#undef SPDLOG_DISABLE

// Global flag to track if spdlog has been initialized
static bool spdlog_initialized = false;

// Function to safely initialize spdlog only once
void initialize_spdlog_safely() {
    if (!spdlog_initialized) {
        try {
            // Completely disable spdlog
            spdlog::set_level(spdlog::level::off);
            spdlog::flush_on(spdlog::level::off);
            
            // Try to get existing logger first
            auto existing_logger = spdlog::get("Ramulator::Base");
            if (!existing_logger) {
                // Create a simple logger that does nothing
                auto logger = spdlog::stdout_color_mt("Ramulator::Base");
                if (logger) {
                    logger->set_level(spdlog::level::off);
                }
            }
            spdlog_initialized = true;
        } catch (const spdlog::spdlog_ex& e) {
            // Logger already exists, ignore the error
            spdlog_initialized = true;
        } catch (...) {
            // Any other error, just mark as initialized
            spdlog_initialized = true;
        }
    }
}

// Global flag to track if ramulator library has been initialized
static bool ramulator_initialized = false;

// Function to safely initialize ramulator library only once
void initialize_ramulator_safely() {
    if (!ramulator_initialized) {
        try {
            // Any global ramulator initialization can go here
            ramulator_initialized = true;
        } catch (...) {
            // Any error, just mark as initialized
            ramulator_initialized = true;
        }
    }
}


RamulatorWrapper::RamulatorWrapper(const char *config_path)
    : config_path(config_path) {
  if (!config_path) {
    throw std::invalid_argument("Invalid argument");
  }

  // Initialize spdlog safely before any ramulator operations
  initialize_spdlog_safely();
  
  // Initialize ramulator library safely
  initialize_ramulator_safely();

  std::cout << "Loading config file: " << config_path << std::endl;

  // Parse configuration
  auto config = Ramulator::Config::parse_config_file(config_path, {});
  // auto test = Ramulator::Config::parse_content(config_path, {});

  // Create frontend with the configuration
  // frontend = std::make_unique<Ramulator::Frontend>(*(config));
  frontend = Ramulator::Factory::create_frontend(config);
  memory_system = Ramulator::Factory::create_memory_system(config);

  frontend->connect_memory_system(memory_system);
  memory_system->connect_frontend(frontend);

  initialized = true;
}

RamulatorWrapper::~RamulatorWrapper() {
  if (initialized) {
    frontend->finalize();
    memory_system->finalize();
  }
}

bool RamulatorWrapper::send_request(uint64_t addr, bool is_write) {
  if (!initialized) {
    throw std::runtime_error("Ramulator instance not initialized");
  }

  std::lock_guard<std::mutex> lock(frontend_mutex);

  bool enqueue_success = false;

  if (is_write) {
    enqueue_success = frontend->receive_external_requests(
        1, addr, 0, [this](Ramulator::Request &req) {
          this->out_queue.push(req.addr);
          // int chan_id = req.addr_vec[0];
          std::atomic_fetch_sub(&num_outstanding_writes, 1);
        });

    if (enqueue_success) {
      // std::cout << "Request enqueued successfully" << std::endl;
      // auto &packet = outstandingReads.find(addr)->second;
      std::atomic_fetch_add(&num_outstanding_writes, 1);
    } else {
      // std::cout << "Request enqueue failed (queue full)" << std::endl;
    }
  } else {
    enqueue_success = frontend->receive_external_requests(
        0, addr, 0, [this](Ramulator::Request &req) {
          this->out_queue.push(req.addr);
          std::atomic_fetch_sub(&num_outstanding_reads, 1);
        });
    if (enqueue_success) {
      // std::cout << "Request enqueued successfully" << std::endl;
      // auto &packet = outstandingReads.find(addr)->second;
      std::atomic_fetch_add(&num_outstanding_reads, 1);
    } else {
      // std::cout << "Request enqueue failed (queue full)" << std::endl;
    }
  }

  return enqueue_success;
}

void RamulatorWrapper::tick() {
  std::atomic_fetch_add(&tCK, 1);
  frontend->tick();
  memory_system->tick();
}

bool RamulatorWrapper::is_finished() {
  return frontend->is_finished();
  // return num_outstanding_reads == 0 && num_outstanding_writes == 0 && frontend->is_finished();
  // return outgoing_reqs == 0 && frontend->is_finished();
}


// Global simulator instance for BDPI functions
static RamulatorWrapper* global_sim = nullptr;
static const char* default_config_path = "./ramulator2/hbm.yaml";

// BDPI functions for Bluespec interface
void init_ramulator(void) {
  if (global_sim == nullptr) {
    try {
      global_sim = new RamulatorWrapper(default_config_path);
      std::cout << "Ramulator wrapper initialized successfully" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Failed to initialize ramulator wrapper: " << e.what() << std::endl;
    }
  }
}

void free_ramulator(void) {
  if (global_sim != nullptr) {
    delete global_sim;
    global_sim = nullptr;
    std::cout << "Ramulator wrapper freed" << std::endl;
  }
}

bool ramulator_send(uint64_t addr, bool is_write) {
  if (global_sim == nullptr) {
    std::cerr << "Error: Ramulator wrapper not initialized (called send before init)" << std::endl;
    return false;
  }
  
  bool success = global_sim->send_request(addr, is_write);
  if (!success) {
    // std::cerr << "Warning: Failed to send request to ramulator" << std::endl;
  }
  return success;
}

void ramulator_tick(void) {
  if (global_sim == nullptr) {
    std::cerr << "Error: Ramulator wrapper not initialized (called tick before init)" << std::endl;
    return;
  }
  
  global_sim->tick();
}

uint64_t ramulator_get_cycle(void) {
  if (global_sim == nullptr) {
    std::cerr << "Error: Ramulator wrapper not initialized (called get_cycle before init)" << std::endl;
    return -1;
  }
  
  uint64_t cycle = global_sim->get_cycle();
  // Note: In a real implementation, you might want to return this value
  // For now, we just print it
  std::cout << "Current cycle: " << cycle << std::endl;
  return cycle;
}

bool ramulator_ret_available(void) {
  if (global_sim == nullptr) {
    // std::cerr << "Error: Ramulator wrapper not initialized (called return_available before init)" << std::endl;
    return false;
  }
  
  bool available = global_sim->return_available();
  // Note: In a real implementation, you might want to return this value
  // For now, we just print it
  // std::cout << "Return available: " << (available ? "true" : "false") << std::endl;
  return available;
}

uint64_t ramulator_pop(void) {
  if (global_sim == nullptr) {
    std::cerr << "Error: Ramulator wrapper not initialized" << std::endl;
    return 0;
  }
  
  if (!global_sim->return_available()) {
    std::cerr << "Warning: No return value available" << std::endl;
    return 0;
  }
  
  return global_sim->pop();
}

bool ramulator_is_finished(void) {
  if (global_sim == nullptr) {
    std::cerr << "Error: Ramulator wrapper not initialized" << std::endl;
    return true; // Treat as finished if not initialized
  }
  
  return global_sim->is_finished();
}