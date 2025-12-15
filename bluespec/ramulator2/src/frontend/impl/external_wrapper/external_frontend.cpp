#include <filesystem>
#include <iostream>
#include <fstream>
#include <atomic>

#include "../../frontend.h"
#include "base/exception.h"

namespace Ramulator {

class ExternalFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, ExternalFrontend, "ExternalFrontend", "External frontend for direct memory requests.")

  private:
    std::atomic<int> outstanding_requests{0};
    Logger_t m_logger;

  public:
    void init() override { 
      m_logger = Logging::create_logger("ExternalFrontend");
      m_logger->info("External frontend initialized");
    };
    
    void tick() override { 
      // External frontend doesn't need to do anything in tick
    };

    bool receive_external_requests(int req_type_id, Addr_t addr, int source_id, std::function<void(Request&)> callback) override {
      // Wrap the callback to track outstanding requests
      auto wrapped_callback = [this, callback](Request& req) {
        outstanding_requests--;
        m_logger->debug("Request completed for addr 0x{:x}, outstanding: {}", req.addr, outstanding_requests.load());
        callback(req);
      };
      
      bool success = m_memory_system->send({addr, req_type_id, source_id, wrapped_callback});
      if (success) {
        outstanding_requests++;
        m_logger->debug("Request sent for addr 0x{:x}, outstanding: {}", addr, outstanding_requests.load());
      }
      return success;
    }

  private:
    bool is_finished() override { 
      bool finished = (outstanding_requests == 0);
      if (finished) {
        m_logger->info("External frontend finished - no outstanding requests");
      }
      return finished;
    };
};

}        // namespace Ramulator 