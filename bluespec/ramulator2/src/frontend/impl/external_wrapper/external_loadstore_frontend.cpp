#include <filesystem>
#include <iostream>
#include <fstream>
#include <atomic>
#include <queue>
#include <mutex>

#include "../../frontend.h"
#include "base/exception.h"

namespace Ramulator {

class ExternalLoadStoreFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, ExternalLoadStoreFrontend, "ExternalLoadStoreFrontend", "External frontend that supports both trace files and external requests.")

  private:
    struct Trace {
      bool is_write;
      Addr_t addr;
    };
    std::vector<Trace> m_trace;
    std::queue<std::pair<Addr_t, bool>> m_external_requests; // addr, is_write
    std::mutex m_external_mutex;

    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;
    size_t m_trace_count = 0;
    std::atomic<int> m_outstanding_requests{0};

    Logger_t m_logger;
    bool m_use_trace = false;

  public:
    void init() override { 
      m_logger = Logging::create_logger("ExternalLoadStoreFrontend");
      
      // Check if we have a trace file parameter
      try {
        std::string trace_path_str = param<std::string>("path").desc("Path to the load store trace file.").default_val("");
        if (!trace_path_str.empty()) {
          m_use_trace = true;
          m_clock_ratio = param<uint>("clock_ratio").required();
          m_logger->info("Loading trace file {} ...", trace_path_str);
          init_trace(trace_path_str);
          m_logger->info("Loaded {} lines.", m_trace.size());
        } else {
          m_clock_ratio = param<uint>("clock_ratio").default_val(8);
          m_logger->info("External frontend initialized (no trace file)");
        }
      } catch (...) {
        // No trace file parameter, use external requests only
        m_clock_ratio = param<uint>("clock_ratio").default_val(8);
        m_logger->info("External frontend initialized (no trace file)");
      }
    };
    
    void tick() override { 
      if (m_use_trace && m_curr_trace_idx < m_trace_length) {
        const Trace& t = m_trace[m_curr_trace_idx];
        bool request_sent = m_memory_system->send({t.addr, t.is_write ? Request::Type::Write : Request::Type::Read});
        if (request_sent) {
          m_curr_trace_idx++;
          m_trace_count++;
          m_outstanding_requests++;
        }
      }
    };

    bool receive_external_requests(int req_type_id, Addr_t addr, int source_id, std::function<void(Request&)> callback) override {
      // Wrap the callback to track outstanding requests
      auto wrapped_callback = [this, callback](Request& req) {
        m_outstanding_requests--;
        m_logger->debug("Request completed for addr 0x{:x}, outstanding: {}", req.addr, m_outstanding_requests.load());
        callback(req);
      };
      
      bool success = m_memory_system->send({addr, req_type_id, source_id, wrapped_callback});
      if (success) {
        m_outstanding_requests++;
        m_logger->debug("External request sent for addr 0x{:x}, outstanding: {}", addr, m_outstanding_requests.load());
      }
      return success;
    }

  private:
    void init_trace(const std::string& file_path_str) {
      fs::path trace_path(file_path_str);
      if (!fs::exists(trace_path)) {
        throw ConfigurationError("Trace {} does not exist!", file_path_str);
      }

      std::ifstream trace_file(trace_path);
      if (!trace_file.is_open()) {
        throw ConfigurationError("Trace {} cannot be opened!", file_path_str);
      }

      std::string line;
      while (std::getline(trace_file, line)) {
        std::vector<std::string> tokens;
        tokenize(tokens, line, " ");

        if (tokens.size() != 2) {
          throw ConfigurationError("Trace {} format invalid!", file_path_str);
        }

        bool is_write = false; 
        if (tokens[0] == "LD") {
          is_write = false;
        } else if (tokens[0] == "ST") {
          is_write = true;
        } else {
          throw ConfigurationError("Trace {} format invalid!", file_path_str);
        }

        Addr_t addr = -1;
        if (tokens[1].compare(0, 2, "0x") == 0 | tokens[1].compare(0, 2, "0X") == 0) {
          addr = std::stoll(tokens[1].substr(2), nullptr, 16);
        } else {
          addr = std::stoll(tokens[1]);
        }
        m_trace.push_back({is_write, addr});
      }

      trace_file.close();
      m_trace_length = m_trace.size();
    }

    bool is_finished() override { 
      bool trace_finished = !m_use_trace || (m_trace_count >= m_trace_length);
      bool external_finished = (m_outstanding_requests == 0);
      bool finished = trace_finished && external_finished;
      
      if (finished) {
        m_logger->info("External frontend finished - trace: {}, external: {}", trace_finished, external_finished);
      }
      return finished;
    };
};

}        // namespace Ramulator 