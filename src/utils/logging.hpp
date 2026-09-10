#pragma once

// -- Pre-requisites --
// 1. Link against spdlog.
// 2. For debug-only logs (DLOG), compile with -DNDEBUG for release builds.

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <atomic> // For std::atomic
#include <chrono> // For std::chrono
#include <memory>
#include <mutex> // For LOG_EVERY_T
#include <sstream> // For stream-based logging
#include <string.h> // For strrchr
#include <string> // For CHECK macros
#include <vector>

//--------------------------------------------------------------------------------//
// Platform and Filename Helpers
//--------------------------------------------------------------------------------//

#ifndef OS_SEP
#ifdef _MSC_VER
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif
#endif

#define FILENAME_ (strrchr(__FILE__, OS_SEP) ? strrchr(__FILE__, OS_SEP) + 1 : __FILE__)

//--------------------------------------------------------------------------------//
// Internal Macro Helpers (Do not use directly)
//--------------------------------------------------------------------------------//

namespace utils { namespace logger_internal {

// A NullStream that does nothing, used to disable logging statements at compile time.
class NullStream {
public:
    template <typename T>
    NullStream& operator<<(const T&) {
        return *this;
    }
    NullStream& stream() { return *this; }
};

// A LogStream that captures data and logs it on destruction.
class LogStream {
public:
    LogStream(spdlog::level::level_enum level, const char* file, int line, const char* func) : level_(level), loc_{file, line, func} {}

    ~LogStream() {
        // Log the buffered message. Use a simple format string to pass it through.
        spdlog::log(loc_, level_, "{}", stream_.str());
    }

    template <typename T>
    LogStream& operator<<(const T& msg) {
        stream_ << msg;
        return *this;
    }

    // Returns the underlying stream for CHECK macros
    std::stringstream& stream() { return stream_; }

protected:
    std::stringstream stream_;
    spdlog::level::level_enum level_;
    spdlog::source_loc loc_;
};

// A FatalLogStream that logs and then terminates the program.
class FatalLogStream : public LogStream {
public:
    using LogStream::LogStream; // Inherit constructor

    ~FatalLogStream() {
        // The message is logged first by the base class destructor,
        // which runs AFTER this destructor's body. So we log here and then abort.
        spdlog::log(loc_, level_, "{}", stream_.str());
        std::abort();
    }
};

// Helper for time-based logging state
struct LogEveryTState {
    std::mutex mtx;
    std::chrono::steady_clock::time_point last_log{};
};
inline bool ShouldLogEveryT(LogEveryTState& state, double seconds) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(state.mtx);
    if (now - state.last_log >
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds))) {
        state.last_log = now;
        return true;
    }
    return false;
}
}} // namespace utils::logger_internal

// Core `fmt`-style logging implementation
#define HELPER_LOG_TRACE(...)    spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::trace, __VA_ARGS__)
#define HELPER_LOG_DEBUG(...)    spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::debug, __VA_ARGS__)
#define HELPER_LOG_INFO(...)     spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::info, __VA_ARGS__)
#define HELPER_LOG_WARN(...)     spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::warn, __VA_ARGS__)
#define HELPER_LOG_ERROR(...)    spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::err, __VA_ARGS__)
#define HELPER_LOG_CRITICAL(...) spdlog::log({FILENAME_, __LINE__, __FUNCTION__}, spdlog::level::critical, __VA_ARGS__)

// Helper for CHECK_XX `fmt` macros
#define HELPER_CHECK_OP_FMT(val1, val2, op_str, op)                                           \
    do {                                                                                      \
        auto v1 = (val1);                                                                     \
        auto v2 = (val2);                                                                     \
        if (!(v1 op v2)) {                                                                    \
            LOG(CRITICAL, "Check failed: {} {} {} ({} vs {})", #val1, op_str, #val2, v1, v2); \
            std::abort();                                                                     \
        }                                                                                     \
    } while (0)

//--------------------------------------------------------------------------------//
// Public Glog-Style Logging Macros
//--------------------------------------------------------------------------------//

// --- 1. FMT-Style Logging (High Performance) ---

#define LOG(level, ...) HELPER_LOG_##level(__VA_ARGS__)
#define LOG_IF(level, condition, ...) \
    if (condition) {                  \
        LOG(level, __VA_ARGS__);      \
    }
#define LOG_EVERY_N(level, n, ...)                                                   \
    do {                                                                             \
        static std::atomic<size_t> counter(0);                                       \
        if ((n) > 0 && counter.fetch_add(1, std::memory_order_relaxed) % (n) == 0) { \
            LOG(level, __VA_ARGS__);                                                 \
        }                                                                            \
    } while (0)
#define LOG_FIRST_N(level, n, ...)                                                      \
    do {                                                                                \
        static std::atomic<size_t> counter(0);                                          \
        if (counter.fetch_add(1, std::memory_order_relaxed) < static_cast<size_t>(n)) { \
            LOG(level, __VA_ARGS__);                                                    \
        }                                                                               \
    } while (0)
#define LOG_EVERY_T(level, seconds, ...)                               \
    do {                                                               \
        static utils::logger_internal::LogEveryTState state;           \
        if (utils::logger_internal::ShouldLogEveryT(state, seconds)) { \
            LOG(level, __VA_ARGS__);                                   \
        }                                                              \
    } while (0)
#define LOG_IF_EVERY_N(level, condition, n, ...) \
    if (condition) {                             \
        LOG_EVERY_N(level, n, __VA_ARGS__);      \
    }

// --- 2. Stream-Style Logging (Flexible, glog-like) ---

#define LOG_STREAM(level) utils::logger_internal::LogStream(HELPER_GET_SPDLOG_LEVEL(level), FILENAME_, __LINE__, __FUNCTION__)
#define LOG_STREAM_IF(level, condition) \
    if (condition) LOG_STREAM(level) else utils::logger_internal::NullStream()
#define LOG_STREAM_EVERY_N(level, n)                                         \
    (((n) > 0 &&                                                             \
      [] {                                                                   \
          static std::atomic<size_t> counter(0);                             \
          return counter.fetch_add(1, std::memory_order_relaxed) % (n) == 0; \
      }())                                                                   \
         ? LOG_STREAM(level)                                                 \
         : utils::logger_internal::NullStream())

// --- 3. Simplified Level-Only Macros (As requested) ---
#define LOG_TRACE(...)    HELPER_LOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...)    HELPER_LOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...)     HELPER_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)     HELPER_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...)    HELPER_LOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) HELPER_LOG_CRITICAL(__VA_ARGS__)

// --- 4. Debug-Only Logging (DLOG - compiled out if NDEBUG is defined) ---

#ifdef NDEBUG
#define DLOG(level, ...)                 (void)0
#define DLOG_IF(level, ...)              (void)0
#define DLOG_STREAM(level)               utils::logger_internal::NullStream()
#define DLOG_STREAM_IF(level, condition) utils::logger_internal::NullStream()
#else
#define DLOG(level, ...)                 LOG(level, __VA_ARGS__)
#define DLOG_IF(level, condition, ...)   LOG_IF(level, condition, __VA_ARGS__)
#define DLOG_STREAM(level)               LOG_STREAM(level)
#define DLOG_STREAM_IF(level, condition) LOG_STREAM_IF(level, condition)
#endif

// --- 5. Fatal Assertion Checks (CHECK - ALWAYS ON) ---

// `fmt` style check
#define CHECK(condition, ...)                                            \
    do {                                                                 \
        if (!(condition)) {                                              \
            LOG(CRITICAL, "Check failed: " #condition ". " __VA_ARGS__); \
            std::abort();                                                \
        }                                                                \
    } while (0)
#define CHECK_EQ(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, "==", ==)
#define CHECK_NE(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, "!=", !=)
#define CHECK_LT(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, "<", <)
#define CHECK_LE(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, "<=", <=)
#define CHECK_GT(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, ">", >)
#define CHECK_GE(v1, v2) HELPER_CHECK_OP_FMT(v1, v2, ">=", >=)

// Stream style check
#define CHECK_STREAM(condition)                                                                                     \
    if (condition)                                                                                                  \
        (void)0;                                                                                                    \
    else                                                                                                            \
        utils::logger_internal::FatalLogStream(spdlog::level::critical, FILENAME_, __LINE__, __FUNCTION__).stream() \
            << "Check failed: " #condition ". "

//--------------------------------------------------------------------------------//
// Logger Initialization and Configuration
//--------------------------------------------------------------------------------//
namespace utils { namespace logger {
enum class LogLevel : int8_t { TRACE, DEBUG, INFO, WARN, ERR, CRITICAL, OFF };
struct LoggerParam {
    LogLevel logLevel = LogLevel::DEBUG; // default log level
    std::string logDir; // if empty, no file logger will be created
    std::string logName; // if empty, default name will be used
    size_t maxRetentionDays = 0; // if 0, no retention limit
};
class LoggerGenerator {
public:
    static spdlog::level::level_enum to_spdlog_level(const LogLevel& log_level) {
        if (log_level == LogLevel::TRACE) return spdlog::level::trace;
        if (log_level == LogLevel::DEBUG) return spdlog::level::debug;
        if (log_level == LogLevel::INFO) return spdlog::level::info;
        if (log_level == LogLevel::WARN) return spdlog::level::warn;
        if (log_level == LogLevel::ERR) return spdlog::level::err;
        if (log_level == LogLevel::CRITICAL) return spdlog::level::critical;
        return spdlog::level::off;
    }
    static std::shared_ptr<spdlog::logger> gen_logger(const LoggerParam& param) {
        std::vector<spdlog::sink_ptr> sinks;
        auto spdlog_level = to_spdlog_level(param.logLevel);
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        // if logDir is not empty, create a daily rotating file logger
        if (!param.logDir.empty()) {
            std::string path = param.logDir + OS_SEP + param.logName;
            sinks.push_back(std::make_shared<spdlog::sinks::daily_file_sink_mt>(path, 0, 0, false, param.maxRetentionDays));
        }
        auto logger = std::make_shared<spdlog::logger>(param.logName, sinks.begin(), sinks.end());
        logger->set_level(spdlog_level);

        // log_level, datetime, thread_id, file_name, line_num, message
        // eg: D25-0627 01:08:45.513 T44956 TextureManager.cpp:43] Cleaning up textures
        logger->set_pattern("%^%L%$%C-%m%d %T.%e T%t %s:%#] %v");

        // glog style log format.
        // eg: D0627 01:13:20.856707 45548 TextureManager.cpp:43] Cleaning up textures
        // logger->set_pattern("%^%L%$%m%d %H:%M:%S.%f %t %s:%#] %v");

        // 更详细的pattern.
        // eg: [2025-06-27 01:16:27.088] [debug  ] [thread 43712] [TextureManager.cpp:43] Cleaning up textures
        // logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%-7l%$] [thread %t] [%s:%#] %v");

        return logger;
    }
    static void set_default_logger(LoggerParam&& logParam) {
        auto logger = gen_logger(logParam);
        spdlog::set_default_logger(logger);
        spdlog::flush_on(to_spdlog_level(logParam.logLevel));
        spdlog::flush_every(std::chrono::seconds(1));
    }
};
static void InitializeGlobalLogger(LoggerParam logParam = {}) {
    // if no logParam is provided, use default log level and no file logger
    LoggerGenerator::set_default_logger(std::move(logParam));
}
}} // namespace utils::logger