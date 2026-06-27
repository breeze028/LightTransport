#pragma once

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace lt {

enum class LogLevel {
    Trace = 0,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off,
};

struct LogConfig {
    std::string logger_name = "lt";
    bool enable_console = false;
    bool enable_file = false;
    LogLevel console_level = LogLevel::Info;
    LogLevel file_level = LogLevel::Debug;
    std::string file_path;
    std::size_t max_file_size = 5u * 1024u * 1024u;
    std::size_t max_files = 3;
};

struct LogRecord {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level = LogLevel::Info;
    std::string logger;
    std::string message;
    std::string file;
    int line = 0;
    std::string function;
    std::uint32_t thread_id = 0;
};

using LogObserver = std::function<void(const LogRecord&)>;
using LogObserverHandle = std::uint64_t;

void initialize_logging(const LogConfig& config);
void shutdown_logging();
void set_log_level(LogLevel level);
LogLevel log_level();
void flush_logs();
LogObserverHandle add_log_observer(LogObserver observer);
void remove_log_observer(LogObserverHandle handle);

const char* log_level_name(LogLevel level);
LogLevel parse_log_level(std::string_view name, LogLevel fallback = LogLevel::Info);
bool should_log(LogLevel level);
void log_message(LogLevel level, const char* file, int line, const char* function, std::string message);

namespace detail {

template <typename T>
std::string log_arg_to_string(T&& value) {
    std::ostringstream output;
    output << std::forward<T>(value);
    return output.str();
}

inline void append_formatted_message(std::string& output, std::string_view format, const std::string* args, std::size_t arg_count) {
    std::size_t arg_index = 0;
    for (std::size_t i = 0; i < format.size(); ++i) {
        if (format[i] == '{' && i + 1 < format.size() && format[i + 1] == '{') {
            output.push_back('{');
            ++i;
        } else if (format[i] == '}' && i + 1 < format.size() && format[i + 1] == '}') {
            output.push_back('}');
            ++i;
        } else if (format[i] == '{' && i + 1 < format.size() && format[i + 1] == '}') {
            if (arg_index < arg_count) {
                output += args[arg_index++];
            } else {
                output += "{}";
            }
            ++i;
        } else {
            output.push_back(format[i]);
        }
    }
    while (arg_index < arg_count) {
        output.push_back(' ');
        output += args[arg_index++];
    }
}

template <typename... Args>
std::string format_log_message(std::string_view format, Args&&... args) {
    std::array<std::string, sizeof...(Args)> values{log_arg_to_string(std::forward<Args>(args))...};
    std::string output;
    output.reserve(format.size() + values.size() * 8u);
    append_formatted_message(output, format, values.data(), values.size());
    return output;
}

} // namespace detail

template <typename... Args>
void log_formatted(
    LogLevel level,
    const char* file,
    int line,
    const char* function,
    std::string_view format,
    Args&&... args) {
    if (!should_log(level)) {
        return;
    }
    try {
        log_message(level, file, line, function, detail::format_log_message(format, std::forward<Args>(args)...));
    } catch (const std::exception& exception) {
        log_message(LogLevel::Error, file, line, function, std::string("Log formatting failed: ") + exception.what());
    }
}

} // namespace lt

#define LT_LOG_TRACE(...) ::lt::log_formatted(::lt::LogLevel::Trace, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LT_LOG_DEBUG(...) ::lt::log_formatted(::lt::LogLevel::Debug, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LT_LOG_INFO(...) ::lt::log_formatted(::lt::LogLevel::Info, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LT_LOG_WARN(...) ::lt::log_formatted(::lt::LogLevel::Warn, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LT_LOG_ERROR(...) ::lt::log_formatted(::lt::LogLevel::Error, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LT_LOG_CRITICAL(...) ::lt::log_formatted(::lt::LogLevel::Critical, __FILE__, __LINE__, __func__, __VA_ARGS__)
