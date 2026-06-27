#include "lt/log.h"

#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lt {
namespace {

struct LogState {
    std::mutex mutex;
    std::shared_ptr<spdlog::logger> logger;
    std::string logger_name = "lt";
    std::vector<std::pair<LogObserverHandle, LogObserver>> observers;
    LogObserverHandle next_observer = 1;
};

LogState& state() {
    static LogState value;
    return value;
}

std::atomic<int>& active_level_storage() {
    static std::atomic<int> value{static_cast<int>(LogLevel::Off)};
    return value;
}

spdlog::level::level_enum to_spdlog(LogLevel level) {
    switch (level) {
    case LogLevel::Trace: return spdlog::level::trace;
    case LogLevel::Debug: return spdlog::level::debug;
    case LogLevel::Info: return spdlog::level::info;
    case LogLevel::Warn: return spdlog::level::warn;
    case LogLevel::Error: return spdlog::level::err;
    case LogLevel::Critical: return spdlog::level::critical;
    case LogLevel::Off: return spdlog::level::off;
    }
    return spdlog::level::info;
}

LogLevel active_level_from_config(const LogConfig& config) {
    LogLevel level = LogLevel::Off;
    bool any_enabled = false;
    const auto consider = [&](bool enabled, LogLevel candidate) {
        if (!enabled || candidate == LogLevel::Off) {
            return;
        }
        if (!any_enabled || static_cast<int>(candidate) < static_cast<int>(level)) {
            level = candidate;
        }
        any_enabled = true;
    };
    consider(config.enable_console, config.console_level);
    consider(config.enable_file && !config.file_path.empty(), config.file_level);
    return any_enabled ? level : LogLevel::Off;
}

std::string lowercase(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

std::shared_ptr<spdlog::logger> current_logger() {
    LogState& log_state = state();
    std::lock_guard lock(log_state.mutex);
    return log_state.logger;
}

void notify_observers(
    LogLevel level,
    const char* file,
    int line,
    const char* function,
    const std::string& message) {
    LogRecord record;
    record.timestamp = std::chrono::system_clock::now();
    record.level = level;
    {
        LogState& log_state = state();
        std::lock_guard lock(log_state.mutex);
        record.logger = log_state.logger_name;
    }
    record.message = message;
    record.file = file ? file : "";
    record.line = line;
    record.function = function ? function : "";
    record.thread_id = static_cast<std::uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    std::vector<LogObserver> observers;
    {
        LogState& log_state = state();
        std::lock_guard lock(log_state.mutex);
        observers.reserve(log_state.observers.size());
        for (const auto& entry : log_state.observers) {
            observers.push_back(entry.second);
        }
    }
    for (const LogObserver& observer : observers) {
        if (observer) {
            observer(record);
        }
    }
}

} // namespace

void initialize_logging(const LogConfig& config) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.reserve(2);

    if (config.enable_console && config.console_level != LogLevel::Off) {
        auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        console->set_level(to_spdlog(config.console_level));
        console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        sinks.push_back(std::move(console));
    }

    if (config.enable_file && config.file_level != LogLevel::Off && !config.file_path.empty()) {
        try {
            const std::filesystem::path path(config.file_path);
            if (const std::filesystem::path parent = path.parent_path(); !parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            const std::size_t max_file_size = std::max<std::size_t>(1024u, config.max_file_size);
            const std::size_t max_files = std::max<std::size_t>(1u, config.max_files);
            auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(config.file_path, max_file_size, max_files);
            file->set_level(to_spdlog(config.file_level));
            file->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [t%t] [%s:%# %!] %v");
            sinks.push_back(std::move(file));
        } catch (const std::exception&) {
        }
    }

    if (sinks.empty()) {
        sinks.push_back(std::make_shared<spdlog::sinks::null_sink_mt>());
    }

    const std::string logger_name = config.logger_name.empty() ? "lt" : config.logger_name;
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    const LogLevel active_level = active_level_from_config(config);
    logger->set_level(to_spdlog(active_level));
    logger->flush_on(spdlog::level::warn);

    LogState& log_state = state();
    {
        std::lock_guard lock(log_state.mutex);
        log_state.logger = std::move(logger);
        log_state.logger_name = logger_name;
    }
    active_level_storage().store(static_cast<int>(active_level), std::memory_order_relaxed);
}

void shutdown_logging() {
    std::shared_ptr<spdlog::logger> logger;
    {
        LogState& log_state = state();
        std::lock_guard lock(log_state.mutex);
        logger = std::move(log_state.logger);
    }
    if (logger) {
        logger->flush();
    }
    active_level_storage().store(static_cast<int>(LogLevel::Off), std::memory_order_relaxed);
}

void set_log_level(LogLevel level) {
    LogState& log_state = state();
    std::lock_guard lock(log_state.mutex);
    if (log_state.logger) {
        log_state.logger->set_level(to_spdlog(level));
        for (const spdlog::sink_ptr& sink : log_state.logger->sinks()) {
            sink->set_level(to_spdlog(level));
        }
    }
    active_level_storage().store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel log_level() {
    return static_cast<LogLevel>(active_level_storage().load(std::memory_order_relaxed));
}

void flush_logs() {
    if (std::shared_ptr<spdlog::logger> logger = current_logger()) {
        logger->flush();
    }
}

LogObserverHandle add_log_observer(LogObserver observer) {
    if (!observer) {
        return 0;
    }
    LogState& log_state = state();
    std::lock_guard lock(log_state.mutex);
    const LogObserverHandle handle = log_state.next_observer++;
    log_state.observers.push_back({handle, std::move(observer)});
    return handle;
}

void remove_log_observer(LogObserverHandle handle) {
    if (handle == 0) {
        return;
    }
    LogState& log_state = state();
    std::lock_guard lock(log_state.mutex);
    const auto it = std::remove_if(log_state.observers.begin(), log_state.observers.end(), [&](const auto& entry) {
        return entry.first == handle;
    });
    log_state.observers.erase(it, log_state.observers.end());
}

const char* log_level_name(LogLevel level) {
    switch (level) {
    case LogLevel::Trace: return "trace";
    case LogLevel::Debug: return "debug";
    case LogLevel::Info: return "info";
    case LogLevel::Warn: return "warn";
    case LogLevel::Error: return "error";
    case LogLevel::Critical: return "critical";
    case LogLevel::Off: return "off";
    }
    return "info";
}

LogLevel parse_log_level(std::string_view name, LogLevel fallback) {
    const std::string value = lowercase(name);
    if (value == "trace") return LogLevel::Trace;
    if (value == "debug") return LogLevel::Debug;
    if (value == "info") return LogLevel::Info;
    if (value == "warn" || value == "warning") return LogLevel::Warn;
    if (value == "error" || value == "err") return LogLevel::Error;
    if (value == "critical" || value == "fatal") return LogLevel::Critical;
    if (value == "off" || value == "none") return LogLevel::Off;
    return fallback;
}

bool should_log(LogLevel level) {
    const int active = active_level_storage().load(std::memory_order_relaxed);
    return level != LogLevel::Off &&
        active != static_cast<int>(LogLevel::Off) &&
        static_cast<int>(level) >= active;
}

void log_message(LogLevel level, const char* file, int line, const char* function, std::string message) {
    if (!should_log(level)) {
        return;
    }
    if (std::shared_ptr<spdlog::logger> logger = current_logger()) {
        logger->log(
            spdlog::source_loc{file, line, function},
            to_spdlog(level),
            spdlog::string_view_t(message.data(), message.size()));
    }
    notify_observers(level, file, line, function, message);
}

} // namespace lt
