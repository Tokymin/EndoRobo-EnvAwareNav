#include "core/logger.h"
#include <iostream>
#include <ctime>

namespace endorobo {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() 
    : current_level_(LogLevel::INFO), use_file_(false) {
}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::setLogLevel(LogLevel level) {
    current_level_ = level;
}

void Logger::setLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
        log_file_.close();
    }
    log_file_.open(filename, std::ios::app);
    use_file_ = log_file_.is_open();
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level < current_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "[" << getCurrentTime() << "] "
        << "[" << logLevelToString(level) << "] "
        << message;
    
    if (file != nullptr) {
        oss << " (" << file << ":" << line << ")";
    }
    
    std::string log_message = oss.str();
    
    // 输出到控制台
    if (level >= LogLevel::ERROR) {
        std::cerr << log_message << std::endl;
    } else {
        std::cout << log_message << std::endl;
    }
    
    // 输出到文件
    if (use_file_ && log_file_.is_open()) {
        log_file_ << log_message << std::endl;
        log_file_.flush();
    }
}

std::string Logger::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNOWN";
    }
}

} // namespace endorobo

