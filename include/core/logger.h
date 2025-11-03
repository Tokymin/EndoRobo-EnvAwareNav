#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace endorobo {

/**
 * @brief 日志级别
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

/**
 * @brief 简单的日志类
 */
class Logger {
public:
    /**
     * @brief 获取Logger单例
     */
    static Logger& getInstance();
    
    /**
     * @brief 设置日志级别
     */
    void setLogLevel(LogLevel level);
    
    /**
     * @brief 设置日志文件
     */
    void setLogFile(const std::string& filename);
    
    /**
     * @brief 记录日志
     */
    void log(LogLevel level, const std::string& message, 
             const char* file = nullptr, int line = 0);
    
    /**
     * @brief 便捷的日志宏
     */
    template<typename... Args>
    void debug(const Args&... args) {
        logImpl(LogLevel::DEBUG, args...);
    }
    
    template<typename... Args>
    void info(const Args&... args) {
        logImpl(LogLevel::INFO, args...);
    }
    
    template<typename... Args>
    void warning(const Args&... args) {
        logImpl(LogLevel::WARNING, args...);
    }
    
    template<typename... Args>
    void error(const Args&... args) {
        logImpl(LogLevel::ERROR, args...);
    }
    
    template<typename... Args>
    void fatal(const Args&... args) {
        logImpl(LogLevel::FATAL, args...);
    }
    
private:
    Logger();
    ~Logger();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    template<typename... Args>
    void logImpl(LogLevel level, const Args&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        log(level, oss.str());
    }
    
    std::string getCurrentTime();
    std::string logLevelToString(LogLevel level);
    
    LogLevel current_level_;
    std::ofstream log_file_;
    std::mutex mutex_;
    bool use_file_;
};

// 便捷的全局日志宏
#define LOG_DEBUG(...) endorobo::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) endorobo::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARNING(...) endorobo::Logger::getInstance().warning(__VA_ARGS__)
#define LOG_ERROR(...) endorobo::Logger::getInstance().error(__VA_ARGS__)
#define LOG_FATAL(...) endorobo::Logger::getInstance().fatal(__VA_ARGS__)

} // namespace endorobo

