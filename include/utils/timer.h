#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace endorobo {

/**
 * @brief 高精度计时器类
 */
class Timer {
public:
    /**
     * @brief 构造函数
     */
    Timer();
    
    /**
     * @brief 开始计时
     * @param name 计时器名称
     */
    void start(const std::string& name = "default");
    
    /**
     * @brief 停止计时
     * @param name 计时器名称
     * @return 经过的时间（毫秒）
     */
    double stop(const std::string& name = "default");
    
    /**
     * @brief 重置计时器
     * @param name 计时器名称
     */
    void reset(const std::string& name = "default");
    
    /**
     * @brief 获取经过的时间（不停止计时器）
     * @param name 计时器名称
     * @return 经过的时间（毫秒）
     */
    double elapsed(const std::string& name = "default") const;
    
    /**
     * @brief 打印计时信息
     * @param name 计时器名称
     * @param prefix 前缀信息
     */
    void print(const std::string& name = "default", 
               const std::string& prefix = "") const;
    
private:
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> timers_;
};

/**
 * @brief RAII风格的作用域计时器
 */
class ScopedTimer {
public:
    /**
     * @brief 构造函数，开始计时
     * @param name 计时器名称
     */
    explicit ScopedTimer(const std::string& name);
    
    /**
     * @brief 析构函数，打印计时信息
     */
    ~ScopedTimer();
    
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace endorobo

