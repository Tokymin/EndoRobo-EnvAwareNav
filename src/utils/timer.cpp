#include "utils/timer.h"
#include "core/logger.h"

namespace endorobo {

Timer::Timer() {
}

void Timer::start(const std::string& name) {
    timers_[name] = std::chrono::steady_clock::now();
}

double Timer::stop(const std::string& name) {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        return 0.0;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - it->second);
    
    return duration.count() / 1000.0;  // 转换为毫秒
}

void Timer::reset(const std::string& name) {
    timers_.erase(name);
}

double Timer::elapsed(const std::string& name) const {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        return 0.0;
    }
    
    auto current_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        current_time - it->second);
    
    return duration.count() / 1000.0;  // 转换为毫秒
}

void Timer::print(const std::string& name, const std::string& prefix) const {
    double time_ms = elapsed(name);
    if (!prefix.empty()) {
        LOG_INFO(prefix, ": ", time_ms, " ms");
    } else {
        LOG_INFO("Timer [", name, "]: ", time_ms, " ms");
    }
}

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name)
    , start_time_(std::chrono::steady_clock::now()) {
}

ScopedTimer::~ScopedTimer() {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    double time_ms = duration.count() / 1000.0;
    
    LOG_INFO("[", name_, "] took ", time_ms, " ms");
}

} // namespace endorobo

