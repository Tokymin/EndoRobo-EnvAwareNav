#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include "core/config_manager.h"

namespace endorobo {

/**
 * @brief 相机采集类
 * 负责实时采集摄像头图像数据
 */
class CameraCapture {
public:
    /**
     * @brief 构造函数
     * @param config 相机配置
     */
    explicit CameraCapture(const CameraConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~CameraCapture();
    
    /**
     * @brief 初始化相机
     * @return 初始化是否成功
     */
    bool initialize();
    
    /**
     * @brief 开始采集
     * @return 是否成功开始采集
     */
    bool startCapture();
    
    /**
     * @brief 停止采集
     */
    void stopCapture();
    
    /**
     * @brief 获取最新帧
     * @param frame 输出帧
     * @param timeout_ms 超时时间（毫秒）
     * @return 是否成功获取帧
     */
    bool getLatestFrame(cv::Mat& frame, int timeout_ms = 1000);
    
    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return is_running_; }
    
    /**
     * @brief 获取帧率统计
     */
    double getFPS() const { return fps_; }
    
    /**
     * @brief 获取丢帧数
     */
    int getDroppedFrames() const { return dropped_frames_; }
    
private:
    /**
     * @brief 采集线程函数
     */
    void captureLoop();
    
    /**
     * @brief 更新FPS统计
     */
    void updateFPS();
    
    CameraConfig config_;
    std::unique_ptr<cv::VideoCapture> capture_;
    
    std::atomic<bool> is_running_;
    std::thread capture_thread_;
    
    // 帧缓冲
    cv::Mat latest_frame_;
    std::mutex frame_mutex_;
    
    // 性能统计
    std::atomic<double> fps_;
    std::atomic<int> dropped_frames_;
    std::chrono::steady_clock::time_point last_fps_update_;
    int frame_count_for_fps_;
};

} // namespace endorobo

