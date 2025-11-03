#include "camera/camera_capture.h"
#include "core/logger.h"
#include <chrono>

namespace endorobo {

CameraCapture::CameraCapture(const CameraConfig& config)
    : config_(config)
    , is_running_(false)
    , fps_(0.0)
    , dropped_frames_(0)
    , frame_count_for_fps_(0) {
}

CameraCapture::~CameraCapture() {
    stopCapture();
}

bool CameraCapture::initialize() {
    LOG_INFO("Initializing camera with ID: ", config_.camera_id);
    
    capture_ = std::make_unique<cv::VideoCapture>(config_.camera_id);
    
    if (!capture_->isOpened()) {
        LOG_ERROR("Failed to open camera with ID: ", config_.camera_id);
        return false;
    }
    
    // 设置相机参数
    capture_->set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
    capture_->set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    capture_->set(cv::CAP_PROP_FPS, config_.fps);
    
    // 验证设置
    int actual_width = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = capture_->get(cv::CAP_PROP_FPS);
    
    LOG_INFO("Camera initialized: ", actual_width, "x", actual_height, " @ ", actual_fps, " FPS");
    
    if (actual_width != config_.width || actual_height != config_.height) {
        LOG_WARNING("Camera resolution differs from config: ",
                   actual_width, "x", actual_height, " vs ",
                   config_.width, "x", config_.height);
    }
    
    return true;
}

bool CameraCapture::startCapture() {
    if (is_running_) {
        LOG_WARNING("Camera capture is already running");
        return false;
    }
    
    if (!capture_ || !capture_->isOpened()) {
        LOG_ERROR("Camera not initialized");
        return false;
    }
    
    is_running_ = true;
    last_fps_update_ = std::chrono::steady_clock::now();
    frame_count_for_fps_ = 0;
    dropped_frames_ = 0;
    
    capture_thread_ = std::thread(&CameraCapture::captureLoop, this);
    
    LOG_INFO("Camera capture started");
    return true;
}

void CameraCapture::stopCapture() {
    if (!is_running_) {
        return;
    }
    
    LOG_INFO("Stopping camera capture...");
    is_running_ = false;
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    LOG_INFO("Camera capture stopped. Dropped frames: ", dropped_frames_.load());
}

void CameraCapture::captureLoop() {
    LOG_INFO("Capture loop started");
    
    cv::Mat frame;
    
    while (is_running_) {
        if (!capture_->read(frame)) {
            LOG_ERROR("Failed to read frame from camera");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (frame.empty()) {
            LOG_WARNING("Empty frame received");
            continue;
        }
        
        // 更新最新帧
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame.copyTo(latest_frame_);
        }
        
        // 更新FPS
        frame_count_for_fps_++;
        updateFPS();
    }
    
    LOG_INFO("Capture loop ended");
}

bool CameraCapture::getLatestFrame(cv::Mat& frame, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (!latest_frame_.empty()) {
                latest_frame_.copyTo(frame);
                return true;
            }
        }
        
        // 检查超时
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            LOG_WARNING("Timeout waiting for frame");
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void CameraCapture::updateFPS() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_update_);
    
    // 每秒更新一次FPS
    if (elapsed.count() >= 1000) {
        fps_ = frame_count_for_fps_ * 1000.0 / elapsed.count();
        frame_count_for_fps_ = 0;
        last_fps_update_ = now;
    }
}

} // namespace endorobo

