#include "camera/image_processor.h"
#include "core/logger.h"

namespace endorobo {

ImageProcessor::ImageProcessor(const CameraConfig& camera_config,
                               const PreprocessingConfig& preprocessing_config)
    : camera_config_(camera_config)
    , preprocessing_config_(preprocessing_config)
    , initialized_(false) {
}

ImageProcessor::~ImageProcessor() {
}

bool ImageProcessor::initialize() {
    LOG_INFO("Initializing image processor...");
    
    // 获取相机矩阵和畸变系数
    camera_matrix_ = camera_config_.getCameraMatrix();
    dist_coeffs_ = camera_config_.getDistortionCoeffs();
    
    // 如果启用畸变校正，预计算映射表
    if (preprocessing_config_.undistort) {
        cv::Size image_size(camera_config_.width, camera_config_.height);
        cv::initUndistortRectifyMap(
            camera_matrix_, dist_coeffs_,
            cv::Mat(), camera_matrix_, image_size,
            CV_32FC1, map1_, map2_);
        LOG_INFO("Undistortion maps initialized");
    }
    
    initialized_ = true;
    LOG_INFO("Image processor initialized successfully");
    return true;
}

bool ImageProcessor::process(const cv::Mat& input, cv::Mat& output) {
    if (!initialized_) {
        LOG_ERROR("Image processor not initialized");
        return false;
    }
    
    if (input.empty()) {
        LOG_ERROR("Input image is empty");
        return false;
    }
    
    cv::Mat temp = input.clone();
    
    // 畸变校正
    if (preprocessing_config_.undistort) {
        undistort(temp, output);
        temp = output.clone();
    }
    
    // 直方图均衡化
    if (preprocessing_config_.histogram_equalization) {
        histogramEqualization(temp, output);
        temp = output.clone();
    }
    
    // 图像缩放
    if (preprocessing_config_.scale_factor != 1.0) {
        resize(temp, output, preprocessing_config_.scale_factor);
    } else {
        output = temp;
    }
    
    return true;
}

void ImageProcessor::undistort(const cv::Mat& input, cv::Mat& output) {
    if (map1_.empty() || map2_.empty()) {
        cv::undistort(input, output, camera_matrix_, dist_coeffs_);
    } else {
        cv::remap(input, output, map1_, map2_, cv::INTER_LINEAR);
    }
}

void ImageProcessor::histogramEqualization(const cv::Mat& input, cv::Mat& output) {
    if (input.channels() == 1) {
        // 灰度图直接均衡化
        cv::equalizeHist(input, output);
    } else {
        // 彩色图转换到YCrCb空间，只对Y通道均衡化
        cv::Mat ycrcb;
        cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);
        
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        
        cv::equalizeHist(channels[0], channels[0]);
        
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);
    }
}

void ImageProcessor::resize(const cv::Mat& input, cv::Mat& output, double scale_factor) {
    if (scale_factor <= 0) {
        LOG_ERROR("Invalid scale factor: ", scale_factor);
        output = input.clone();
        return;
    }
    
    cv::Size new_size(
        static_cast<int>(input.cols * scale_factor),
        static_cast<int>(input.rows * scale_factor)
    );
    
    cv::resize(input, output, new_size, 0, 0, cv::INTER_LINEAR);
}

} // namespace endorobo

