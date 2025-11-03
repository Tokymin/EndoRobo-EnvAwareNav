#pragma once

#include <opencv2/opencv.hpp>
#include "core/config_manager.h"

namespace endorobo {

/**
 * @brief 图像预处理类
 * 负责图像的畸变校正、直方图均衡化等预处理操作
 */
class ImageProcessor {
public:
    /**
     * @brief 构造函数
     * @param camera_config 相机配置
     * @param preprocessing_config 预处理配置
     */
    ImageProcessor(const CameraConfig& camera_config,
                   const PreprocessingConfig& preprocessing_config);
    
    /**
     * @brief 析构函数
     */
    ~ImageProcessor();
    
    /**
     * @brief 初始化处理器
     * @return 是否成功初始化
     */
    bool initialize();
    
    /**
     * @brief 处理图像
     * @param input 输入图像
     * @param output 输出图像
     * @return 是否成功处理
     */
    bool process(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief 畸变校正
     * @param input 输入图像
     * @param output 输出图像
     */
    void undistort(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief 直方图均衡化
     * @param input 输入图像
     * @param output 输出图像
     */
    void histogramEqualization(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief 图像缩放
     * @param input 输入图像
     * @param output 输出图像
     * @param scale_factor 缩放因子
     */
    void resize(const cv::Mat& input, cv::Mat& output, double scale_factor);
    
private:
    CameraConfig camera_config_;
    PreprocessingConfig preprocessing_config_;
    
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat map1_, map2_;  // 用于畸变校正的映射表
    
    bool initialized_;
};

} // namespace endorobo

