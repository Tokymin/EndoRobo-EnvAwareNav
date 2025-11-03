#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "python_interface/python_wrapper.h"
#include "core/config_manager.h"

namespace endorobo {

/**
 * @brief 特征点结构
 */
struct FeaturePoint {
    cv::Point2f pixel;           // 像素坐标
    Eigen::Vector3d point_3d;    // 3D坐标
    float depth;                 // 深度值
    float confidence;            // 置信度
    
    FeaturePoint() : depth(0.0f), confidence(0.0f) {}
};

/**
 * @brief 深度估计结果
 */
struct DepthEstimation {
    cv::Mat depth_map;                      // 深度图
    std::vector<FeaturePoint> features;     // 特征点
    bool valid;                             // 是否有效
    
    DepthEstimation() : valid(false) {}
};

/**
 * @brief 深度估计器类
 * 通过调用Python深度学习模型进行单目深度估计
 */
class DepthEstimator {
public:
    /**
     * @brief 构造函数
     * @param config 模型配置
     * @param camera_config 相机配置
     */
    DepthEstimator(const PythonModelConfig& config,
                   const CameraConfig& camera_config);
    
    /**
     * @brief 析构函数
     */
    ~DepthEstimator();
    
    /**
     * @brief 初始化深度估计器
     * @param python_wrapper Python包装器
     * @return 是否初始化成功
     */
    bool initialize(std::shared_ptr<PythonWrapper> python_wrapper);
    
    /**
     * @brief 估计深度
     * @param frame 输入帧
     * @param depth 输出深度估计结果
     * @return 是否估计成功
     */
    bool estimateDepth(const cv::Mat& frame, DepthEstimation& depth);
    
    /**
     * @brief 从深度图提取特征点
     * @param depth_map 深度图
     * @param rgb_image RGB图像
     * @param features 输出特征点
     * @param num_features 提取的特征点数量
     * @return 是否提取成功
     */
    bool extractFeatures(const cv::Mat& depth_map,
                        const cv::Mat& rgb_image,
                        std::vector<FeaturePoint>& features,
                        int num_features = 1000);
    
    /**
     * @brief 将2D像素点和深度转换为3D点
     * @param pixel 像素坐标
     * @param depth 深度值
     * @return 3D点坐标
     */
    Eigen::Vector3d pixelToPoint3D(const cv::Point2f& pixel, float depth);
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
private:
    PythonModelConfig config_;
    CameraConfig camera_config_;
    std::shared_ptr<PythonWrapper> python_wrapper_;
    
    PyObject* model_module_;
    PyObject* predict_func_;
    
    bool initialized_;
    
    /**
     * @brief 预处理图像
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief 后处理深度图
     */
    cv::Mat postprocessDepth(const cv::Mat& depth_map);
};

} // namespace endorobo

