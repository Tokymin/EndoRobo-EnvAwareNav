#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include "python_interface/python_wrapper.h"
#include "core/config_manager.h"

namespace endorobo {

/**
 * @brief 位姿估计结果
 */
struct PoseEstimation {
    Eigen::Matrix4d transformation;  // 4x4变换矩阵
    Eigen::Vector3d translation;     // 平移向量
    Eigen::Quaterniond rotation;     // 旋转四元数
    double confidence;               // 置信度
    bool valid;                      // 是否有效
    
    PoseEstimation() : confidence(0.0), valid(false) {
        transformation = Eigen::Matrix4d::Identity();
        translation = Eigen::Vector3d::Zero();
        rotation = Eigen::Quaterniond::Identity();
    }
};

/**
 * @brief 位姿估计器类
 * 通过调用Python深度学习模型进行相机位姿估计
 */
class PoseEstimator {
public:
    /**
     * @brief 构造函数
     * @param config 模型配置
     */
    explicit PoseEstimator(const PythonModelConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~PoseEstimator();
    
    /**
     * @brief 初始化位姿估计器
     * @param python_wrapper Python包装器
     * @return 是否初始化成功
     */
    bool initialize(std::shared_ptr<PythonWrapper> python_wrapper);
    
    /**
     * @brief 估计位姿
     * @param current_frame 当前帧
     * @param previous_frame 前一帧（可选）
     * @param pose 输出位姿
     * @return 是否估计成功
     */
    bool estimatePose(const cv::Mat& current_frame,
                     const cv::Mat& previous_frame,
                     PoseEstimation& pose);
    
    /**
     * @brief 估计绝对位姿
     * @param frame 输入帧
     * @param pose 输出位姿
     * @return 是否估计成功
     */
    bool estimateAbsolutePose(const cv::Mat& frame, PoseEstimation& pose);
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
private:
    PythonModelConfig config_;
    std::shared_ptr<PythonWrapper> python_wrapper_;
    
    PyObject* model_module_;
    PyObject* predict_func_;
    
    bool initialized_;
    
    /**
     * @brief 预处理图像
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief 解析Python返回的位姿结果
     */
    bool parsePoseResult(PyObject* result, PoseEstimation& pose);
};

} // namespace endorobo

