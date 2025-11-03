#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace endorobo {

/**
 * @brief 相机配置结构
 */
struct CameraConfig {
    int width;
    int height;
    int fps;
    int camera_id;
    
    // 内参矩阵参数
    double fx, fy, cx, cy;
    
    // 畸变系数
    double k1, k2, k3, p1, p2;
    
    // 获取OpenCV格式的相机矩阵
    cv::Mat getCameraMatrix() const;
    
    // 获取OpenCV格式的畸变系数
    cv::Mat getDistortionCoeffs() const;
};

/**
 * @brief 图像预处理配置
 */
struct PreprocessingConfig {
    bool undistort;
    bool histogram_equalization;
    double scale_factor;
};

/**
 * @brief Python模型配置
 */
struct PythonModelConfig {
    std::string model_path;
    cv::Size input_size;
    bool use_gpu;
    
    // 深度估计特定参数
    double min_depth;
    double max_depth;
};

/**
 * @brief 肠腔重建配置
 */
struct IntestinalReconstructionConfig {
    double min_radius;
    double max_radius;
    int smoothing_iterations;
};

/**
 * @brief 冗余点去除配置
 */
struct RedundancyRemovalConfig {
    double distance_threshold;
    double normal_angle_threshold;
    int nb_neighbors;
    double std_ratio;
};

/**
 * @brief 3D重建配置
 */
struct ReconstructionConfig {
    double voxel_size;
    IntestinalReconstructionConfig intestinal;
    RedundancyRemovalConfig redundancy_removal;
};

/**
 * @brief 性能配置
 */
struct PerformanceConfig {
    int max_queue_size;
    bool enable_multithreading;
    int num_threads;
};

/**
 * @brief 可视化配置
 */
struct VisualizationConfig {
    bool show_camera_feed;
    bool show_3d_reconstruction;
    bool show_depth_map;
    int window_width;
    int window_height;
};

/**
 * @brief 配置管理器类
 * 负责加载和管理所有配置参数
 */
class ConfigManager {
public:
    /**
     * @brief 构造函数
     * @param config_file 配置文件路径
     */
    explicit ConfigManager(const std::string& config_file);
    
    /**
     * @brief 析构函数
     */
    ~ConfigManager();
    
    /**
     * @brief 加载配置文件
     * @return 加载是否成功
     */
    bool loadConfig();
    
    /**
     * @brief 重新加载配置文件
     * @return 重新加载是否成功
     */
    bool reloadConfig();
    
    // Getters
    const CameraConfig& getCameraConfig() const { return camera_config_; }
    const PreprocessingConfig& getPreprocessingConfig() const { return preprocessing_config_; }
    const PythonModelConfig& getPoseModelConfig() const { return pose_model_config_; }
    const PythonModelConfig& getDepthModelConfig() const { return depth_model_config_; }
    const ReconstructionConfig& getReconstructionConfig() const { return reconstruction_config_; }
    const PerformanceConfig& getPerformanceConfig() const { return performance_config_; }
    const VisualizationConfig& getVisualizationConfig() const { return visualization_config_; }
    
private:
    std::string config_file_;
    
    CameraConfig camera_config_;
    PreprocessingConfig preprocessing_config_;
    PythonModelConfig pose_model_config_;
    PythonModelConfig depth_model_config_;
    ReconstructionConfig reconstruction_config_;
    PerformanceConfig performance_config_;
    VisualizationConfig visualization_config_;
    
    /**
     * @brief 解析配置文件
     */
    void parseConfig();
};

} // namespace endorobo

