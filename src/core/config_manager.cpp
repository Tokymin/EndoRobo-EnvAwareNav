#include "core/config_manager.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

namespace endorobo {

cv::Mat CameraConfig::getCameraMatrix() const {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1);
    return camera_matrix;
}

cv::Mat CameraConfig::getDistortionCoeffs() const {
    cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    return dist_coeffs;
}

ConfigManager::ConfigManager(const std::string& config_file)
    : config_file_(config_file) {
}

ConfigManager::~ConfigManager() {
}

bool ConfigManager::loadConfig() {
    try {
        YAML::Node config = YAML::LoadFile(config_file_);
        
        // 加载相机配置
        if (config["camera"]) {
            auto cam = config["camera"];
            camera_config_.width = cam["width"].as<int>();
            camera_config_.height = cam["height"].as<int>();
            camera_config_.fps = cam["fps"].as<int>();
            camera_config_.camera_id = cam["camera_id"].as<int>();
            
            auto intrinsics = cam["intrinsics"];
            camera_config_.fx = intrinsics["fx"].as<double>();
            camera_config_.fy = intrinsics["fy"].as<double>();
            camera_config_.cx = intrinsics["cx"].as<double>();
            camera_config_.cy = intrinsics["cy"].as<double>();
            
            auto distortion = cam["distortion"];
            camera_config_.k1 = distortion["k1"].as<double>();
            camera_config_.k2 = distortion["k2"].as<double>();
            camera_config_.k3 = distortion["k3"].as<double>();
            camera_config_.p1 = distortion["p1"].as<double>();
            camera_config_.p2 = distortion["p2"].as<double>();
        }
        
        // 加载预处理配置
        if (config["preprocessing"]) {
            auto prep = config["preprocessing"];
            preprocessing_config_.undistort = prep["undistort"].as<bool>();
            preprocessing_config_.histogram_equalization = prep["histogram_equalization"].as<bool>();
            preprocessing_config_.scale_factor = prep["scale_factor"].as<double>();
        }
        
        // 加载Python模型配置
        if (config["python_models"]) {
            auto models = config["python_models"];
            
            // 位姿估计模型
            if (models["pose_estimation"]) {
                auto pose = models["pose_estimation"];
                pose_model_config_.model_path = pose["model_path"].as<std::string>();
                auto size = pose["input_size"];
                pose_model_config_.input_size = cv::Size(size[0].as<int>(), size[1].as<int>());
                pose_model_config_.use_gpu = pose["use_gpu"].as<bool>();
                pose_model_config_.min_depth = 0.0;
                pose_model_config_.max_depth = 0.0;
            }
            
            // 深度估计模型
            if (models["depth_estimation"]) {
                auto depth = models["depth_estimation"];
                depth_model_config_.model_path = depth["model_path"].as<std::string>();
                auto size = depth["input_size"];
                depth_model_config_.input_size = cv::Size(size[0].as<int>(), size[1].as<int>());
                depth_model_config_.use_gpu = depth["use_gpu"].as<bool>();
                depth_model_config_.min_depth = depth["min_depth"].as<double>();
                depth_model_config_.max_depth = depth["max_depth"].as<double>();
            }
        }
        
        // 加载3D重建配置
        if (config["reconstruction"]) {
            auto recon = config["reconstruction"];
            reconstruction_config_.voxel_size = recon["voxel_size"].as<double>();
            
            // 肠腔特定配置
            if (recon["intestinal"]) {
                auto intestinal = recon["intestinal"];
                reconstruction_config_.intestinal.min_radius = intestinal["min_radius"].as<double>();
                reconstruction_config_.intestinal.max_radius = intestinal["max_radius"].as<double>();
                reconstruction_config_.intestinal.smoothing_iterations = intestinal["smoothing_iterations"].as<int>();
            }
            
            // 冗余点去除配置
            if (recon["redundancy_removal"]) {
                auto redundancy = recon["redundancy_removal"];
                reconstruction_config_.redundancy_removal.distance_threshold = redundancy["distance_threshold"].as<double>();
                reconstruction_config_.redundancy_removal.normal_angle_threshold = redundancy["normal_angle_threshold"].as<double>();
                
                auto statistical = redundancy["statistical_outlier"];
                reconstruction_config_.redundancy_removal.nb_neighbors = statistical["nb_neighbors"].as<int>();
                reconstruction_config_.redundancy_removal.std_ratio = statistical["std_ratio"].as<double>();
            }
        }
        
        // 加载性能配置
        if (config["performance"]) {
            auto perf = config["performance"];
            performance_config_.max_queue_size = perf["max_queue_size"].as<int>();
            performance_config_.enable_multithreading = perf["enable_multithreading"].as<bool>();
            performance_config_.num_threads = perf["num_threads"].as<int>();
        }
        
        // 加载可视化配置
        if (config["visualization"]) {
            auto vis = config["visualization"];
            visualization_config_.show_camera_feed = vis["show_camera_feed"].as<bool>();
            visualization_config_.show_3d_reconstruction = vis["show_3d_reconstruction"].as<bool>();
            visualization_config_.show_depth_map = vis["show_depth_map"].as<bool>();
            visualization_config_.window_width = vis["window_width"].as<int>();
            visualization_config_.window_height = vis["window_height"].as<int>();
        }
        
        std::cout << "[ConfigManager] Configuration loaded successfully from: " << config_file_ << std::endl;
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "[ConfigManager] Error loading configuration: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::reloadConfig() {
    return loadConfig();
}

} // namespace endorobo

