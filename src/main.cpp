#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "core/config_manager.h"
#include "core/logger.h"
#include "camera/camera_capture.h"
#include "camera/image_processor.h"
#include "python_interface/python_wrapper.h"
#include "python_interface/pose_estimator.h"
#include "python_interface/depth_estimator.h"
// PCL-dependent headers temporarily disabled
// #include "reconstruction/point_cloud_builder.h"
// #include "reconstruction/intestinal_reconstructor.h"
// #include "reconstruction/redundancy_remover.h"
#include "utils/timer.h"

using namespace endorobo;

/**
 * @brief 主应用程序类
 */
class EndoRoboApp {
public:
    EndoRoboApp(const std::string& config_file)
        : config_manager_(config_file)
        , running_(false)
        , frame_count_(0) {
    }
    
    ~EndoRoboApp() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("========================================");
        LOG_INFO("EndoRobo Environment-Aware Navigation");
        LOG_INFO("========================================");
        
        // 加载配置
        if (!config_manager_.loadConfig()) {
            LOG_ERROR("Failed to load configuration");
            return false;
        }
        
        // 初始化Python解释器
        python_wrapper_ = std::make_shared<PythonWrapper>();
        if (!python_wrapper_->initialize()) {
            LOG_ERROR("Failed to initialize Python wrapper");
            return false;
        }
        
        // 初始化相机
        camera_ = std::make_unique<CameraCapture>(
            config_manager_.getCameraConfig());
        if (!camera_->initialize()) {
            LOG_ERROR("Failed to initialize camera");
            return false;
        }
        
        // 初始化图像处理器
        image_processor_ = std::make_unique<ImageProcessor>(
            config_manager_.getCameraConfig(),
            config_manager_.getPreprocessingConfig());
        if (!image_processor_->initialize()) {
            LOG_ERROR("Failed to initialize image processor");
            return false;
        }
        
        // 初始化位姿估计器
        pose_estimator_ = std::make_unique<PoseEstimator>(
            config_manager_.getPoseModelConfig());
        if (!pose_estimator_->initialize(python_wrapper_)) {
            LOG_WARNING("Failed to initialize pose estimator (Python model may not be ready)");
            // 继续运行，但位姿估计功能不可用
        }
        
        // 初始化深度估计器
        depth_estimator_ = std::make_unique<DepthEstimator>(
            config_manager_.getDepthModelConfig(),
            config_manager_.getCameraConfig());
        if (!depth_estimator_->initialize(python_wrapper_)) {
            LOG_WARNING("Failed to initialize depth estimator (Python model may not be ready)");
            // 继续运行，但深度估计功能不可用
        }
        
        // PCL-dependent initialization temporarily disabled
        // point_cloud_builder_ = std::make_unique<PointCloudBuilder>();
        // intestinal_reconstructor_ = std::make_unique<IntestinalReconstructor>(
        //     config_manager_.getReconstructionConfig());
        // redundancy_remover_ = std::make_unique<RedundancyRemover>(
        //     config_manager_.getReconstructionConfig().redundancy_removal);
        
        LOG_WARNING("3D reconstruction features disabled (PCL not available)");
        
        LOG_INFO("Application initialized successfully");
        return true;
    }
    
    bool start() {
        if (!camera_->startCapture()) {
            LOG_ERROR("Failed to start camera capture");
            return false;
        }
        
        running_ = true;
        processing_thread_ = std::thread(&EndoRoboApp::processingLoop, this);
        
        LOG_INFO("Application started");
        return true;
    }
    
    void stop() {
        if (!running_) return;
        
        LOG_INFO("Stopping application...");
        running_ = false;
        
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        
        camera_->stopCapture();
        
        LOG_INFO("Application stopped");
        LOG_INFO("Total frames processed: ", frame_count_);
    }
    
    void run() {
        const auto& vis_config = config_manager_.getVisualizationConfig();
        
        if (vis_config.show_camera_feed) {
            cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
        }
        if (vis_config.show_depth_map) {
            cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        }
        
        LOG_INFO("Press 'q' to quit, 's' to save reconstruction, 'r' to reset");
        
        while (running_) {
            // 显示相机画面
            if (vis_config.show_camera_feed) {
                cv::Mat display_frame;
                {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (!latest_frame_.empty()) {
                        latest_frame_.copyTo(display_frame);
                    }
                }
                
                if (!display_frame.empty()) {
                    // 添加信息文本
                    std::string info = "Frame: " + std::to_string(frame_count_) +
                                      " | FPS: " + std::to_string(static_cast<int>(camera_->getFPS()));
                    cv::putText(display_frame, info, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    
                    cv::imshow("Camera Feed", display_frame);
                }
            }
            
            // 显示深度图
            if (vis_config.show_depth_map) {
                cv::Mat display_depth;
                {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (!latest_depth_.empty()) {
                        latest_depth_.copyTo(display_depth);
                    }
                }
                
                if (!display_depth.empty()) {
                    // 归一化深度图用于显示
                    cv::Mat depth_normalized;
                    cv::normalize(display_depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
                    cv::applyColorMap(depth_normalized, depth_normalized, cv::COLORMAP_JET);
                    cv::imshow("Depth Map", depth_normalized);
                }
            }
            
            // 处理按键
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {  // ESC
                break;
            } else if (key == 's' || key == 'S') {
                saveReconstruction();
            } else if (key == 'r' || key == 'R') {
                resetReconstruction();
            }
        }
        
        cv::destroyAllWindows();
    }
    
private:
    void processingLoop() {
        LOG_INFO("Processing loop started");
        
        Timer frame_timer;
        cv::Mat previous_frame;
        
        while (running_) {
            frame_timer.start("total");
            
            // 获取最新帧
            cv::Mat frame;
            if (!camera_->getLatestFrame(frame, 100)) {
                continue;
            }
            
            // 图像预处理
            cv::Mat processed_frame;
            frame_timer.start("preprocessing");
            if (!image_processor_->process(frame, processed_frame)) {
                LOG_WARNING("Image processing failed");
                continue;
            }
            double preprocess_time = frame_timer.stop("preprocessing");
            
            // 更新显示用的帧
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                processed_frame.copyTo(latest_frame_);
            }
            
            // 位姿估计
            PoseEstimation pose;
            frame_timer.start("pose_estimation");
            if (pose_estimator_ && pose_estimator_->isInitialized()) {
                if (!pose_estimator_->estimatePose(processed_frame, previous_frame, pose)) {
                    LOG_DEBUG("Pose estimation failed, using identity");
                    pose.transformation = Eigen::Matrix4d::Identity();
                    pose.valid = false;
                }
            } else {
                pose.transformation = Eigen::Matrix4d::Identity();
                pose.valid = false;
            }
            double pose_time = frame_timer.stop("pose_estimation");
            
            // 深度估计（降低频率以提高响应速度 - 每10帧估计一次）
            DepthEstimation depth;
            double depth_time = 0.0;
            
            // 只在每10帧执行深度估计（CPU上很慢）
            if (frame_count_ % 10 == 0) {
                frame_timer.start("depth_estimation");
                if (depth_estimator_ && depth_estimator_->isInitialized()) {
                    if (depth_estimator_->estimateDepth(processed_frame, depth)) {
                        if (depth.valid && !depth.depth_map.empty()) {
                            // 更新显示用的深度图
                            std::lock_guard<std::mutex> lock(display_mutex_);
                            depth.depth_map.copyTo(latest_depth_);
                            
                            if (frame_count_ % 100 == 0) {
                                LOG_INFO("Depth map updated (", depth.depth_map.cols, "x", 
                                        depth.depth_map.rows, ")");
                            }
                        }
                    } else if (frame_count_ % 100 == 0) {
                        LOG_WARNING("Depth estimation failed");
                    }
                }
                depth_time = frame_timer.stop("depth_estimation");
            }
            
            // 点云构建 (temporarily disabled)
            double cloud_time = 0.0;
            // if (depth.valid && !depth.depth_map.empty()) {
            //     frame_timer.start("point_cloud");
            //     const auto& cam_config = config_manager_.getCameraConfig();
            //     auto cloud = point_cloud_builder_->createPointCloud(
            //         depth.depth_map, processed_frame, pose,
            //         cam_config.fx, cam_config.fy, cam_config.cx, cam_config.cy);
            //     if (cloud && !cloud->empty()) {
            //         intestinal_reconstructor_->addFrame(cloud, pose);
            //     }
            //     cloud_time = frame_timer.stop("point_cloud");
            // }
            
            // 更新显示
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                frame.copyTo(latest_frame_);
                if (depth.valid) {
                    depth.depth_map.copyTo(latest_depth_);
                }
            }
            
            // 保存前一帧
            processed_frame.copyTo(previous_frame);
            
            frame_count_++;
            
            double total_time = frame_timer.stop("total");
            
            // 定期打印性能统计
            if (frame_count_ % 30 == 0) {
                LOG_INFO("Frame ", frame_count_, " - Total: ", total_time, "ms | ",
                        "Preprocess: ", preprocess_time, "ms | ",
                        "Pose: ", pose_time, "ms | ",
                        "Depth: ", depth_time, "ms | ",
                        "Cloud: ", cloud_time, "ms");
            }
        }
        
        LOG_INFO("Processing loop ended");
    }
    
    void saveReconstruction() {
        LOG_WARNING("Reconstruction save disabled (PCL not available)");
        // auto cloud = intestinal_reconstructor_->getProcessedCloud();
        // if (!cloud || cloud->empty()) {
        //     LOG_WARNING("No reconstruction data to save");
        //     return;
        // }
        // std::string cloud_filename = "reconstruction_" + 
        //     std::to_string(std::time(nullptr)) + ".pcd";
        // pcl::io::savePCDFileBinary(cloud_filename, *cloud);
        // LOG_INFO("Point cloud saved to: ", cloud_filename);
    }
    
    void resetReconstruction() {
        LOG_WARNING("Reconstruction reset disabled (PCL not available)");
        // intestinal_reconstructor_->reset();
        // frame_count_ = 0;
    }
    
    ConfigManager config_manager_;
    
    std::shared_ptr<PythonWrapper> python_wrapper_;
    std::unique_ptr<CameraCapture> camera_;
    std::unique_ptr<ImageProcessor> image_processor_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    std::unique_ptr<DepthEstimator> depth_estimator_;
    // PCL-dependent members temporarily disabled
    // std::unique_ptr<PointCloudBuilder> point_cloud_builder_;
    // std::unique_ptr<IntestinalReconstructor> intestinal_reconstructor_;
    // std::unique_ptr<RedundancyRemover> redundancy_remover_;
    
    std::atomic<bool> running_;
    std::thread processing_thread_;
    std::atomic<int> frame_count_;
    
    // 显示用的数据
    std::mutex display_mutex_;
    cv::Mat latest_frame_;
    cv::Mat latest_depth_;
};

int main(int argc, char** argv) {
    // 设置日志
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().setLogFile("endorobo.log");
    
    // 解析命令行参数
    std::string config_file = "config/camera_config.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    try {
        // 创建应用程序
        EndoRoboApp app(config_file);
        
        // 初始化
        if (!app.initialize()) {
            LOG_FATAL("Failed to initialize application");
            return -1;
        }
        
        // 启动
        if (!app.start()) {
            LOG_FATAL("Failed to start application");
            return -1;
        }
        
        // 运行主循环
        app.run();
        
        // 停止
        app.stop();
        
        LOG_INFO("Application exited successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_FATAL("Exception: ", e.what());
        return -1;
    }
}

