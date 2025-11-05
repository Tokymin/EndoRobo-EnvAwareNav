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
#include "navigation/visual_odometry.h"
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
        , frame_count_(0)
        , depth_running_(false)
        , depth_frame_ready_(false) {
    }
    
    ~EndoRoboApp() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("========================================");
        LOG_INFO("EndoRobo Environment-Aware Navigation");
        LOG_INFO("========================================");
        
        // Load configuration
        if (!config_manager_.loadConfig()) {
            LOG_ERROR("Failed to load configuration");
            return false;
        }
        
        // Initialize Python interpreter
        python_wrapper_ = std::make_shared<PythonWrapper>();
        if (!python_wrapper_->initialize()) {
            LOG_ERROR("Failed to initialize Python wrapper");
            return false;
        }
        
        // Initialize camera
        camera_ = std::make_unique<CameraCapture>(
            config_manager_.getCameraConfig());
        if (!camera_->initialize()) {
            LOG_ERROR("Failed to initialize camera");
            return false;
        }
        
        // Initialize image processor
        image_processor_ = std::make_unique<ImageProcessor>(
            config_manager_.getCameraConfig(),
            config_manager_.getPreprocessingConfig());
        if (!image_processor_->initialize()) {
            LOG_ERROR("Failed to initialize image processor");
            return false;
        }
        
        // Initialize pose estimator
        pose_estimator_ = std::make_unique<PoseEstimator>(
            config_manager_.getPoseModelConfig());
        if (!pose_estimator_->initialize(python_wrapper_)) {
            LOG_WARNING("Failed to initialize pose estimator (Python model may not be ready)");
            // Continue execution, pose estimation will be unavailable
        }
        
        // Initialize depth estimator
        depth_estimator_ = std::make_unique<DepthEstimator>(
            config_manager_.getDepthModelConfig(),
            config_manager_.getCameraConfig());
        if (!depth_estimator_->initialize(python_wrapper_)) {
            LOG_WARNING("Failed to initialize depth estimator (Python model may not be ready)");
            // Continue execution, depth estimation will be unavailable
        }
        
        // Initialize visual odometry
        visual_odometry_ = std::make_unique<VisualOdometry>();
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << 
            config_manager_.getCameraConfig().fx, 0, config_manager_.getCameraConfig().cx,
            0, config_manager_.getCameraConfig().fy, config_manager_.getCameraConfig().cy,
            0, 0, 1);
        if (!visual_odometry_->initialize(camera_matrix)) {
            LOG_WARNING("Failed to initialize visual odometry");
        } else {
            LOG_INFO("Visual odometry initialized successfully");
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
        
        // Start async depth estimation thread
        if (depth_estimator_ && depth_estimator_->isInitialized()) {
            depth_running_ = true;
            depth_thread_ = std::thread(&EndoRoboApp::depthEstimationLoop, this);
            LOG_INFO("Depth estimation thread started");
        }
        
        LOG_INFO("Application started");
        return true;
    }
    
    void stop() {
        if (!running_) return;
        
        LOG_INFO("Stopping application...");
        running_ = false;
        depth_running_ = false;
        
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        
        if (depth_thread_.joinable()) {
            depth_thread_.join();
        }
        
        camera_->stopCapture();
        
        LOG_INFO("Application stopped");
        LOG_INFO("Total frames processed: ", frame_count_);
    }
    
    void run() {
        const auto& vis_config = config_manager_.getVisualizationConfig();
        
        // Create and position windows in a row
        const int window_width = 640;
        const int window_height = 480;
        const int window_spacing = 10;
        
        if (vis_config.show_camera_feed) {
            cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
            cv::resizeWindow("Camera Feed", window_width, window_height);
            cv::moveWindow("Camera Feed", 0, 50);
        }
        if (vis_config.show_depth_map) {
            cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
            cv::resizeWindow("Depth Map", window_width, window_height);
            cv::moveWindow("Depth Map", window_width + window_spacing, 50);
        }
        // Always show trajectory window
        cv::namedWindow("Camera Trajectory", cv::WINDOW_NORMAL);
        cv::resizeWindow("Camera Trajectory", window_width, window_height);
        cv::moveWindow("Camera Trajectory", (window_width + window_spacing) * 2, 50);
        
        LOG_INFO("Press 'q' to quit, 's' to save reconstruction, 'r' to reset trajectory");
        
        while (running_) {
            // Display camera feed with tracked features
            if (vis_config.show_camera_feed) {
                cv::Mat display_frame;
                {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (!latest_frame_.empty()) {
                        latest_frame_.copyTo(display_frame);
                    }
                }
                
                if (!display_frame.empty()) {
                    // Draw tracked features
                    if (visual_odometry_) {
                        display_frame = visual_odometry_->drawFeatures(display_frame);
                    }
                    
                    // Add info text - Line 1: Frame and FPS (Green)
                    std::string info_line1 = "Frame: " + std::to_string(frame_count_) +
                                            " | FPS: " + std::to_string(static_cast<int>(camera_->getFPS()));
                    cv::putText(display_frame, info_line1, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    
                    // Add info text - Line 2: Features and Distance (Yellow)
                    if (visual_odometry_) {
                        std::string info_line2 = "Features: " + std::to_string(visual_odometry_->getTrackedFeatureCount()) +
                                                " | Distance: " + std::to_string(static_cast<int>(visual_odometry_->getDistanceTraveled() * 100)) + "cm";
                        cv::putText(display_frame, info_line2, cv::Point(10, 65),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                    }
                    
                    cv::imshow("Camera Feed", display_frame);
                }
            }
            
            // Display depth map
            if (vis_config.show_depth_map) {
                cv::Mat display_depth;
                {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (!latest_depth_.empty()) {
                        latest_depth_.copyTo(display_depth);
                    }
                }
                
                if (!display_depth.empty()) {
                    // Normalize depth map for display
                    cv::Mat depth_normalized;
                    cv::normalize(display_depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
                    cv::applyColorMap(depth_normalized, depth_normalized, cv::COLORMAP_JET);
                    cv::imshow("Depth Map", depth_normalized);
                }
            }
            
            // Display trajectory
            if (visual_odometry_) {
                cv::Mat trajectory_view = visual_odometry_->drawTrajectory(800, 600, 100.0);
                cv::imshow("Camera Trajectory", trajectory_view);
            }
            
            // Handle key press
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {  // ESC
                break;
            } else if (key == 's' || key == 'S') {
                saveReconstruction();
            } else if (key == 'r' || key == 'R') {
                if (visual_odometry_) {
                    visual_odometry_->reset();
                    LOG_INFO("Trajectory reset");
                }
            }
        }
        
        cv::destroyAllWindows();
    }
    
private:
    void depthEstimationLoop() {
        LOG_INFO("Depth estimation loop started");
        
        int depth_frame_count = 0;
        
        while (depth_running_) {
            cv::Mat frame_to_process;
            bool has_frame = false;
            
            // Check if there's a new frame to process
            {
                std::lock_guard<std::mutex> lock(depth_input_mutex_);
                if (depth_frame_ready_ && !depth_input_frame_.empty()) {
                    depth_input_frame_.copyTo(frame_to_process);
                    depth_frame_ready_ = false;
                    has_frame = true;
                }
            }
            
            if (has_frame) {
                // Execute depth estimation with GIL protection
                Timer depth_timer;
                depth_timer.start("depth_estimation");
                
                // Acquire Python GIL for thread-safe Python calls
                PyGILState_STATE gstate = PyGILState_Ensure();
                
                DepthEstimation depth;
                bool success = depth_estimator_->estimateDepth(frame_to_process, depth);
                
                // Release Python GIL
                PyGILState_Release(gstate);
                
                if (success) {
                    if (depth.valid && !depth.depth_map.empty()) {
                        // Update depth map for display
                        std::lock_guard<std::mutex> lock(display_mutex_);
                        depth.depth_map.copyTo(latest_depth_);
                        
                        depth_frame_count++;
                        double elapsed = depth_timer.stop("depth_estimation");
                        
                        // Log performance every 10 estimates
                        if (depth_frame_count % 10 == 0) {
                            LOG_INFO("Depth estimation: frame ", depth_frame_count, 
                                    " took ", elapsed, "ms");
                        }
                    }
                }
            } else {
                // No new frame, sleep briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        LOG_INFO("Depth estimation loop ended. Total depth frames: ", depth_frame_count);
    }
    
    void processingLoop() {
        LOG_INFO("Processing loop started");
        
        Timer frame_timer;
        cv::Mat previous_frame;
        
        while (running_) {
            frame_timer.start("total");
            
            // Get latest frame
            cv::Mat frame;
            if (!camera_->getLatestFrame(frame, 100)) {
                continue;
            }
            
            // Image preprocessing
            cv::Mat processed_frame;
            frame_timer.start("preprocessing");
            if (!image_processor_->process(frame, processed_frame)) {
                LOG_WARNING("Image processing failed");
                continue;
            }
            double preprocess_time = frame_timer.stop("preprocessing");
            
            // Update frame for display
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                processed_frame.copyTo(latest_frame_);
            }
            
            // Pose estimation
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
            
            // Submit depth estimation task asynchronously (every 10 frames)
            if (depth_running_ && frame_count_ % 10 == 0) {
                std::lock_guard<std::mutex> lock(depth_input_mutex_);
                // Only submit new frame if previous one is processed (avoid backlog)
                if (!depth_frame_ready_) {
                    processed_frame.copyTo(depth_input_frame_);
                    depth_frame_ready_ = true;
                }
            }
            
            // Visual odometry: estimate camera pose using depth map
            if (visual_odometry_) {
                cv::Mat depth_for_vo;
                {
                    std::lock_guard<std::mutex> lock(display_mutex_);
                    if (!latest_depth_.empty()) {
                        depth_for_vo = latest_depth_.clone();
                    }
                }
                
                if (!depth_for_vo.empty()) {
                    CameraPose current_pose;
                    visual_odometry_->processFrame(processed_frame, depth_for_vo, current_pose);
                }
            }
            
            // Point cloud construction (temporarily disabled)
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
            
            // Save previous frame
            processed_frame.copyTo(previous_frame);
            
            frame_count_++;
            
            double total_time = frame_timer.stop("total");
            
            // Periodically print performance stats
            if (frame_count_ % 30 == 0) {
                LOG_INFO("Frame ", frame_count_, " - Total: ", total_time, "ms | ",
                        "Preprocess: ", preprocess_time, "ms | ",
                        "Pose: ", pose_time, "ms");
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
    std::unique_ptr<VisualOdometry> visual_odometry_;
    // PCL-dependent members temporarily disabled
    // std::unique_ptr<PointCloudBuilder> point_cloud_builder_;
    // std::unique_ptr<IntestinalReconstructor> intestinal_reconstructor_;
    // std::unique_ptr<RedundancyRemover> redundancy_remover_;
    
    std::atomic<bool> running_;
    std::thread processing_thread_;
    std::atomic<int> frame_count_;
    
    // Async depth estimation
    std::thread depth_thread_;
    std::atomic<bool> depth_running_;
    std::mutex depth_input_mutex_;
    cv::Mat depth_input_frame_;
    bool depth_frame_ready_;
    
    // Data for display
    std::mutex display_mutex_;
    cv::Mat latest_frame_;
    cv::Mat latest_depth_;
};

int main(int argc, char** argv) {
    // Setup logging
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().setLogFile("endorobo.log");
    
    // Parse command line arguments
    std::string config_file = "config/camera_config.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    try {
        // Create application
        EndoRoboApp app(config_file);
        
        // Initialize
        if (!app.initialize()) {
            LOG_FATAL("Failed to initialize application");
            return -1;
        }
        
        // Start
        if (!app.start()) {
            LOG_FATAL("Failed to start application");
            return -1;
        }
        
        // Run main loop
        app.run();
        
        // Stop
        app.stop();
        
        LOG_INFO("Application exited successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_FATAL("Exception: ", e.what());
        return -1;
    }
}

