#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <Eigen/Dense>

// Include logger.h FIRST to prevent any namespace pollution
#include "core/logger.h"

#include <opencv2/opencv.hpp>
#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros that conflict with std::min/max
#include <windows.h>
#endif

#include "core/config_manager.h"
#include "camera/camera_capture.h"
#include "camera/image_processor.h"
#include "python_interface/python_wrapper.h"
#include "python_interface/pose_estimator.h"
#include "python_interface/depth_estimator.h"
#include "navigation/visual_odometry.h"
#include "utils/timer.h"
// PCL integration - Stage 3: Complete integration
#include "reconstruction/point_cloud_builder.h"
#include "reconstruction/intestinal_reconstructor.h"
#include "reconstruction/redundancy_remover.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/common/common.h>

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
        , depth_frame_ready_(false)
        , visualization_running_(false)
        , visualization_cloud_(nullptr)
        , cloud_exported_(false)
        , keyframe_initialized_(false)
        , keyframe_min_translation_(0.01)
        , keyframe_min_rotation_deg_(5.0) {
        if (const char* env = std::getenv("KEYFRAME_MIN_TRANSLATION_M")) {
            try { keyframe_min_translation_ = std::stod(env); } catch (...) {}
        }
        if (const char* env = std::getenv("KEYFRAME_MIN_ROTATION_DEG")) {
            try { keyframe_min_rotation_deg_ = std::stod(env); } catch (...) {}
        }
    }
    
    ~EndoRoboApp() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("========================================");
        LOG_INFO("EndoRobo Environment-Aware Navigation");
        LOG_INFO("========================================");
        
        cloud_exported_.store(false);

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
        
        // PCL initialization - Stage 3: Complete integration
        point_cloud_builder_ = std::make_unique<PointCloudBuilder>();
        LOG_INFO("PointCloudBuilder initialized");
        
        intestinal_reconstructor_ = std::make_unique<IntestinalReconstructor>(
            config_manager_.getReconstructionConfig());
        
        if (intestinal_reconstructor_->initialize()) {
            LOG_INFO("IntestinalReconstructor initialized successfully");
        } else {
            LOG_WARNING("Failed to initialize IntestinalReconstructor");
        }
        
        redundancy_remover_ = std::make_unique<RedundancyRemover>(
            config_manager_.getReconstructionConfig().redundancy_removal);
        LOG_INFO("RedundancyRemover initialized");
        
        LOG_INFO("Stage 3: PCL integration complete - All reconstruction features ready");
        
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
        
        // Start PCL visualization thread
        if (intestinal_reconstructor_) {
            visualization_running_ = true;
            visualization_thread_ = std::thread(&EndoRoboApp::visualizationLoop, this);
            LOG_INFO("Point cloud visualization thread started");
        }
        
        LOG_INFO("Application started");
        return true;
    }
    
    void stop() {
        if (!running_) return;
        
        LOG_INFO("Stopping application...");
        running_ = false;
        depth_running_ = false;
        visualization_running_ = false;
        
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        
        if (depth_thread_.joinable()) {
            depth_thread_.join();
        }
        
        if (visualization_thread_.joinable()) {
            // Close PCL viewer window
            if (pcl_viewer_) {
                pcl_viewer_->close();
            }
            visualization_thread_.join();
            LOG_INFO("Visualization thread stopped");
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
    
    void visualizationLoop() {
        LOG_INFO("Point cloud visualization started");
        
        // Create PCL visualizer
        pcl_viewer_ = pcl::visualization::PCLVisualizer::Ptr(
            new pcl::visualization::PCLVisualizer("3D Reconstruction Viewer"));
        
        // Set background color
        pcl_viewer_->setBackgroundColor(0.0, 0.0, 0.0);
        
        // Add coordinate system
        pcl_viewer_->addCoordinateSystem(0.1);
        
        // Set camera position
        pcl_viewer_->initCameraParameters();
        pcl_viewer_->setCameraPosition(0, 0, -0.5, 0, -1, 0);
        
        bool cloud_added = false;
        int update_count = 0;
        
        while (visualization_running_ && !pcl_viewer_->wasStopped()) {
            // Update point cloud every 30 frames (approximately 1 second)
            if (update_count % 30 == 0) {
                PointCloudBuilder::PointCloudType::Ptr display_cloud;
                {
                    std::lock_guard<std::mutex> lock(cloud_mutex_);
                    display_cloud = visualization_cloud_;
                }

                if (display_cloud && !display_cloud->empty()) {
                    pcl::visualization::PointCloudColorHandlerRGBField<PointCloudBuilder::PointType> rgb(display_cloud);

                    if (!cloud_added) {
                        // First time adding cloud
                        if (rgb.isCapable()) {
                            pcl_viewer_->addPointCloud<PointCloudBuilder::PointType>(display_cloud, rgb, "reconstruction");
                        } else {
                            pcl::visualization::PointCloudColorHandlerCustom<PointCloudBuilder::PointType> single_color(
                                display_cloud, 255, 255, 255);
                            pcl_viewer_->addPointCloud<PointCloudBuilder::PointType>(display_cloud, single_color, "reconstruction");
                        }
                        pcl_viewer_->setPointCloudRenderingProperties(
                            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
                        cloud_added = true;

                        PointCloudBuilder::PointType min_pt;
                        PointCloudBuilder::PointType max_pt;
                        pcl::getMinMax3D(*display_cloud, min_pt, max_pt);
                        Eigen::Vector3f min_vec = min_pt.getVector3fMap();
                        Eigen::Vector3f max_vec = max_pt.getVector3fMap();
                        if (std::isfinite(min_vec.x()) && std::isfinite(min_vec.y()) && std::isfinite(min_vec.z()) &&
                            std::isfinite(max_vec.x()) && std::isfinite(max_vec.y()) && std::isfinite(max_vec.z())) {
                            Eigen::Vector3f center = 0.5f * (min_vec + max_vec);
                            float diagonal = (max_vec - min_vec).norm();
                            float distance = std::max(diagonal * 1.2f, 0.3f);
                            float far_clip = std::max(distance * 5.0f, 1.0f);

                            pcl_viewer_->setCameraPosition(
                                center.x(), center.y(), center.z() + distance,
                                center.x(), center.y(), center.z(),
                                0.0, -1.0, 0.0);
                            pcl_viewer_->setCameraClipDistances(0.01, far_clip);
                        } else {
                            // Use default camera position when bounds are invalid
                            pcl_viewer_->setCameraPosition(0, 0, 3, 0, 0, 0, 0, -1, 0);
                            pcl_viewer_->setCameraClipDistances(0.01, 100.0);
                            LOG_WARNING("Invalid point cloud bounds, using default camera position");
                        }

                        if (!pcl_viewer_->contains("axes")) {
                            pcl_viewer_->addCoordinateSystem(0.1, "axes");
                        }
                        LOG_INFO("Point cloud added to viewer (", display_cloud->size(), " points)");
                    } else {
                        // Update existing cloud
                        if (rgb.isCapable()) {
                            pcl_viewer_->updatePointCloud<PointCloudBuilder::PointType>(display_cloud, rgb, "reconstruction");
                        } else {
                            pcl::visualization::PointCloudColorHandlerCustom<PointCloudBuilder::PointType> single_color(
                                display_cloud, 255, 255, 255);
                            pcl_viewer_->updatePointCloud<PointCloudBuilder::PointType>(display_cloud, single_color, "reconstruction");
                        }
                    }

                    // Update title with point count
                    std::string title = "3D Reconstruction - " +
                                      std::to_string(display_cloud->size()) + " points";
                    pcl_viewer_->setWindowName(title);
                }
            }
            
            // Spin once to handle UI events
            pcl_viewer_->spinOnce(33);  // ~30 FPS
            update_count++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
        
        LOG_INFO("Point cloud visualization ended");
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
            // Temporarily disable Python-based pose estimator to focus on VO debugging
            pose.transformation = Eigen::Matrix4d::Identity();
            pose.valid = false;
            double pose_time = frame_timer.stop("pose_estimation");
            
            // Submit depth estimation task asynchronously (always keep latest frame ready)
            if (depth_running_) {
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
                    if (visual_odometry_->processFrame(processed_frame, depth_for_vo, current_pose) && current_pose.valid) {
                        pose.transformation = current_pose.getTransformMatrix();
                        pose.translation = current_pose.position;
                        pose.rotation = Eigen::Quaterniond(current_pose.rotation);
                        pose.valid = true;
                        
                        // Debug: Log pose information
                        if (frame_count_ % 30 == 0) {  // Log every 30 frames
                            LOG_INFO("Visual Odometry Pose - Position: (", 
                                current_pose.position.x(), ", ", current_pose.position.y(), ", ", current_pose.position.z(), ")");
                            LOG_INFO("Pose valid: ", pose.valid);
                        }
                    }
                }
            }
            
            // Point cloud construction (use latest depth map if available)
            double cloud_time = 0.0;
            cv::Mat depth_for_cloud;
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                if (!latest_depth_.empty()) {
                    depth_for_cloud = latest_depth_.clone();
                }
            }
            
            // PCL point cloud construction - Stage 3 enabled
            // Build point cloud even if pose is invalid (for visualization)
            if (!depth_for_cloud.empty() && point_cloud_builder_ && intestinal_reconstructor_) {
                frame_timer.start("point_cloud");
                const auto& cam_config = config_manager_.getCameraConfig();
                auto cloud = point_cloud_builder_->createPointCloud(
                    depth_for_cloud, processed_frame, pose,
                    cam_config.fx, cam_config.fy, cam_config.cx, cam_config.cy);
                if (cloud && !cloud->empty()) {
                    if (!pose.valid) {
                        LOG_WARNING("Skipping cloud accumulation due to invalid pose");
                    } else {
                        double keyframe_trans = 0.0;
                        double keyframe_rot = 0.0;
                        if (shouldAddKeyframe(pose, keyframe_trans, keyframe_rot)) {
                            intestinal_reconstructor_->addFrame(cloud, pose);
                        } else {
                            LOG_INFO("Skipping frame (not a keyframe) trans=", keyframe_trans,
                                     ", rot=", keyframe_rot);
                        }
                    }

                    if (visualization_running_) {
                        auto accumulated = intestinal_reconstructor_->getAccumulatedCloud();
                        if (accumulated && !accumulated->empty()) {
                            PointCloudBuilder::PointCloudType::Ptr display_cloud(
                                new PointCloudBuilder::PointCloudType());

                            if (accumulated->size() > 100000) {
                                pcl::VoxelGrid<PointCloudBuilder::PointType> vg;
                                vg.setInputCloud(accumulated);
                                const float leaf_size = 0.003f;  // 3mm voxel size
                                vg.setLeafSize(leaf_size, leaf_size, leaf_size);
                                vg.filter(*display_cloud);
                            } else {
                                *display_cloud = *accumulated;
                            }

                            PointCloudBuilder::PointCloudType::Ptr filtered_cloud(
                                new PointCloudBuilder::PointCloudType());
                            std::vector<int> valid_indices;
                            pcl::removeNaNFromPointCloud(*display_cloud, *filtered_cloud, valid_indices);

                            if (!filtered_cloud->empty()) {
                                exportPointCloudOnce(filtered_cloud); // Export the filtered cloud
                                
                                // Debug: Log point cloud bounds every 30 frames
                                if (frame_count_ % 30 == 0) {
                                    PointCloudBuilder::PointType min_pt, max_pt;
                                    pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);
                                    LOG_INFO("Point cloud bounds - Min: (", min_pt.x, ", ", min_pt.y, ", ", min_pt.z, 
                                            ") Max: (", max_pt.x, ", ", max_pt.y, ", ", max_pt.z, ")");
                                }
                                
                                std::lock_guard<std::mutex> lock(cloud_mutex_);
                                visualization_cloud_ = filtered_cloud;
                            }
                        }
                    }
                }
                cloud_time = frame_timer.stop("point_cloud");
            }
            
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
        // Stage 3: PCD file saving enabled
        if (!intestinal_reconstructor_) {
            LOG_WARNING("Reconstruction not initialized");
            return;
        }
        
        auto cloud = intestinal_reconstructor_->getProcessedCloud();
        if (!cloud || cloud->empty()) {
            LOG_WARNING("No reconstruction data to save");
            return;
        }
        
        std::string cloud_filename = "reconstruction_" + 
            std::to_string(std::time(nullptr)) + ".pcd";
        
        if (pcl::io::savePCDFileBinary(cloud_filename, *cloud) == 0) {
            LOG_INFO("Point cloud saved to: ", cloud_filename);
            LOG_INFO("Total points saved: ", cloud->size());
        } else {
            LOG_ERROR("Failed to save point cloud");
        }
    }
    
    void resetReconstruction() {
        // Stage 3: Reconstruction reset enabled
        if (!intestinal_reconstructor_) {
            LOG_WARNING("Reconstruction not initialized");
            return;
        }
        
        intestinal_reconstructor_->reset();
        LOG_INFO("Reconstruction reset");
    }
    
    ConfigManager config_manager_;
    
    std::shared_ptr<PythonWrapper> python_wrapper_;
    std::unique_ptr<CameraCapture> camera_;
    std::unique_ptr<ImageProcessor> image_processor_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    std::unique_ptr<DepthEstimator> depth_estimator_;
    std::unique_ptr<VisualOdometry> visual_odometry_;
    // PCL members - Stage 3: Complete integration
    std::unique_ptr<PointCloudBuilder> point_cloud_builder_;
    std::unique_ptr<IntestinalReconstructor> intestinal_reconstructor_;
    std::unique_ptr<RedundancyRemover> redundancy_remover_;
    
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
    
    // PCL Visualization
    std::thread visualization_thread_;
    std::atomic<bool> visualization_running_;
    pcl::visualization::PCLVisualizer::Ptr pcl_viewer_;
    std::mutex cloud_mutex_;
    PointCloudBuilder::PointCloudType::Ptr visualization_cloud_;
    std::atomic<bool> cloud_exported_;
    bool keyframe_initialized_;
    double keyframe_min_translation_;
    double keyframe_min_rotation_deg_;
    Eigen::Matrix4d last_keyframe_transform_ = Eigen::Matrix4d::Identity();
    Eigen::Vector3d last_keyframe_translation_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond last_keyframe_rotation_ = Eigen::Quaterniond::Identity();

    bool shouldAddKeyframe(const PoseEstimation& pose, double& translation_delta, double& rotation_delta_deg) {
        translation_delta = 0.0;
        rotation_delta_deg = 0.0;
        if (!keyframe_initialized_) {
            last_keyframe_transform_ = pose.transformation;
            last_keyframe_translation_ = pose.translation;
            last_keyframe_rotation_ = pose.rotation;
            keyframe_initialized_ = true;
            return true;
        }
        translation_delta = (pose.translation - last_keyframe_translation_).norm();
        Eigen::Quaterniond delta_q = last_keyframe_rotation_.conjugate() * pose.rotation;
        delta_q.normalize();
        double angle_rad = 2.0 * std::acos(std::clamp(delta_q.w(), -1.0, 1.0));
        rotation_delta_deg = angle_rad * 180.0 / 3.14159265358979323846;
        if (translation_delta >= keyframe_min_translation_ || rotation_delta_deg >= keyframe_min_rotation_deg_) {
            last_keyframe_transform_ = pose.transformation;
            last_keyframe_translation_ = pose.translation;
            last_keyframe_rotation_ = pose.rotation;
            return true;
        }
        return false;
    }

    void exportPointCloudOnce(const PointCloudBuilder::PointCloudType::Ptr& cloud) {
        if (cloud_exported_.load()) return;
        if (!cloud || cloud->empty()) return;

        std::error_code ec;
        std::filesystem::create_directories("output", ec);
        if (ec) {
            LOG_WARNING("Failed to create output directory: ", ec.message());
            return;
        }

        const std::string file_path = "output/latest_cloud.pcd";
        int result = pcl::io::savePCDFileBinary(file_path, *cloud);
        if (result == 0) {
            LOG_INFO("Exported point cloud to ", file_path, " (", cloud->size(), " points)");
            cloud_exported_.store(true);
        } else {
            LOG_ERROR("Failed to export point cloud to ", file_path, ", error code: ", result);
        }
    }
};

int main(int argc, char** argv) {
    // Set console encoding to UTF-8 on Windows to avoid garbled characters
#ifdef _WIN32
    // Set console code page to UTF-8 (65001)
    // This ensures Chinese and other Unicode characters display correctly
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
    
    // Setup logging
    endorobo::Logger::getInstance().setLogLevel(endorobo::LogLevel::INFO);
    endorobo::Logger::getInstance().setLogFile("endorobo.log");
    
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

