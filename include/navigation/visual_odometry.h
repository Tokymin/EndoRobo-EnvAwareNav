#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <deque>

namespace endorobo {

/**
 * @brief Camera pose (position and orientation)
 */
struct CameraPose {
    Eigen::Vector3d position;      // Position (x, y, z) in meters
    Eigen::Matrix3d rotation;      // Rotation matrix (3x3)
    double timestamp;              // Timestamp in seconds
    bool valid;                    // Whether this pose is valid
    
    CameraPose() 
        : position(Eigen::Vector3d::Zero())
        , rotation(Eigen::Matrix3d::Identity())
        , timestamp(0.0)
        , valid(false) {}
    
    // Get Euler angles (roll, pitch, yaw) in degrees
    Eigen::Vector3d getEulerAngles() const;
    
    // Get 4x4 transformation matrix
    Eigen::Matrix4d getTransformMatrix() const;
};

/**
 * @brief Visual Odometry - estimates camera motion from image sequences
 * 
 * This class implements a simple but effective visual odometry system:
 * 1. Detect feature points in the image
 * 2. Track features across frames using optical flow
 * 3. Combine with depth information to get 3D points
 * 4. Estimate camera motion from 3D-3D point correspondences
 */
class VisualOdometry {
public:
    /**
     * @brief Configuration for visual odometry
     */
    struct Config {
        // Feature detection
        int max_features = 500;           // Maximum number of features to track
        double feature_quality = 0.01;    // Quality level for feature detection
        double min_feature_distance = 10; // Minimum distance between features
        
        // Optical flow
        int flow_window_size = 21;        // Window size for optical flow
        int flow_max_level = 3;           // Maximum pyramid level
        
        // Motion estimation
        double min_depth = 0.1;           // Minimum valid depth (meters)
        double max_depth = 10.0;          // Maximum valid depth (meters)
        int min_inliers = 30;             // Minimum inliers for valid motion
        double ransac_threshold = 0.01;   // RANSAC threshold (meters)
        
        // Trajectory
        int max_trajectory_length = 1000; // Maximum trajectory history
        
        Config() {}
    };
    
    VisualOdometry(const Config& config = Config());
    ~VisualOdometry();
    
    /**
     * @brief Initialize visual odometry with camera intrinsics
     */
    bool initialize(const cv::Mat& camera_matrix);
    
    /**
     * @brief Process a new frame and update pose
     * 
     * @param image Current RGB image
     * @param depth_map Corresponding depth map (same size as image)
     * @param pose Output: estimated camera pose
     * @return true if pose estimation successful
     */
    bool processFrame(const cv::Mat& image, const cv::Mat& depth_map, CameraPose& pose);
    
    /**
     * @brief Reset odometry (clear trajectory and features)
     */
    void reset();
    
    /**
     * @brief Get current camera pose
     */
    CameraPose getCurrentPose() const { return current_pose_; }
    
    /**
     * @brief Get full trajectory (all historical poses)
     */
    const std::deque<CameraPose>& getTrajectory() const { return trajectory_; }
    
    /**
     * @brief Get distance traveled (in meters)
     */
    double getDistanceTraveled() const { return distance_traveled_; }
    
    /**
     * @brief Draw tracked features on image (for visualization)
     */
    cv::Mat drawFeatures(const cv::Mat& image) const;
    
    /**
     * @brief Draw trajectory on a top-down view
     */
    cv::Mat drawTrajectory(int width = 800, int height = 600, double scale = 100.0) const;
    
    /**
     * @brief Get statistics
     */
    int getTrackedFeatureCount() const { return static_cast<int>(current_points_.size()); }
    int getInlierCount() const { return last_inlier_count_; }
    
private:
    // Internal methods
    void detectFeatures(const cv::Mat& image);
    bool trackFeatures(const cv::Mat& prev_image, const cv::Mat& curr_image);
    bool estimateMotion(const cv::Mat& depth_map, Eigen::Matrix4d& transform);
    void updatePose(const Eigen::Matrix4d& relative_transform);
    
    // Configuration
    Config config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // State
    bool initialized_;
    int frame_count_;
    
    // Feature tracking
    cv::Mat previous_image_;
    std::vector<cv::Point2f> previous_points_;
    std::vector<cv::Point2f> current_points_;
    std::vector<uchar> tracking_status_;
    
    // 3D points
    std::vector<cv::Point3d> previous_points_3d_;
    std::vector<cv::Point3d> current_points_3d_;
    
    // Pose
    CameraPose current_pose_;
    std::deque<CameraPose> trajectory_;
    double distance_traveled_;
    
    // Statistics
    int last_inlier_count_;
};

} // namespace endorobo

