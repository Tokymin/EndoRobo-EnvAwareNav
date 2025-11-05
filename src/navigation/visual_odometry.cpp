#include "navigation/visual_odometry.h"
#include "core/logger.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/SVD>
#include <cmath>
#include <iomanip>
#include <sstream>

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace endorobo {

// Helper function: Convert rotation matrix to Euler angles (ZYX convention)
Eigen::Vector3d CameraPose::getEulerAngles() const {
    Eigen::Vector3d euler;
    
    // Extract Euler angles from rotation matrix (ZYX convention)
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    euler(0) = std::atan2(rotation(2, 1), rotation(2, 2)) * 180.0 / M_PI;  // roll
    euler(1) = std::atan2(-rotation(2, 0), 
                          std::sqrt(rotation(2, 1) * rotation(2, 1) + 
                                   rotation(2, 2) * rotation(2, 2))) * 180.0 / M_PI;  // pitch
    euler(2) = std::atan2(rotation(1, 0), rotation(0, 0)) * 180.0 / M_PI;  // yaw
    
    return euler;
}

Eigen::Matrix4d CameraPose::getTransformMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = rotation;
    T.block<3, 1>(0, 3) = position;
    return T;
}

VisualOdometry::VisualOdometry(const Config& config)
    : config_(config)
    , initialized_(false)
    , frame_count_(0)
    , distance_traveled_(0.0)
    , last_inlier_count_(0) {
}

VisualOdometry::~VisualOdometry() {
}

bool VisualOdometry::initialize(const cv::Mat& camera_matrix) {
    if (camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        LOG_ERROR("Invalid camera matrix for visual odometry");
        return false;
    }
    
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);  // Assume no distortion (already corrected)
    
    // Initialize pose at origin
    current_pose_ = CameraPose();
    current_pose_.valid = true;
    trajectory_.clear();
    trajectory_.push_back(current_pose_);
    
    initialized_ = true;
    frame_count_ = 0;
    distance_traveled_ = 0.0;
    
    LOG_INFO("Visual odometry initialized successfully");
    return true;
}

bool VisualOdometry::processFrame(const cv::Mat& image, const cv::Mat& depth_map, CameraPose& pose) {
    if (!initialized_) {
        LOG_ERROR("Visual odometry not initialized");
        return false;
    }
    
    if (image.empty() || depth_map.empty()) {
        LOG_WARNING("Empty image or depth map");
        return false;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image.clone();
    }
    
    frame_count_++;
    
    // First frame: just detect features
    if (frame_count_ == 1) {
        detectFeatures(gray_image);
        previous_image_ = gray_image.clone();
        pose = current_pose_;
        return true;
    }
    
    // Track features from previous frame
    bool tracking_success = trackFeatures(previous_image_, gray_image);
    
    if (!tracking_success || current_points_.size() < config_.min_inliers) {
        LOG_WARNING("Feature tracking failed or too few features (", current_points_.size(), 
                   " < ", config_.min_inliers, "), re-detecting");
        detectFeatures(gray_image);
        previous_image_ = gray_image.clone();
        previous_points_ = current_points_;  // Save newly detected features for next frame!
        pose = current_pose_;
        return false;
    }
    
    LOG_INFO("Successfully tracked ", current_points_.size(), " features");
    
    // Estimate camera motion using 3D-3D correspondences
    Eigen::Matrix4d relative_transform;
    bool motion_estimated = estimateMotion(depth_map, relative_transform);
    
    if (motion_estimated) {
        // Update global pose
        updatePose(relative_transform);
        
        // Add to trajectory
        trajectory_.push_back(current_pose_);
        if (trajectory_.size() > config_.max_trajectory_length) {
            trajectory_.pop_front();
        }
        
        pose = current_pose_;
    } else {
        LOG_WARNING("Motion estimation failed");
        pose = current_pose_;
    }
    
    // Prepare for next frame
    previous_image_ = gray_image.clone();
    previous_points_ = current_points_;
    
    // Re-detect features if we have too few
    if (current_points_.size() < config_.max_features / 2) {
        detectFeatures(gray_image);
    }
    
    return motion_estimated;
}

void VisualOdometry::detectFeatures(const cv::Mat& image) {
    std::vector<cv::Point2f> corners;
    
    // Detect good features to track (Shi-Tomasi corners)
    cv::goodFeaturesToTrack(
        image,
        corners,
        config_.max_features,
        config_.feature_quality,
        config_.min_feature_distance
    );
    
    current_points_ = corners;
    LOG_INFO("Detected ", corners.size(), " features");
}

bool VisualOdometry::trackFeatures(const cv::Mat& prev_image, const cv::Mat& curr_image) {
    if (previous_points_.empty()) {
        LOG_WARNING("No previous points for tracking");
        return false;
    }
    
    std::vector<float> err;
    
    // Calculate optical flow (Lucas-Kanade)
    cv::calcOpticalFlowPyrLK(
        prev_image,
        curr_image,
        previous_points_,
        current_points_,
        tracking_status_,
        err,
        cv::Size(config_.flow_window_size, config_.flow_window_size),
        config_.flow_max_level
    );
    
    // Filter out bad matches
    std::vector<cv::Point2f> good_prev, good_curr;
    for (size_t i = 0; i < tracking_status_.size(); i++) {
        if (tracking_status_[i]) {
            good_prev.push_back(previous_points_[i]);
            good_curr.push_back(current_points_[i]);
        }
    }
    
    LOG_INFO("Optical flow: ", previous_points_.size(), " -> ", good_curr.size(), " tracked");
    
    previous_points_ = good_prev;
    current_points_ = good_curr;
    
    return !current_points_.empty();
}

bool VisualOdometry::estimateMotion(const cv::Mat& depth_map, Eigen::Matrix4d& transform) {
    if (previous_points_.size() < config_.min_inliers) {
        return false;
    }
    
    // Get camera parameters
    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    
    // Convert 2D points to 3D using depth
    previous_points_3d_.clear();
    current_points_3d_.clear();
    
    for (size_t i = 0; i < previous_points_.size(); i++) {
        // Get depth at current point
        int x_curr = static_cast<int>(current_points_[i].x);
        int y_curr = static_cast<int>(current_points_[i].y);
        
        if (x_curr < 0 || x_curr >= depth_map.cols || y_curr < 0 || y_curr >= depth_map.rows) {
            continue;
        }
        
        // Depth map is CV_8U (0-255), normalize to meters (assuming max depth = 10m)
        float depth = static_cast<float>(depth_map.at<uint8_t>(y_curr, x_curr) / 255.0 * config_.max_depth);
        
        if (depth < config_.min_depth || depth > config_.max_depth) {
            continue;
        }
        
        // Unproject to 3D (current frame)
        cv::Point3d p3d_curr;
        p3d_curr.z = depth;
        p3d_curr.x = (current_points_[i].x - cx) * depth / fx;
        p3d_curr.y = (current_points_[i].y - cy) * depth / fy;
        
        // For previous frame, use the same depth (approximation)
        cv::Point3d p3d_prev;
        p3d_prev.z = depth;
        p3d_prev.x = (previous_points_[i].x - cx) * depth / fx;
        p3d_prev.y = (previous_points_[i].y - cy) * depth / fy;
        
        previous_points_3d_.push_back(p3d_prev);
        current_points_3d_.push_back(p3d_curr);
    }
    
    if (previous_points_3d_.size() < config_.min_inliers) {
        LOG_WARNING("Not enough 3D points: ", previous_points_3d_.size(), " < ", config_.min_inliers);
        return false;
    }
    
    LOG_INFO("Got ", previous_points_3d_.size(), " valid 3D correspondences");
    
    // Estimate 3D-3D transformation using SVD (Umeyama algorithm)
    // Convert to Eigen format
    Eigen::Matrix3Xd prev_pts(3, previous_points_3d_.size());
    Eigen::Matrix3Xd curr_pts(3, current_points_3d_.size());
    
    for (size_t i = 0; i < previous_points_3d_.size(); i++) {
        prev_pts.col(i) << previous_points_3d_[i].x, previous_points_3d_[i].y, previous_points_3d_[i].z;
        curr_pts.col(i) << current_points_3d_[i].x, current_points_3d_[i].y, current_points_3d_[i].z;
    }
    
    // Compute centroids
    Eigen::Vector3d centroid_prev = prev_pts.rowwise().mean();
    Eigen::Vector3d centroid_curr = curr_pts.rowwise().mean();
    
    // Center the point clouds
    Eigen::Matrix3Xd prev_centered = prev_pts.colwise() - centroid_prev;
    Eigen::Matrix3Xd curr_centered = curr_pts.colwise() - centroid_curr;
    
    // Compute covariance matrix
    Eigen::Matrix3d H = curr_centered * prev_centered.transpose();
    
    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Compute rotation
    Eigen::Matrix3d R = U * V.transpose();
    
    // Ensure proper rotation (det(R) = 1)
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = U * V.transpose();
    }
    
    // Compute translation
    Eigen::Vector3d t = centroid_curr - R * centroid_prev;
    
    // Build transformation matrix
    transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = t;
    
    last_inlier_count_ = static_cast<int>(previous_points_3d_.size());
    
    return true;
}

void VisualOdometry::updatePose(const Eigen::Matrix4d& relative_transform) {
    // Extract rotation and translation from relative transform
    Eigen::Matrix3d R_rel = relative_transform.block<3, 3>(0, 0);
    Eigen::Vector3d t_rel = relative_transform.block<3, 1>(0, 3);
    
    // Update global pose: T_new = T_old * T_relative
    Eigen::Matrix3d R_new = current_pose_.rotation * R_rel;
    Eigen::Vector3d t_new = current_pose_.rotation * t_rel + current_pose_.position;
    
    // Calculate distance traveled
    double step_distance = (t_new - current_pose_.position).norm();
    distance_traveled_ += step_distance;
    
    // Update pose
    current_pose_.rotation = R_new;
    current_pose_.position = t_new;
    current_pose_.timestamp = frame_count_;
    current_pose_.valid = true;
}

void VisualOdometry::reset() {
    current_pose_ = CameraPose();
    current_pose_.valid = true;
    trajectory_.clear();
    trajectory_.push_back(current_pose_);
    
    frame_count_ = 0;
    distance_traveled_ = 0.0;
    
    previous_points_.clear();
    current_points_.clear();
    previous_points_3d_.clear();
    current_points_3d_.clear();
    
    LOG_INFO("Visual odometry reset");
}

cv::Mat VisualOdometry::drawFeatures(const cv::Mat& image) const {
    cv::Mat output;
    if (image.channels() == 1) {
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    } else {
        output = image.clone();
    }
    
    // Draw current feature points
    for (const auto& pt : current_points_) {
        cv::circle(output, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // Draw optical flow vectors if we have previous points
    if (previous_points_.size() == current_points_.size()) {
        for (size_t i = 0; i < current_points_.size(); i++) {
            cv::line(output, previous_points_[i], current_points_[i], cv::Scalar(0, 255, 255), 1);
            cv::circle(output, current_points_[i], 3, cv::Scalar(0, 255, 0), -1);
        }
    }
    
    // Text info is now drawn in main.cpp to avoid overlap
    
    return output;
}

cv::Mat VisualOdometry::drawTrajectory(int width, int height, double scale) const {
    cv::Mat canvas = cv::Mat::zeros(height, width, CV_8UC3);
    
    if (trajectory_.size() < 2) {
        return canvas;
    }
    
    // Draw trajectory path
    for (size_t i = 1; i < trajectory_.size(); i++) {
        if (!trajectory_[i-1].valid || !trajectory_[i].valid) {
            continue;
        }
        
        // Convert 3D position to 2D canvas coordinates (top-down view: X-Z plane)
        int x1 = width / 2 + static_cast<int>(trajectory_[i-1].position.x() * scale);
        int y1 = height / 2 - static_cast<int>(trajectory_[i-1].position.z() * scale);  // Negate Z for screen coords
        int x2 = width / 2 + static_cast<int>(trajectory_[i].position.x() * scale);
        int y2 = height / 2 - static_cast<int>(trajectory_[i].position.z() * scale);
        
        // Color gradient: green (start) to red (current)
        double progress = static_cast<double>(i) / trajectory_.size();
        cv::Scalar color(0, 255 * (1 - progress), 255 * progress);
        
        cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    }
    
    // Draw current position
    const auto& curr = trajectory_.back();
    int x_curr = width / 2 + static_cast<int>(curr.position.x() * scale);
    int y_curr = height / 2 - static_cast<int>(curr.position.z() * scale);
    cv::circle(canvas, cv::Point(x_curr, y_curr), 5, cv::Scalar(0, 0, 255), -1);
    
    // Draw origin
    cv::circle(canvas, cv::Point(width / 2, height / 2), 5, cv::Scalar(0, 255, 0), -1);
    cv::putText(canvas, "START", cv::Point(width / 2 + 10, height / 2 + 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    
    // Add info text
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Position: (" << curr.position.x() << ", " << curr.position.y() << ", " << curr.position.z() << ")m";
    cv::putText(canvas, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    ss.str("");
    auto euler = curr.getEulerAngles();
    ss << "Orientation: R=" << euler(0) << " P=" << euler(1) << " Y=" << euler(2) << " deg";
    cv::putText(canvas, ss.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    ss.str("");
    ss << "Distance: " << distance_traveled_ << "m | Frames: " << frame_count_;
    cv::putText(canvas, ss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // Draw axis
    cv::line(canvas, cv::Point(width / 2, height / 2), cv::Point(width / 2 + 50, height / 2), cv::Scalar(0, 0, 255), 2);  // X-axis (red)
    cv::line(canvas, cv::Point(width / 2, height / 2), cv::Point(width / 2, height / 2 - 50), cv::Scalar(255, 0, 0), 2);  // Z-axis (blue)
    cv::putText(canvas, "X", cv::Point(width / 2 + 55, height / 2 + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(canvas, "Z", cv::Point(width / 2 + 5, height / 2 - 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    
    return canvas;
}

} // namespace endorobo

