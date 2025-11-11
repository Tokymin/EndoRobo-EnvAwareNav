#pragma once

// Include logger.h first to prevent PCL namespace pollution
#include "core/logger.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "python_interface/depth_estimator.h"
#include "python_interface/pose_estimator.h"

namespace endorobo {

/**
 * @brief 点云构建器类
 * 负责从深度图和位姿信息构建3D点云
 */
class PointCloudBuilder {
public:
    using PointType = pcl::PointXYZRGB;
    using PointCloudType = pcl::PointCloud<PointType>;
    
    /**
     * @brief 构造函数
     */
    PointCloudBuilder();
    
    /**
     * @brief 析构函数
     */
    ~PointCloudBuilder();
    
    /**
     * @brief 从深度图和RGB图像创建点云
     * @param depth_map 深度图
     * @param rgb_image RGB图像
     * @param pose 相机位姿
     * @param fx 焦距x
     * @param fy 焦距y
     * @param cx 主点x
     * @param cy 主点y
     * @return 点云指针
     */
    PointCloudType::Ptr createPointCloud(
        const cv::Mat& depth_map,
        const cv::Mat& rgb_image,
        const PoseEstimation& pose,
        double fx, double fy, double cx, double cy);
    
    /**
     * @brief 从特征点创建稀疏点云
     * @param features 特征点集合
     * @param pose 相机位姿
     * @return 点云指针
     */
    PointCloudType::Ptr createSparsePointCloud(
        const std::vector<FeaturePoint>& features,
        const PoseEstimation& pose);
    
    /**
     * @brief 合并点云
     * @param target 目标点云
     * @param source 源点云
     * @param transform 变换矩阵
     */
    void mergePointCloud(PointCloudType::Ptr target,
                        PointCloudType::Ptr source,
                        const Eigen::Matrix4d& transform);
    
    /**
     * @brief 体素滤波下采样
     * @param cloud 输入点云
     * @param leaf_size 体素大小
     * @return 下采样后的点云
     */
    PointCloudType::Ptr voxelDownsample(PointCloudType::Ptr cloud, double leaf_size);
    
private:
    /**
     * @brief 变换点到世界坐标系
     */
    Eigen::Vector3d transformPoint(const Eigen::Vector3d& point,
                                   const Eigen::Matrix4d& transform);
};

} // namespace endorobo

