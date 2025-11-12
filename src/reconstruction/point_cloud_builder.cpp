#include "reconstruction/point_cloud_builder.h"
#include "core/logger.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

namespace endorobo {

PointCloudBuilder::PointCloudBuilder() {
}

PointCloudBuilder::~PointCloudBuilder() {
}

PointCloudBuilder::PointCloudType::Ptr PointCloudBuilder::createPointCloud(
    const cv::Mat& depth_map,
    const cv::Mat& rgb_image,
    const PoseEstimation& pose,
    double fx, double fy, double cx, double cy) {
    
    PointCloudType::Ptr cloud(new PointCloudType());
    
    if (depth_map.empty() || rgb_image.empty()) {
        LOG_ERROR("Empty depth map or RGB image");
        return cloud;
    }
    
    if (depth_map.size() != rgb_image.size()) {
        LOG_ERROR("Depth map and RGB image size mismatch");
        return cloud;
    }
    
    int width = depth_map.cols;
    int height = depth_map.rows;
    
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(width * height);
    
    // 遍历每个像素
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            PointType& pt = cloud->at(u, v);
            
            float depth = depth_map.at<float>(v, u);
            
            // 跳过无效深度 - 添加合理范围检查
            if (depth <= 0.0f || depth > 10.0f || std::isnan(depth) || std::isinf(depth)) {
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // 反投影到相机坐标系
            Eigen::Vector3d point_camera;
            point_camera.x() = (u - cx) * depth / fx;
            point_camera.y() = (v - cy) * depth / fy;
            point_camera.z() = depth;
            
            // 变换到世界坐标系
            Eigen::Vector3d point_world = transformPoint(point_camera, pose.transformation);
            
            // 检查变换后的坐标是否有效
            if (std::isnan(point_world.x()) || std::isnan(point_world.y()) || std::isnan(point_world.z()) ||
                std::isinf(point_world.x()) || std::isinf(point_world.y()) || std::isinf(point_world.z()) ||
                std::abs(point_world.x()) > 100.0 || std::abs(point_world.y()) > 100.0 || std::abs(point_world.z()) > 100.0) {
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            pt.x = static_cast<float>(point_world.x());
            pt.y = static_cast<float>(point_world.y());
            pt.z = static_cast<float>(point_world.z());
            
            // 添加颜色信息
            cv::Vec3b color = rgb_image.at<cv::Vec3b>(v, u);
            pt.r = color[2];  // BGR to RGB
            pt.g = color[1];
            pt.b = color[0];
        }
    }
    
    return cloud;
}

PointCloudBuilder::PointCloudType::Ptr PointCloudBuilder::createSparsePointCloud(
    const std::vector<FeaturePoint>& features,
    const PoseEstimation& pose) {
    
    PointCloudType::Ptr cloud(new PointCloudType());
    cloud->points.reserve(features.size());
    
    for (const auto& feature : features) {
        // 变换到世界坐标系
        Eigen::Vector3d point_world = transformPoint(feature.point_3d, pose.transformation);
        
        PointType pt;
        pt.x = static_cast<float>(point_world.x());
        pt.y = static_cast<float>(point_world.y());
        pt.z = static_cast<float>(point_world.z());
        
        // 默认颜色（可以从RGB图像中获取）
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
        
        cloud->points.push_back(pt);
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    return cloud;
}

void PointCloudBuilder::mergePointCloud(PointCloudType::Ptr target,
                                       PointCloudType::Ptr source,
                                       const Eigen::Matrix4d& transform) {
    if (!source || source->empty()) {
        return;
    }
    
    // 变换源点云
    PointCloudType::Ptr transformed(new PointCloudType());
    pcl::transformPointCloud(*source, *transformed, transform.cast<float>());
    
    // 合并到目标点云
    *target += *transformed;
}

PointCloudBuilder::PointCloudType::Ptr PointCloudBuilder::voxelDownsample(
    PointCloudType::Ptr cloud, double leaf_size) {
    
    if (!cloud || cloud->empty()) {
        return cloud;
    }
    
    PointCloudType::Ptr filtered(new PointCloudType());
    
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*filtered);
    
    LOG_DEBUG("Voxel downsampling: ", cloud->size(), " -> ", filtered->size(), " points");
    
    return filtered;
}

Eigen::Vector3d PointCloudBuilder::transformPoint(const Eigen::Vector3d& point,
                                                  const Eigen::Matrix4d& transform) {
    Eigen::Vector4d point_homo;
    point_homo << point, 1.0;
    
    Eigen::Vector4d transformed_homo = transform * point_homo;
    
    return transformed_homo.head<3>();
}

} // namespace endorobo

