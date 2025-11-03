#include "reconstruction/intestinal_reconstructor.h"
#include "core/logger.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace endorobo {

IntestinalReconstructor::IntestinalReconstructor(const ReconstructionConfig& config)
    : config_(config)
    , frame_count_(0) {
    accumulated_cloud_.reset(new PointCloudType());
}

IntestinalReconstructor::~IntestinalReconstructor() {
}

bool IntestinalReconstructor::initialize() {
    LOG_INFO("Initializing intestinal reconstructor...");
    LOG_INFO("Voxel size: ", config_.voxel_size);
    LOG_INFO("Radius range: [", config_.intestinal.min_radius, ", ", 
             config_.intestinal.max_radius, "]");
    LOG_INFO("Smoothing iterations: ", config_.intestinal.smoothing_iterations);
    return true;
}

bool IntestinalReconstructor::addFrame(PointCloudType::Ptr cloud, const PoseEstimation& pose) {
    if (!cloud || cloud->empty()) {
        LOG_WARNING("Empty cloud, skipping frame");
        return false;
    }
    
    if (!pose.valid) {
        LOG_WARNING("Invalid pose, skipping frame");
        return false;
    }
    
    // 合并到累积点云
    *accumulated_cloud_ += *cloud;
    frame_count_++;
    
    LOG_DEBUG("Frame ", frame_count_, " added. Total points: ", accumulated_cloud_->size());
    
    // 定期进行体素下采样以控制点云大小
    if (frame_count_ % 10 == 0) {
        PointCloudBuilder builder;
        accumulated_cloud_ = builder.voxelDownsample(accumulated_cloud_, config_.voxel_size);
        LOG_INFO("Downsampled to ", accumulated_cloud_->size(), " points");
    }
    
    return true;
}

PointCloudBuilder::PointCloudType::Ptr IntestinalReconstructor::getProcessedCloud() {
    if (accumulated_cloud_->empty()) {
        LOG_WARNING("No accumulated cloud data");
        return accumulated_cloud_;
    }
    
    LOG_INFO("Processing accumulated cloud...");
    
    // 1. 统计离群点去除
    PointCloudType::Ptr filtered(new PointCloudType());
    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(accumulated_cloud_);
    sor.setMeanK(config_.redundancy_removal.nb_neighbors);
    sor.setStddevMulThresh(config_.redundancy_removal.std_ratio);
    sor.filter(*filtered);
    
    LOG_INFO("After statistical outlier removal: ", filtered->size(), " points");
    
    // 2. 平滑
    PointCloudType::Ptr smoothed = smoothCloud(filtered);
    
    LOG_INFO("Processing complete. Final points: ", smoothed->size());
    
    return smoothed;
}

PointCloudBuilder::PointCloudType::Ptr IntestinalReconstructor::smoothCloud(
    PointCloudType::Ptr cloud) {
    
    if (cloud->empty()) {
        return cloud;
    }
    
    // 使用移动最小二乘法(MLS)进行平滑
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    
    // MLS平滑
    pcl::PointCloud<PointType>::Ptr smoothed(new pcl::PointCloud<PointType>());
    pcl::MovingLeastSquares<PointType, PointType> mls;
    mls.setInputCloud(cloud);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(config_.voxel_size * 3.0);
    mls.setPolynomialOrder(2);
    mls.setComputeNormals(false);
    
    // 执行平滑
    try {
        mls.process(*smoothed);
        LOG_INFO("Cloud smoothed successfully");
        return smoothed;
    } catch (const std::exception& e) {
        LOG_ERROR("MLS smoothing failed: ", e.what());
        return cloud;
    }
}

pcl::PolygonMeshPtr IntestinalReconstructor::reconstructSurface() {
    pcl::PolygonMeshPtr mesh(new pcl::PolygonMesh());
    
    if (accumulated_cloud_->empty()) {
        LOG_ERROR("Cannot reconstruct surface from empty cloud");
        return mesh;
    }
    
    LOG_INFO("Reconstructing surface from ", accumulated_cloud_->size(), " points...");
    
    // 获取处理后的点云
    PointCloudType::Ptr cloud = getProcessedCloud();
    
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    computeNormals(cloud, normals);
    
    // 合并点和法向量
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    
    // 使用贪婪投影三角化
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    
    gp3.setSearchRadius(config_.voxel_size * 5.0);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45度
    gp3.setMinimumAngle(M_PI / 18);        // 10度
    gp3.setMaximumAngle(2 * M_PI / 3);     // 120度
    gp3.setNormalConsistency(false);
    
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*mesh);
    
    LOG_INFO("Surface reconstructed: ", mesh->polygons.size(), " polygons");
    
    return mesh;
}

void IntestinalReconstructor::computeNormals(PointCloudType::Ptr cloud,
                                            pcl::PointCloud<pcl::Normal>::Ptr normals) {
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);  // 使用20个邻近点估计法向量
    ne.compute(*normals);
    
    LOG_DEBUG("Normals computed for ", normals->size(), " points");
}

std::vector<Eigen::Vector3d> IntestinalReconstructor::estimateCenterline() {
    std::vector<Eigen::Vector3d> centerline;
    
    if (accumulated_cloud_->empty()) {
        return centerline;
    }
    
    // 简化版本：沿着主轴方向采样点的中心
    // 更复杂的实现可以使用骨架提取算法
    
    // 计算点云的边界
    PointType min_pt, max_pt;
    pcl::getMinMax3D(*accumulated_cloud_, min_pt, max_pt);
    
    // 沿着主轴（假设为Z轴）分段
    double z_step = config_.voxel_size * 10.0;
    for (double z = min_pt.z; z <= max_pt.z; z += z_step) {
        // 收集该层的点
        std::vector<Eigen::Vector3d> layer_points;
        for (const auto& pt : accumulated_cloud_->points) {
            if (std::abs(pt.z - z) < z_step / 2.0) {
                layer_points.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
            }
        }
        
        // 计算该层的中心
        if (!layer_points.empty()) {
            Eigen::Vector3d center = Eigen::Vector3d::Zero();
            for (const auto& pt : layer_points) {
                center += pt;
            }
            center /= layer_points.size();
            centerline.push_back(center);
        }
    }
    
    LOG_INFO("Estimated centerline with ", centerline.size(), " points");
    
    return centerline;
}

void IntestinalReconstructor::reset() {
    accumulated_cloud_->clear();
    frame_count_ = 0;
    LOG_INFO("Reconstructor reset");
}

} // namespace endorobo

