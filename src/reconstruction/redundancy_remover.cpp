#include "reconstruction/redundancy_remover.h"
#include "core/logger.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <unordered_set>

namespace endorobo {

RedundancyRemover::RedundancyRemover(const RedundancyRemovalConfig& config)
    : config_(config) {
}

RedundancyRemover::~RedundancyRemover() {
}

RedundancyRemover::PointCloudType::Ptr RedundancyRemover::process(PointCloudType::Ptr input) {
    if (!input || input->empty()) {
        LOG_WARNING("Input cloud is empty");
        return input;
    }
    
    LOG_INFO("Starting redundancy removal pipeline...");
    LOG_INFO("Input cloud size: ", input->size());
    
    PointCloudType::Ptr temp1(new PointCloudType());
    PointCloudType::Ptr temp2(new PointCloudType());
    PointCloudType::Ptr output(new PointCloudType());
    
    // 1. 统计离群点去除
    int removed = removeStatisticalOutliers(input, temp1);
    LOG_INFO("Statistical outlier removal: removed ", removed, " points");
    
    // 2. 重复点去除
    removed = removeDuplicatePoints(temp1, temp2);
    LOG_INFO("Duplicate points removal: removed ", removed, " points");
    
    // 3. 半径离群点去除
    removed = removeRadiusOutliers(temp2, temp1, config_.distance_threshold * 2.0, 5);
    LOG_INFO("Radius outlier removal: removed ", removed, " points");
    
    // 4. 法向量一致性检查
    removed = removeByNormalConsistency(temp1, output);
    LOG_INFO("Normal consistency check: removed ", removed, " points");
    
    LOG_INFO("Redundancy removal complete. Output cloud size: ", output->size());
    LOG_INFO("Total removed: ", input->size() - output->size(), " points (",
             100.0 * (input->size() - output->size()) / input->size(), "%)");
    
    return output;
}

int RedundancyRemover::removeRedundancy(PointCloudType::Ptr input, PointCloudType::Ptr output) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    output = process(input);
    return original_size - output->size();
}

int RedundancyRemover::removeStatisticalOutliers(PointCloudType::Ptr input,
                                                 PointCloudType::Ptr output) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    
    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(input);
    sor.setMeanK(config_.nb_neighbors);
    sor.setStddevMulThresh(config_.std_ratio);
    sor.filter(*output);
    
    return original_size - output->size();
}

int RedundancyRemover::removeRadiusOutliers(PointCloudType::Ptr input,
                                           PointCloudType::Ptr output,
                                           double radius,
                                           int min_neighbors) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    
    pcl::RadiusOutlierRemoval<PointType> ror;
    ror.setInputCloud(input);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(min_neighbors);
    ror.filter(*output);
    
    return original_size - output->size();
}

int RedundancyRemover::removeDuplicatePoints(PointCloudType::Ptr input,
                                            PointCloudType::Ptr output) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    output->clear();
    
    // 使用KD树进行快速邻近搜索
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(input);
    
    std::vector<bool> processed(input->size(), false);
    double threshold_sq = config_.distance_threshold * config_.distance_threshold;
    
    for (size_t i = 0; i < input->size(); ++i) {
        if (processed[i]) continue;
        
        const PointType& point = input->points[i];
        output->points.push_back(point);
        processed[i] = true;
        
        // 查找邻近点
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(point, config_.distance_threshold, indices, distances);
        
        // 标记所有邻近点为已处理
        for (const int idx : indices) {
            if (idx != static_cast<int>(i)) {
                processed[idx] = true;
            }
        }
    }
    
    output->width = output->size();
    output->height = 1;
    output->is_dense = true;
    
    return original_size - output->size();
}

int RedundancyRemover::removeByNormalConsistency(PointCloudType::Ptr input,
                                                PointCloudType::Ptr output) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    
    ne.setInputCloud(input);
    ne.setSearchMethod(tree);
    ne.setKSearch(config_.nb_neighbors);
    ne.compute(*normals);
    
    // 检查法向量一致性
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(input);
    
    double angle_threshold_rad = config_.normal_angle_threshold * M_PI / 180.0;
    double cos_threshold = std::cos(angle_threshold_rad);
    
    output->clear();
    for (size_t i = 0; i < input->size(); ++i) {
        const PointType& point = input->points[i];
        const pcl::Normal& normal = normals->points[i];
        
        // 跳过无效法向量
        if (!std::isfinite(normal.normal_x) ||
            !std::isfinite(normal.normal_y) ||
            !std::isfinite(normal.normal_z)) {
            continue;
        }
        
        // 查找邻近点
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(point, config_.distance_threshold * 3.0, indices, distances);
        
        // 检查与邻近点的法向量一致性
        int consistent_neighbors = 0;
        for (const int idx : indices) {
            if (idx == static_cast<int>(i)) continue;
            
            const pcl::Normal& neighbor_normal = normals->points[idx];
            
            // 计算法向量夹角的余弦值
            double dot_product = 
                normal.normal_x * neighbor_normal.normal_x +
                normal.normal_y * neighbor_normal.normal_y +
                normal.normal_z * neighbor_normal.normal_z;
            
            if (std::abs(dot_product) > cos_threshold) {
                consistent_neighbors++;
            }
        }
        
        // 如果有足够多的一致邻居，保留该点
        if (consistent_neighbors >= static_cast<int>(indices.size()) / 2) {
            output->points.push_back(point);
        }
    }
    
    output->width = output->size();
    output->height = 1;
    output->is_dense = true;
    
    return original_size - output->size();
}

int RedundancyRemover::adaptiveVoxelFilter(PointCloudType::Ptr input,
                                          PointCloudType::Ptr output) {
    if (!input || input->empty()) {
        return 0;
    }
    
    int original_size = input->size();
    
    // 计算局部密度
    std::vector<double> densities = computeLocalDensity(input);
    
    // 根据密度自适应调整体素大小
    // 高密度区域使用较大的体素，低密度区域使用较小的体素
    double avg_density = 0.0;
    for (double d : densities) {
        avg_density += d;
    }
    avg_density /= densities.size();
    
    // 简化实现：使用固定体素大小
    pcl::VoxelGrid<PointType> vg;
    vg.setInputCloud(input);
    vg.setLeafSize(config_.distance_threshold, 
                   config_.distance_threshold, 
                   config_.distance_threshold);
    vg.filter(*output);
    
    return original_size - output->size();
}

std::vector<double> RedundancyRemover::computeLocalDensity(PointCloudType::Ptr cloud) {
    std::vector<double> densities(cloud->size());
    
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(cloud);
    
    double search_radius = config_.distance_threshold * 5.0;
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;
        
        kdtree.radiusSearch(cloud->points[i], search_radius, indices, distances);
        
        // 密度 = 邻近点数 / 搜索体积
        double volume = (4.0 / 3.0) * M_PI * std::pow(search_radius, 3);
        densities[i] = indices.size() / volume;
    }
    
    return densities;
}

} // namespace endorobo

