#pragma once

// Include logger.h first to prevent PCL namespace pollution
#include "core/logger.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include "core/config_manager.h"

namespace endorobo {

/**
 * @brief 冗余点去除器类
 * 负责去除点云中的冗余点和离群点
 */
class RedundancyRemover {
public:
    using PointType = pcl::PointXYZRGB;
    using PointCloudType = pcl::PointCloud<PointType>;
    
    /**
     * @brief 构造函数
     * @param config 冗余去除配置
     */
    explicit RedundancyRemover(const RedundancyRemovalConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~RedundancyRemover();
    
    /**
     * @brief 去除冗余点
     * @param input 输入点云
     * @param output 输出点云
     * @return 去除的点数
     */
    int removeRedundancy(PointCloudType::Ptr input, PointCloudType::Ptr out_cloud);
    
    /**
     * @brief 统计离群点去除
     * @param input 输入点云
     * @param output 输出点云
     * @return 去除的点数
     */
    int removeStatisticalOutliers(PointCloudType::Ptr input, PointCloudType::Ptr out_cloud);
    
    /**
     * @brief 半径离群点去除
     * @param input 输入点云
     * @param output 输出点云
     * @param radius 搜索半径
     * @param min_neighbors 最小邻居数
     * @return 去除的点数
     */
    int removeRadiusOutliers(PointCloudType::Ptr input,
                            PointCloudType::Ptr out_cloud,
                            double radius,
                            int min_neighbors);
    
    /**
     * @brief 基于距离的重复点去除
     * @param input 输入点云
     * @param output 输出点云
     * @return 去除的点数
     */
    int removeDuplicatePoints(PointCloudType::Ptr input, PointCloudType::Ptr out_cloud);
    
    /**
     * @brief 基于法向量一致性的点去除
     * @param input 输入点云
     * @param output 输出点云
     * @return 去除的点数
     */
    int removeByNormalConsistency(PointCloudType::Ptr input, PointCloudType::Ptr out_cloud);
    
    /**
     * @brief 自适应体素滤波
     * @param input 输入点云
     * @param output 输出点云
     * @return 去除的点数
     */
    int adaptiveVoxelFilter(PointCloudType::Ptr input, PointCloudType::Ptr out_cloud);
    
    /**
     * @brief 完整的冗余去除流程
     * @param input 输入点云
     * @return 处理后的点云
     */
    PointCloudType::Ptr process(PointCloudType::Ptr input);
    
private:
    RedundancyRemovalConfig config_;
    
    /**
     * @brief 计算点云的局部密度
     */
    std::vector<double> computeLocalDensity(PointCloudType::Ptr cloud);
    
    /**
     * @brief 计算两点间的距离
     */
    inline double distance(const PointType& p1, const PointType& p2) const {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

} // namespace endorobo

