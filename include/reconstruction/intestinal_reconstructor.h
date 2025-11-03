#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include <vector>
#include "core/config_manager.h"
#include "reconstruction/point_cloud_builder.h"

namespace endorobo {

/**
 * @brief 肠腔重建器类
 * 专门针对管状肠腔结构的3D重建
 */
class IntestinalReconstructor {
public:
    using PointType = pcl::PointXYZRGB;
    using PointCloudType = pcl::PointCloud<PointType>;
    
    /**
     * @brief 构造函数
     * @param config 重建配置
     */
    explicit IntestinalReconstructor(const ReconstructionConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~IntestinalReconstructor();
    
    /**
     * @brief 初始化重建器
     * @return 是否初始化成功
     */
    bool initialize();
    
    /**
     * @brief 添加新的点云帧
     * @param cloud 新点云
     * @param pose 对应的位姿
     * @return 是否添加成功
     */
    bool addFrame(PointCloudType::Ptr cloud, const PoseEstimation& pose);
    
    /**
     * @brief 获取累积的点云
     * @return 累积点云
     */
    PointCloudType::Ptr getAccumulatedCloud() const { return accumulated_cloud_; }
    
    /**
     * @brief 获取处理后的点云（去除冗余点）
     * @return 处理后的点云
     */
    PointCloudType::Ptr getProcessedCloud();
    
    /**
     * @brief 表面重建
     * @return 重建的网格表面
     */
    pcl::PolygonMeshPtr reconstructSurface();
    
    /**
     * @brief 平滑点云
     * @param cloud 输入点云
     * @return 平滑后的点云
     */
    PointCloudType::Ptr smoothCloud(PointCloudType::Ptr cloud);
    
    /**
     * @brief 估计管状结构的中心线
     * @return 中心线点集
     */
    std::vector<Eigen::Vector3d> estimateCenterline();
    
    /**
     * @brief 重置重建器
     */
    void reset();
    
    /**
     * @brief 获取累积帧数
     */
    int getFrameCount() const { return frame_count_; }
    
private:
    ReconstructionConfig config_;
    PointCloudType::Ptr accumulated_cloud_;
    int frame_count_;
    
    /**
     * @brief 过滤管状结构外的点
     * @param cloud 输入点云
     * @param centerline 中心线
     * @return 过滤后的点云
     */
    PointCloudType::Ptr filterByTubularStructure(
        PointCloudType::Ptr cloud,
        const std::vector<Eigen::Vector3d>& centerline);
    
    /**
     * @brief 计算点云法向量
     */
    void computeNormals(PointCloudType::Ptr cloud,
                       pcl::PointCloud<pcl::Normal>::Ptr normals);
};

} // namespace endorobo

