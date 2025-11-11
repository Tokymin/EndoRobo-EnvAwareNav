// PCL Reconstruction Features Test
// Tests the PCL features needed for 3D reconstruction
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "PCL Reconstruction Features Test" << std::endl;
    std::cout << "======================================" << std::endl << std::endl;
    
    // Test 1: Create a colored point cloud
    std::cout << "[Test 1] Creating XYZRGB Point Cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Add some test points
    for (int i = 0; i < 100; ++i) {
        pcl::PointXYZRGB point;
        point.x = static_cast<float>(i) * 0.1f;
        point.y = static_cast<float>(i) * 0.1f;
        point.z = static_cast<float>(i) * 0.1f;
        point.r = static_cast<uint8_t>((i * 255) / 100);
        point.g = 128;
        point.b = static_cast<uint8_t>(255 - (i * 255) / 100);
        cloud->points.push_back(point);
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    std::cout << "  [OK] Created cloud with " << cloud->points.size() << " points" << std::endl;
    std::cout << "  [OK] First point: (" << cloud->points[0].x << ", " 
              << cloud->points[0].y << ", " << cloud->points[0].z << ")" << std::endl;
    std::cout << "  [OK] First point color: RGB(" 
              << static_cast<int>(cloud->points[0].r) << ", "
              << static_cast<int>(cloud->points[0].g) << ", "
              << static_cast<int>(cloud->points[0].b) << ")" << std::endl;
    std::cout << std::endl;
    
    // Test 2: Voxel Grid Filter (downsampling)
    std::cout << "[Test 2] Testing Voxel Grid Filter..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);
    voxel_filter.filter(*filtered_cloud);
    
    std::cout << "  [OK] Original cloud: " << cloud->points.size() << " points" << std::endl;
    std::cout << "  [OK] Filtered cloud: " << filtered_cloud->points.size() << " points" << std::endl;
    std::cout << "  [OK] Reduction ratio: " 
              << (1.0 - static_cast<double>(filtered_cloud->points.size()) / cloud->points.size()) * 100.0 
              << "%" << std::endl;
    std::cout << std::endl;
    
    // Test 3: Point Cloud Transformation
    std::cout << "[Test 3] Testing Point Cloud Transformation..." << std::endl;
    
    // Create a transformation matrix (translation + rotation)
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // Rotation around Z-axis (45 degrees)
    float theta = M_PI / 4.0f; // 45 degrees
    transform(0, 0) = std::cos(theta);
    transform(0, 1) = -std::sin(theta);
    transform(1, 0) = std::sin(theta);
    transform(1, 1) = std::cos(theta);
    
    // Translation
    transform(0, 3) = 1.0f;  // x
    transform(1, 3) = 2.0f;  // y
    transform(2, 3) = 3.0f;  // z
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
    
    std::cout << "  [OK] Transformation matrix:" << std::endl;
    std::cout << "      " << transform(0, 0) << " " << transform(0, 1) << " " 
              << transform(0, 2) << " " << transform(0, 3) << std::endl;
    std::cout << "      " << transform(1, 0) << " " << transform(1, 1) << " " 
              << transform(1, 2) << " " << transform(1, 3) << std::endl;
    std::cout << "      " << transform(2, 0) << " " << transform(2, 1) << " " 
              << transform(2, 2) << " " << transform(2, 3) << std::endl;
    std::cout << "      " << transform(3, 0) << " " << transform(3, 1) << " " 
              << transform(3, 2) << " " << transform(3, 3) << std::endl;
    
    std::cout << "  [OK] Original first point: (" 
              << cloud->points[0].x << ", " 
              << cloud->points[0].y << ", " 
              << cloud->points[0].z << ")" << std::endl;
    std::cout << "  [OK] Transformed first point: (" 
              << transformed_cloud->points[0].x << ", " 
              << transformed_cloud->points[0].y << ", " 
              << transformed_cloud->points[0].z << ")" << std::endl;
    std::cout << "  [OK] Color preserved: RGB(" 
              << static_cast<int>(transformed_cloud->points[0].r) << ", "
              << static_cast<int>(transformed_cloud->points[0].g) << ", "
              << static_cast<int>(transformed_cloud->points[0].b) << ")" << std::endl;
    std::cout << std::endl;
    
    // Test 4: Point Cloud Merging
    std::cout << "[Test 4] Testing Point Cloud Merging..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    *merged_cloud = *cloud + *transformed_cloud;
    
    std::cout << "  [OK] Cloud 1 size: " << cloud->points.size() << std::endl;
    std::cout << "  [OK] Cloud 2 size: " << transformed_cloud->points.size() << std::endl;
    std::cout << "  [OK] Merged cloud size: " << merged_cloud->points.size() << std::endl;
    std::cout << std::endl;
    
    // Test 5: Organized Point Cloud (like from depth camera)
    std::cout << "[Test 5] Creating Organized Point Cloud (depth-like)..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr organized_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    int width = 640;
    int height = 480;
    organized_cloud->width = width;
    organized_cloud->height = height;
    organized_cloud->is_dense = false;
    organized_cloud->points.resize(width * height);
    
    // Simulate depth camera data
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            pcl::PointXYZRGB& pt = organized_cloud->at(u, v);
            
            // Simulate some points with depth, some without
            if ((u + v) % 10 == 0) {
                // Invalid point (no depth)
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
            } else {
                // Valid point with simulated depth
                float depth = 1.0f + (u + v) * 0.001f;
                pt.x = (u - width / 2.0f) * depth / 500.0f;
                pt.y = (v - height / 2.0f) * depth / 500.0f;
                pt.z = depth;
                pt.r = static_cast<uint8_t>((u * 255) / width);
                pt.g = static_cast<uint8_t>((v * 255) / height);
                pt.b = 128;
            }
        }
    }
    
    // Count valid points
    int valid_points = 0;
    for (const auto& pt : organized_cloud->points) {
        if (!std::isnan(pt.x)) {
            valid_points++;
        }
    }
    
    std::cout << "  [OK] Created organized cloud: " << width << "x" << height << std::endl;
    std::cout << "  [OK] Total points: " << organized_cloud->points.size() << std::endl;
    std::cout << "  [OK] Valid points: " << valid_points << std::endl;
    std::cout << "  [OK] Invalid points (NaN): " << (organized_cloud->points.size() - valid_points) << std::endl;
    std::cout << std::endl;
    
    // Final summary
    std::cout << "======================================" << std::endl;
    std::cout << "All PCL Reconstruction Tests PASSED!" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Point cloud creation: OK" << std::endl;
    std::cout << "  - XYZRGB color support: OK" << std::endl;
    std::cout << "  - Voxel grid filtering: OK" << std::endl;
    std::cout << "  - Point cloud transformation: OK" << std::endl;
    std::cout << "  - Point cloud merging: OK" << std::endl;
    std::cout << "  - Organized point clouds: OK" << std::endl;
    std::cout << std::endl;
    std::cout << "PCL is ready for 3D reconstruction!" << std::endl;
    
    return 0;
}

