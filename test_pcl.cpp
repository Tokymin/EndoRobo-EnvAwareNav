#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PCL Library Test Program" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Create a simple point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Add some test points
        for (int i = 0; i < 10; ++i) {
            pcl::PointXYZ point;
            point.x = static_cast<float>(i);
            point.y = static_cast<float>(i * 2);
            point.z = static_cast<float>(i * 3);
            cloud->points.push_back(point);
        }
        
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = false;
        
        std::cout << "\n[OK] Successfully created point cloud" << std::endl;
        std::cout << "  Number of points: " << cloud->points.size() << std::endl;
        
        // Calculate point cloud bounding box
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cloud, minPt, maxPt);
        
        std::cout << "\n[OK] Successfully calculated bounding box" << std::endl;
        std::cout << "  Min point: (" << minPt.x << ", " << minPt.y << ", " << minPt.z << ")" << std::endl;
        std::cout << "  Max point: (" << maxPt.x << ", " << maxPt.y << ", " << maxPt.z << ")" << std::endl;
        
        // Try to save point cloud (optional, if directory is writable)
        std::string filename = "test_pcl_output.pcd";
        if (pcl::io::savePCDFileASCII(filename, *cloud) == 0) {
            std::cout << "\n[OK] Successfully saved point cloud to file: " << filename << std::endl;
        } else {
            std::cout << "\n[WARNING] Unable to save point cloud file (may not have write permission)" << std::endl;
        }
        
        // Test PCL version info
        std::cout << "\n[OK] PCL Library Information:" << std::endl;
        std::cout << "  PCL library is successfully linked and working" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "PCL Library Test PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "\n[ERROR] Unknown error" << std::endl;
        return 1;
    }
}

